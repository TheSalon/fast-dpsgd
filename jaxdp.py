'''
Code for JAX implementations presented in: Enabling Fast
Differentially Private SGD via Just-in-Time Compilation and Vectorization
'''

import itertools
import time
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap
from jax.experimental import optimizers, stax
from jax.lib import pytree
from jax.tree_util import tree_flatten, tree_multimap, tree_unflatten
from keras.utils.np_utils import to_categorical
from tensorflow_privacy.privacy.analysis.rdp_accountant import (compute_rdp, get_privacy_spent)

import data
import utils


def logistic_model(features, **_):
    return hk.Sequential([hk.Linear(1), jax.nn.sigmoid])(features)


def ffnn_model(features, **_):
    return hk.Sequential([hk.Linear(50), jax.nn.relu, hk.Linear(2)])(features)


def mnist_model(features, **_):
    return hk.Sequential([
        hk.Conv2D(16, (8, 8), padding='SAME', stride=(2, 2)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),  # matches stax
        hk.Conv2D(32, (4, 4), padding='VALID', stride=(2, 2)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),  # matches stax
        hk.Flatten(),
        hk.Linear(32),
        jax.nn.relu,
        hk.Linear(10),
    ])(features)


def lstm_model(x, vocab_size=10_000, seq_len=256, args=None, **_):
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size + 4, 100, w_init=embed_init)
    o2 = token_embedding_map(x)

    o2 = jnp.reshape(o2, (o2.shape[1], o2.shape[0], o2.shape[2]))

    # LSTM Part of Network
    core = hk.LSTM(100)
    if args and args.dynamic_unroll:
        outs, state = hk.dynamic_unroll(core, o2, core.initial_state(x.shape[0]))
    else:
        outs, state = hk.static_unroll(core, o2, core.initial_state(x.shape[0]))
    outs = outs.reshape(outs.shape[1], outs.shape[0], outs.shape[2])

    # Avg Pool -> Linear
    red_dim_outs = hk.avg_pool(outs, seq_len, seq_len, "SAME").squeeze()
    final_layer = hk.Linear(2)
    ret = final_layer(red_dim_outs)

    return ret


def embedding_model(arr, vocab_size=10_000, seq_len=256, **_):
    # embedding part of network
    x = arr
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size + 4, 16, w_init=embed_init)
    o2 = token_embedding_map(x)

    # avg pool -> linear
    o3 = hk.avg_pool(o2, seq_len, seq_len, "SAME").squeeze()
    fcnn = hk.Sequential([hk.Linear(16), jax.nn.relu, hk.Linear(2)])
    return fcnn(o3)


def cifar_model(features, **_):
    out = hk.Conv2D(32, (3, 3), padding='SAME', stride=(1, 1))(features)
    out = jax.nn.relu(out)
    out = hk.Conv2D(32, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.AvgPool(2, strides=2, padding='VALID')(out)

    out = hk.Conv2D(64, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.Conv2D(64, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.AvgPool(2, strides=2, padding='VALID')(out)

    out = hk.Conv2D(128, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.Conv2D(128, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.AvgPool(2, strides=2, padding='VALID')(out)

    out = hk.Conv2D(256, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.Conv2D(10, (3, 3), padding='SAME', stride=(1, 1))(out)

    return out.mean((1, 2))


def multiclass_loss(model, params, batch):
    inputs, targets = batch
    logits = model.apply(params, None, inputs)
    # convert the outputs to one hot shape according to the same shape as
    # logits for vectorized dot product
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    logits = stax.logsoftmax(logits)  # log normalize
    return -jnp.mean(jnp.sum(logits * one_hot, axis=-1))  # cross entropy loss


def logistic_loss(model, params, batch):
    inputs, targets = batch[0], batch[1]
    # have to always supply the RNG field
    logits = model.apply(params, None, inputs)
    logits = jnp.reshape(logits, -1)  # needs to be only scalar per index
    # max_val is required for numerical stability
    max_val = jnp.clip(logits, 0, None)
    loss = jnp.mean(logits - logits * targets + max_val +
                    jnp.log(jnp.exp(-max_val) + jnp.exp((-logits - max_val))))

    return loss


def accuracy(model, params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(model.apply(params, None, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


def clipped_grad(model, loss, params, l2_norm_clip, single_example_batch):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(partial(loss, model))(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.clip(total_grad_norm / l2_norm_clip, a_max=1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads)


def private_grad(model, loss, params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
    """Return differentially private gradients for params, evaluated on batch."""
    clipped_grads = vmap(partial(clipped_grad, model, loss), (None, None, 0))(params, l2_norm_clip,
                                                                              batch)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)
    ]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads
    ]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)


def private_grad_no_vmap(model, loss, params, batch, rng, l2_norm_clip, noise_multiplier,
                         batch_size):
    """Return differentially private gradients for params, evaluated on batch."""
    clipped_grads = tree_multimap(
        lambda *xs: jnp.stack(xs),
        *(clipped_grad(model, loss, params, l2_norm_clip, eg) for eg in zip(*batch)))

    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)
    ]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads
    ]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)


model_dict = {
    'mnist': mnist_model,
    'lstm': lstm_model,
    'embed': embedding_model,
    'ffnn': ffnn_model,
    'logreg': logistic_model,
    'cifar10': cifar_model,
}


def main(args):
    print(args)
    if args.microbatches:
        raise NotImplementedError('Microbatches < batch size not currently supported')
    if args.experiment == 'lstm' and args.no_jit:
        raise ValueError('LSTM with no JIT will fail.')

    data_fn = data.data_fn_dict[args.experiment][int(args.dummy_data)]
    kwargs = {
        'max_features': args.max_features,
        'max_len': args.max_len,
        'format': 'NHWC',
    }
    if args.dummy_data:
        kwargs['num_examples'] = args.batch_size * 2
    (train_data, train_labels), _ = data_fn(**kwargs)
    # train_labels, test_labels = to_categorical(train_labels), to_categorical(
    #     test_labels)

    num_train = train_data.shape[0]
    num_complete_batches, leftover = divmod(num_train, args.batch_size)
    num_batches = num_complete_batches + bool(leftover)
    key = random.PRNGKey(args.seed)

    model = hk.transform(
        partial(model_dict[args.experiment],
                args=args,
                vocab_size=args.max_features,
                seq_len=args.max_len))
    rng = jax.random.PRNGKey(42)
    init_params = model.init(key, train_data[:args.batch_size])
    opt_init, opt_update, get_params = optimizers.sgd(args.learning_rate)
    loss = logistic_loss if args.experiment == 'logreg' else multiclass_loss

    if args.dpsgd:
        train_data, train_labels = train_data[:, None], train_labels[:, None]

    # regular update -- non-private
    def update(_, i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(partial(loss, model))(params, batch), opt_state)

    grad_fn = private_grad_no_vmap if args.no_vmap else private_grad

    # differentially private update
    def private_update(rng, i, opt_state, batch):
        params = get_params(opt_state)
        rng = random.fold_in(rng, i)  # get new key for new random numbers
        return opt_update(
            i,
            grad_fn(model, loss, params, batch, rng, args.l2_norm_clip, args.noise_multiplier,
                    args.batch_size), opt_state)

    opt_state = opt_init(init_params)
    itercount = itertools.count()
    train_fn = private_update if args.dpsgd else update

    if args.no_vmap:
        print('No vmap for dpsgd!')

    if not args.no_jit:
        train_fn = jit(train_fn)
    else:
        print('No jit!')

    dummy = jnp.array(1.)

    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        for i, batch in enumerate(data.dataloader(train_data, train_labels, args.batch_size)):
            opt_state = train_fn(
                key,
                next(itercount),
                opt_state,
                batch,
            )
        (dummy * dummy).block_until_ready()  # synchronize CUDA.
        duration = time.perf_counter() - start
        print("Time Taken: ", duration)
        timings.append(duration)

        if args.dpsgd:
            print('Trained with DP SGD optimizer')
        else:
            print('Trained with vanilla non-private SGD optimizer')

    if not args.no_save:
        append_to_name = ''
        if args.no_jit: append_to_name += '_nojit'
        if args.no_vmap: append_to_name += '_novmap'
        utils.save_runtimes(__file__.split('.')[0], args, timings, append_to_name)
    else:
        print('Not saving!')
    print('Done!')


if __name__ == '__main__':
    parser = utils.get_parser(model_dict.keys())
    parser.add_argument('--no_vmap', dest='no_vmap', action='store_true')
    parser.add_argument('--no_jit', dest='no_jit', action='store_true')
    parser.add_argument('--dynamic_unroll', dest='dynamic_unroll', action='store_true')
    args = parser.parse_args()
    main(args)
