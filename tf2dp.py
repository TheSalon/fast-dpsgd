import time
from functools import partial

import tensorflow as tf
from tensorflow_privacy.privacy.analysis.gdp_accountant import (compute_eps_poisson,
                                                                compute_mu_poisson)
from jax.tree_util import tree_multimap

import data
import utils


def get_logreg_model(features, batch_size=None, **_):
    return tf.keras.Sequential(
        [tf.keras.Input(shape=features.shape[1:], batch_size=batch_size),
         tf.keras.layers.Dense(1)])


def get_ffnn_model(features, batch_size=None, **_):
    return tf.keras.Sequential([
        tf.keras.Input(shape=features.shape[1:], batch_size=batch_size),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2, activation='relu')
    ])


def get_mnist_model(features, batch_size=None, **_):
    return tf.keras.Sequential([
        tf.keras.Input(shape=features.shape[1:], batch_size=batch_size),
        tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])


def get_imdb_model(features, max_features, args, batch_size=None, **_):
    return tf.keras.Sequential([
        tf.keras.Input(shape=features.shape[1:], batch_size=batch_size),
        tf.keras.layers.Embedding(max_features + 4, 100),
        tf.keras.layers.LSTM(100, return_sequences=True, unroll=(not args.no_unroll)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2)
    ])


def get_embedding_model(features, max_features, batch_size=None, **_):
    return tf.keras.Sequential([
        tf.keras.Input(shape=features.shape[1:], batch_size=batch_size),
        tf.keras.layers.Embedding(max_features + 4, 16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2)
    ])


class CIFAR10Model(tf.keras.Model):
    def __init__(self, features, batch_size=None, **_):
        super().__init__()
        layers = tf.keras.layers
        self.layer_list = [
            layers.Conv2D(32, (3, 3), padding='SAME', strides=(1, 1), activation='relu'),
            layers.Conv2D(32, (3, 3), padding='SAME', strides=(1, 1), activation='relu'),
            layers.AveragePooling2D(2, strides=2, padding='VALID'),
            layers.Conv2D(64, (3, 3), padding='SAME', strides=(1, 1), activation='relu'),
            layers.Conv2D(64, (3, 3), padding='SAME', strides=(1, 1), activation='relu'),
            layers.AveragePooling2D(2, strides=2, padding='VALID'),
            layers.Conv2D(128, (3, 3), padding='SAME', strides=(1, 1), activation='relu'),
            layers.Conv2D(128, (3, 3), padding='SAME', strides=(1, 1), activation='relu'),
            layers.AveragePooling2D(2, strides=2, padding='VALID'),
            layers.Conv2D(256, (3, 3), padding='SAME', strides=(1, 1), activation='relu'),
            layers.Conv2D(10, (3, 3), padding='SAME', strides=(1, 1)),
        ]

    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
            # print(x.shape)
        return tf.reduce_mean(x, axis=(1, 2))


def reduce_noise_normalize_batch(args, stacked_grads):
    summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
    noise_stddev = args.l2_norm_clip * args.noise_multiplier
    noise = tf.random.normal(tf.shape(input=summed_grads), stddev=noise_stddev)
    noised_grads = summed_grads + noise
    return noised_grads / tf.cast(args.microbatches, tf.float32)


def compute_per_eg_grad(model, optimizer, loss_fn, args, data):
    features, labels = data
    with tf.GradientTape() as tape:
        # We need to add the extra dimension to features because model
        # expects batched input.
        logits = model(features[None])
        loss = loss_fn(labels=labels, logits=tf.squeeze(logits))
    grads_list = tape.gradient(
        loss,
        model.trainable_variables,
        # This argument should not be necessary, but we include it in case:
        unconnected_gradients=tf.UnconnectedGradients.ZERO)

    # We expect grads_list to be flat already, but we use this structure to mirror TFP.
    grads_flat = tf.nest.flatten(grads_list)
    squared_l2_norms = [tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat]
    global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
    div = tf.maximum(global_norm / args.l2_norm_clip, 1.)
    clipped_flat = [g / div for g in grads_flat]
    clipped_grads = tf.nest.pack_sequence_as(grads_list, clipped_flat)

    return loss, clipped_grads


def private_train_step(model, optimizer, loss_fn, args, data):
    if args.no_vmap:
        x, y = data
        # Manually compute per-example gradients via a loop, then stack the results.
        loss, clipped_grads = tree_multimap(
            lambda *xs: tf.stack(xs),
            *(compute_per_eg_grad(model, optimizer, loss_fn, args, (x[i], y[i]))
              for i in range(x.shape[0])))
    else:
        loss, clipped_grads = tf.vectorized_map(
            partial(compute_per_eg_grad, model, optimizer, loss_fn, args),
            data)  # , fallback_to_while_loop=False)
    final_grads = tf.nest.map_structure(partial(reduce_noise_normalize_batch, args), clipped_grads)
    optimizer.apply_gradients(zip(final_grads, model.trainable_variables))
    return loss


def train_step(model, optimizer, loss_fn, args, data):
    features, labels = data
    with tf.GradientTape() as tape:
        logits = model(features)
        loss = tf.reduce_mean(loss_fn(labels=labels, logits=logits))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def evaluate(model, test_data, test_labels):
    # This function is unused.
    loss_mean = tf.keras.metrics.Mean()
    acc_mean = tf.keras.metrics.SparseCategoricalAccuracy()
    for features, labels in zip(batch_gen(test_data), batch_gen(test_labels)):
        loss, logits = compute_scalar_loss(model, features, labels)
        loss_mean.update_state(loss)
        acc_mean.update_state(labels, logits)
    return {'loss': loss_mean.result(), 'accuracy': acc_mean.result()}


model_dict = {
    'mnist': get_mnist_model,
    'lstm': get_imdb_model,
    'embed': get_embedding_model,
    'ffnn': get_ffnn_model,
    'logreg': get_logreg_model,
    'cifar10': CIFAR10Model,
}


def main(args):
    print(args)
    if args.memory_limit:  # Option to limit GPU memory.
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])

    assert args.microbatches is None  # Only support microbatches=1.
    args.microbatches = args.batch_size

    data_fn = data.data_fn_dict[args.experiment][int(args.dummy_data)]
    kwargs = {
        'max_features': args.max_features,
        'max_len': args.max_len,
        'format': 'NHWC',
    }
    if args.dummy_data:
        kwargs['num_examples'] = args.batch_size * 2
    (train_data, train_labels), _ = data_fn(**kwargs)
    train_data, train_labels = tf.constant(train_data), tf.constant(train_labels)
    num_train_eg = train_data[0].shape[0]

    loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
    if args.experiment == 'logreg':
        loss_fn = lambda labels, logits: tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=tf.squeeze(logits))
        train_labels = tf.cast(train_labels, tf.float32)

    model_bs = 1 if args.dpsgd else args.batch_size
    model = model_dict[args.experiment](
        train_data,
        max_features=args.max_features,
        # batch_size=model_bs,
        args=args)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
    train_fn = private_train_step if args.dpsgd else train_step
    train_fn = partial(train_fn, model, optimizer, loss_fn, args)

    if args.no_vmap:
        print('No vmap for dpsgd!')

    if args.no_jit:
        print('No jit!')
    else:
        train_fn = tf.function(experimental_compile=args.use_xla)(train_fn)

    with tf.device('GPU'):
        dummy = tf.convert_to_tensor(1.)  # we use this to force CUDA sychronization

    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        for i, batch in enumerate(data.dataloader(train_data, train_labels, args.batch_size)):
            train_fn(batch)
        _ = dummy.numpy()  # forces a host->device transfer, synchronizing CUDA.
        duration = time.perf_counter() - start
        print("Time Taken: ", duration)
        timings.append(duration)

        if args.dpsgd:
            # eps = compute_eps_poisson(epoch, args.noise_multiplier, num_train_eg, args.batch_size,
            #                           1e-5)
            # mu = compute_mu_poisson(epoch, args.noise_multiplier, num_train_eg, args.batch_size)
            # print('For delta=1e-5, the current epsilon is: %.2f' % eps)
            # print('For delta=1e-5, the current mu is: %.2f' % mu)
            print('Trained with DPSGD optimizer')
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
    parser.add_argument('--xla', dest='use_xla', action='store_true')
    parser.add_argument('--no_xla', dest='use_xla', action='store_false')
    parser.add_argument('--memory_limit', default=None, type=int)
    parser.add_argument('--no_unroll', dest='no_unroll', action='store_true')

    parser.add_argument('--no_vmap', dest='no_vmap', action='store_true')
    parser.add_argument('--no_jit', dest='no_jit', action='store_true')
    args = parser.parse_args()
    main(args)
