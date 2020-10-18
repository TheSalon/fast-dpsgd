"""Based on: https://github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial_vectorized.py"""

import os
import time
from functools import partial

import tensorflow.compat.v1 as tf
from tensorflow_privacy.privacy.analysis.gdp_accountant import (compute_eps_poisson,
                                                                compute_mu_poisson)
from tensorflow_privacy.privacy.analysis.rdp_accountant import (compute_rdp, get_privacy_spent)
from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized

import data
import utils
from tf2dp import model_dict


def nn_model_fn(model, loss_fn, args, features, labels, mode):
    # the model has to be created inside the estimator function to be on the right graph.
    logits = model()(features['x'])
    vector_loss = loss_fn(labels=labels, logits=logits)
    scalar_loss = tf.reduce_mean(vector_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        if args.dpsgd:
            # Use DP version of GradientDescentOptimizer. Other optimizers are
            # available in dp_optimizer. Most optimizers inheriting from
            # tf.train.Optimizer should be wrappable in differentially private
            # counterparts by calling dp_optimizer.optimizer_from_args().
            optimizer = dp_optimizer_vectorized.VectorizedDPSGD(
                l2_norm_clip=args.l2_norm_clip,
                noise_multiplier=args.noise_multiplier,
                num_microbatches=args.microbatches,
                learning_rate=args.learning_rate)
            opt_loss = vector_loss
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
            opt_loss = scalar_loss
        global_step = tf.train.get_global_step()
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        # In the following, we pass the mean of the loss (scalar_loss) rather than
        # the vector_loss because tf.estimator requires a scalar loss. This is only
        # used for evaluation and debugging by tf.estimator. The actual loss being
        # minimized is opt_loss defined above and passed to optimizer.minimize().
        return tf.estimator.EstimatorSpec(mode=mode, loss=scalar_loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        # This branch is unused (but kept from the TFP tutorial.)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels,
                                            predictions=tf.argmax(input=logits, axis=1))
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          eval_metric_ops=eval_metric_ops)


def compute_epsilon(epoch, num_train_eg, args):
    """Computes epsilon value for given hyperparameters."""
    steps = epoch * num_train_eg // args.batch_size
    if args.noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = args.batch_size / num_train_eg
    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=args.noise_multiplier,
                      steps=steps,
                      orders=orders)
    # Delta is set to approximate 1 / (number of training points).
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def main(args):
    print(args)
    tf.disable_eager_execution()  # TFP is designed to run in TF1 graph mode.
    if args.memory_limit:  # Option to limit GPU memory
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])

    assert args.microbatches is None  # vectorized TFP only supports microbatches=1
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
    num_train_eg = train_data.shape[0]

    loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
    if args.experiment == 'logreg':
        loss_fn = lambda labels, logits: tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=tf.squeeze(logits))
        train_labels = train_labels.astype('float32')

    model = partial(model_dict[args.experiment],
                    features=train_data,
                    max_features=args.max_features,
                    args=args)

    if args.use_xla:
        # Setting both the environment flag and session_config is redundant, but
        # we do this just in case.
        assert os.environ['TF_XLA_FLAGS'] == '--tf_xla_auto_jit=2'
        session_config = tf.ConfigProto()
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
        run_config = tf.estimator.RunConfig(session_config=session_config)
        print('Using XLA!')
    else:
        run_config = None
        print('NOT using XLA!')

    model_obj = tf.estimator.Estimator(model_fn=partial(nn_model_fn, model, loss_fn, args),
                                       config=run_config)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_data},
                                                        y=train_labels,
                                                        batch_size=args.batch_size,
                                                        num_epochs=args.epochs,
                                                        shuffle=True)

    steps_per_epoch = num_train_eg // args.batch_size
    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model_obj.train(input_fn=train_input_fn, steps=steps_per_epoch)
        duration = time.perf_counter() - start
        print("Time Taken: ", duration)
        timings.append(duration)

        if args.dpsgd:
            # eps = compute_epsilon(epoch, num_train_eg, args)
            # print('For delta=1e-5, the current epsilon is: %.2f' % eps)
            print('Trained with DPSGD optimizer')
        else:
            print('Trained with vanilla non-private SGD optimizer')

    if not args.no_save:
        utils.save_runtimes(__file__.split('.')[0], args, timings)
    else:
        print('Not saving!')
    print('Done!')


if __name__ == '__main__':
    parser = utils.get_parser(model_dict.keys())
    parser.add_argument('--memory_limit', default=None, type=int)
    parser.add_argument('--xla', dest='use_xla', action='store_true')
    parser.add_argument('--no_xla', dest='use_xla', action='store_false')
    parser.add_argument('--no_unroll', dest='no_unroll', action='store_true')
    args = parser.parse_args()
    main(args)
