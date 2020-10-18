import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence


def dataloader(x, y, batch_size):
    if batch_size > len(x):
        raise ValueError('Batch Size too big.')
    num_eg = len(x)
    assert num_eg == len(y)
    for i in range(0, num_eg, batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


def load_cifar10(format='NHWC', **_):
    train, test = tf.keras.datasets.cifar10.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.asarray(train_data, dtype=np.float32) / 255.
    test_data = np.asarray(test_data, dtype=np.float32) / 255.

    if format == 'NHWC':
        pass
    elif format == 'NCHW':
        train_data = train_data.transpose((0, 3, 1, 2))
        test_data = test_data.transpose((0, 3, 1, 2))
    else:
        raise ValueError('Invalid format.')

    train_labels = np.asarray(train_labels, dtype=np.int32).squeeze()
    test_labels = np.asarray(test_labels, dtype=np.int32).squeeze()

    return (train_data, train_labels), (test_data, test_labels)


def load_dummy_cifar10(num_examples, format='NHWC', **_):
    train_labels = np.random.randint(0, 10, num_examples).astype(np.int32)

    if format == 'NHWC':
        train_data = np.random.random((num_examples, 32, 32, 3)).astype(np.float32)
    elif format == 'NCHW':
        train_data = np.random.random((num_examples, 3, 32, 32)).astype(np.float32)
    else:
        raise ValueError('Invalid format.')

    return (train_data, train_labels), (train_data, train_labels)


def load_mnist(format='NHWC', **_):
    """Loads MNIST and preprocesses to combine training and validation data."""
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.asarray(train_data, dtype=np.float32) / 255.
    test_data = np.asarray(test_data, dtype=np.float32) / 255.

    if format == 'NHWC':
        train_data, test_data = train_data[..., None], test_data[..., None]
    elif format == 'NCHW':
        train_data, test_data = train_data[:, None], test_data[:, None]
    else:
        raise ValueError('Invalid format.')

    train_labels = np.asarray(train_labels, dtype=np.int32)
    test_labels = np.asarray(test_labels, dtype=np.int32)

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.
    assert train_labels.ndim == 1
    assert test_labels.ndim == 1

    return (train_data, train_labels), (test_data, test_labels)


def load_dummy_mnist(num_examples, format='NHWC', **_):
    train_data = np.random.random((num_examples, 28, 28)).astype(np.float32)
    train_labels = np.random.randint(0, 10, num_examples).astype(np.int32)

    if format == 'NHWC':
        train_data = train_data[..., None]
    elif format == 'NCHW':
        train_data = train_data[:, None]
    else:
        raise ValueError('Invalid format.')

    return (train_data, train_labels), (train_data, train_labels)


def load_imdb(max_features=10_000, max_len=256, **_):
    """Load IMDB movie reviews data."""
    train, test = tf.keras.datasets.imdb.load_data(num_words=max_features)
    (train_data, train_labels), (test_data, test_labels) = train, test

    train_data = sequence.pad_sequences(train_data, maxlen=max_len).astype(np.int32)
    test_data = sequence.pad_sequences(test_data, maxlen=max_len).astype(np.int32)
    train_labels, test_labels = train_labels.astype(np.int32), test_labels.astype(np.int32)
    return (train_data, train_labels), (test_data, test_labels)


def load_dummy_imdb(num_examples, max_features=10_000, max_len=256, **_):
    train_data = np.random.randint(0, max_features, (num_examples, max_len)).astype(np.int32)
    train_labels = np.random.random(num_examples).round().astype(np.int32)
    return (train_data, train_labels), (train_data, train_labels)


def load_adult(**_):
    """Loads ADULT a2a as in LIBSVM and preprocesses to combine training and validation data."""
    # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

    data_x = np.load('adult_processed_x.npy')
    data_y = np.load('adult_processed_y.npy')
    data_y[data_y == -1] = 0
    train_data = data_x.astype(np.float32)
    train_labels = data_y.astype(np.int32)

    return (train_data, train_labels), None


def load_dummy_adult(num_examples, **_):
    train_data = np.random.random((num_examples, 104)).astype(np.float32)
    train_labels = np.random.random(num_examples).round().astype(np.int32)
    return (train_data, train_labels), None


data_fn_dict = {
    'mnist': (load_mnist, load_dummy_mnist),
    'lstm': (load_imdb, load_dummy_imdb),
    'embed': (load_imdb, load_dummy_imdb),
    'ffnn': (load_adult, load_dummy_adult),
    'logreg': (load_adult, load_dummy_adult),
    'cifar10': (load_cifar10, load_dummy_cifar10),
}

if __name__ == '__main__':
    # Test Functionality
    names = ['mnist', 'imdb', 'adult', 'cifar10']
    data_fns = [load_mnist, load_imdb, load_adult, load_cifar10]
    dummy_data_fns = [load_dummy_mnist, load_dummy_imdb, load_dummy_adult, load_dummy_cifar10]
    for name, data_fn, dummy_data_fn in zip(names, data_fns, dummy_data_fns):
        print(f'Checking {name}')
        (x, y), _ = data_fn()
        (dx, dy), _ = dummy_data_fn(x.shape[0])
        assert x.shape == dx.shape, f'Original: {x.shape}, Dummy: {dx.shape}'
        assert y.shape == dy.shape, f'Original: {y.shape}, Dummy: {dy.shape}'
        assert x.dtype == dx.dtype, f'Original: {x.dtype}, Dummy: {dx.dtype}'
        assert y.dtype == dy.dtype, f'Original: {y.dtype}, Dummy: {dy.dtype}'
