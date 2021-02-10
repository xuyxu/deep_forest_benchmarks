import numpy as np
import pandas as pd
import scipy.io as scio
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from keras.datasets import (mnist, fashion_mnist, cifar10)
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_diabetes


mem = Memory('./cache')


def load_all():
    
    load_dict = {
        'ijcnn1': load_ijcnn1,
        'pendigits': load_pendigits,
        'letter': load_letter,
        'connect-4': load_connect,
        'sector': load_sector,
        'covtype': load_covtype,
        'susy': load_susy,
        'higgs': load_higgs,
        'usps': load_usps,
        'mnist': load_mnist,
        'fashion mnist': load_fashionmnist
    }
    
    return load_dict


def load_regression_all():

    load_dict = {
        "abalone": load_abalone,
        "cpusmall": load_cpusmall,
        "boston": load_boston_wrap,
        "diabetes": load_diabetes_wrap
    }

    return load_dict


# ============================================================================
# Classification Datasets
# ============================================================================


def load_ijcnn1():
    train = load_svmlight_file('../../Dataset/LIBSVM/ijcnn1_training')
    test = load_svmlight_file('../../Dataset/LIBSVM/ijcnn1_testing')
    
    X_train, X_test = train[0].toarray(), test[0].toarray()
    y_train, y_test = (train[1]+1) / 2, (test[1]+1) / 2  # {-1, 1} -> {0, 1}

    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_pendigits():
    train = load_svmlight_file('../../Dataset/LIBSVM/pendigits_training')
    test = load_svmlight_file('../../Dataset/LIBSVM/pendigits_testing')
    
    X_train, X_test = train[0].toarray(), test[0].toarray()
    y_train, y_test = train[1], test[1]
    
    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_letter():
    train = load_svmlight_file('../../Dataset/LIBSVM/letter_training')
    test = load_svmlight_file('../../Dataset/LIBSVM/letter_testing')
    
    X_train, X_test = train[0].toarray(), test[0].toarray()
    y_train, y_test = train[1]-1, test[1]-1  # [1, 26] -> [0, 25]

    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_connect():
    data = load_svmlight_file('../../Dataset/LIBSVM/connect-4')
    X = data[0].toarray()
    y = data[1]
    
    X_train, X_test = X[:47290, :], X[47290:, :]
    y_train, y_test = y[:47290]+1, y[47290:]+1  # {-1, 0, 1} -> {0, 1, 2}
    
    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_sector():
    train = load_svmlight_file('../../Dataset/LIBSVM/sector_training')
    test = load_svmlight_file('../../Dataset/LIBSVM/sector_testing')
    
    X_train, X_test = train[0].toarray(), test[0].toarray()
    y_train, y_test = train[1]-1, test[1]-1

    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_covtype():
    data = load_svmlight_file('../../Dataset/LIBSVM/covtype')
    X = data[0].toarray()
    y = data[1]
    
    X_train, X_test = X[:406708, :], X[406708:, :]
    y_train, y_test = y[:406708]-1, y[406708:]-1  # [1, 7] -> [0, 6]
    
    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


@mem.cache
def load_susy(subsample=False, subsample_size=1000000, random_state=0):
    data = load_svmlight_file('../../Dataset/LIBSVM/SUSY')
    X = np.asanyarray(data[0].toarray(), order='C')
    y = data[1]
    
    # Split training / testing
    X_train, X_test = X[:-500000, :], X[-500000:, :]
    y_train, y_test = y[:-500000], y[-500000:]
    
    if subsample:
        rng = check_random_state(random_state)
        sample_indice = rng.choice(X_train.shape[0], 
                                   subsample_size, 
                                   replace=False)
        X_train = X_train[sample_indice]
        y_train = y_train[sample_indice]
    
    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


@mem.cache
def load_higgs(subsample=False, subsample_size=1000000, random_state=0):
    data = load_svmlight_file('../../Dataset/LIBSVM/HIGGS')
    X = np.asanyarray(data[0].toarray(), order='C')
    y = data[1]

    # Split training / testing
    X_train, X_test = X[:-500000, :], X[-500000:, :]
    y_train, y_test = y[:-500000], y[-500000:]
    
    if subsample:
        rng = check_random_state(random_state)
        sample_indice = rng.choice(X_train.shape[0], 
                                   subsample_size, 
                                   replace=False)
        X_train = X_train[sample_indice]
        y_train = y_train[sample_indice]
    
    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_usps():
    train = load_svmlight_file('../../Dataset/LIBSVM/usps_training')
    test = load_svmlight_file('../../Dataset/LIBSVM/usps_testing')
    
    X_train, X_test = train[0].toarray(), test[0].toarray()
    y_train, y_test = train[1]-1, test[1]-1  # [1, 10] -> [0, 9]

    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_fashionmnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train, X_test = X_train.astype(np.float), X_test.astype(np.float)
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_news():
    train = load_svmlight_file('../../Dataset/LIBSVM/news20_training')
    test = load_svmlight_file('../../Dataset/LIBSVM/news20_testing')
    
    X_train, X_test = train[0].toarray(), test[0].toarray()
    y_train, y_test = train[1]-1, test[1]-1  # [1, 10] -> [0, 9]

    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


@mem.cache
def load_epsilon():
    train = load_svmlight_file('../../Dataset/LIBSVM/epsilon_training')
    test = load_svmlight_file('../../Dataset/LIBSVM/epsilon_testing')

    X_train, X_test = train[0].toarray(), test[0].toarray()
    y_train, y_test = train[1]-1, test[1]-1

    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


def load_aloi(test_size=0.33, random_state=0):
    
    if not 0 < test_size < 1:
        msg = '`test_size` should be in the range (0, 1), but got {} instead.'
        raise ValueError(msg.format(test_size))
    
    data = load_svmlight_file('../../Dataset/LIBSVM/aloi')
    X = data[0].toarray()
    y = data[1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    return (X_train.astype(np.float), y_train.astype(np.int),
            X_test.astype(np.float), y_test.astype(np.int))


# ============================================================================
# Regression Datasets
# ============================================================================


def load_abalone():
    data = load_svmlight_file('../Dataset/LIBSVM/abalone')
    X = np.asanyarray(data[0].toarray(), order='C')
    y = data[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)

    return X_train, y_train, X_test, y_test


def load_cpusmall():
    data = load_svmlight_file('../Dataset/LIBSVM/cpusmall')
    X = np.asanyarray(data[0].toarray(), order='C')
    y = data[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)

    return X_train, y_train, X_test, y_test


def load_boston_wrap():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, y_train, X_test, y_test


def load_diabetes_wrap():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, y_train, X_test, y_test
