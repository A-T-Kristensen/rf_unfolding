#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions related to preprocessing of data for the full-duplex application
and settings related to the application.
"""

import argparse
import random

import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow_core import size

import helpers.polynomial_cancellation as poly

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, fftpack


def select_optimizer(params):
    """Returns the Tensorflow optimizer.

    Parameters
    ----------
    params : :obj:
        The arguments for the program.

    Returns
    -------
    optimizer : :obj:
        The optimizer to use for the neural networks.
    """

    if params.lr_schedule:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            params.learning_rate,
            decay_steps=100000,
            decay_rate=0.95,
        )
    else:
        learning_rate = params.learning_rate

    if params.optimizer == "adadelta":
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif params.optimizer == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif params.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif params.optimizer == "adamax":
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif params.optimizer == "ftrl":
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
    elif params.optimizer == "nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif params.optimizer == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif params.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=params.momentum)
    else:
        raise NotImplementedError("Support for the given optimizer is not supported {}".format(params.optimizer))

    return optimizer


def set_seeds(seed):
    """Seed the random number generators for python, numpy and tensorflow.

    Parameters
    ----------
    seed : int
        Seed to use for the random number generators.
    """

    random.seed(seed)
    np.random.seed(seed)

    if tf.__version__[0] == "2":
        tf.random.set_seed(seed)
    else:
        tf.random.set_random_seed(seed)


def gen_seeds(seed, n_seeds):
    """Generate a number of seeds spaced far away from each other

    Parameters
    ----------
    seed : int
        Initial seed to generate.
    n_seeds : int
        Number of seeds to generate.

    Returns
    -------
    seed_array : :class:`numpy.ndarray`
        Array of seeds based on the initial seed.
    """

    seed_array    = np.zeros((n_seeds), dtype=np.int)
    seed_array[0] = seed

    set_seeds(seed_array[0])

    for seed_iteration in range(1, n_seeds):
        seed_array[seed_iteration] = int(np.random.uniform(low=0, high=int(2**31 - 1)))

    return seed_array


def initialize(params):
    """Load the full-duplex dataset, converts to the right types and generates
    the coefficients for the polynomial cancellation.
    """
    x, y, noise, measured_noise_power = load_data('data/fdTestbedData20MHz10dBm', params)

    x = x.astype(np.complex64)
    y = y.astype(np.complex64)

    # Split into training and test sets
    training_samples = int(np.floor(x.size * params.training_ratio))

    x_train = x[0:training_samples]
    y_train = y[0:training_samples]
    x_test  = x[training_samples:]
    y_test  = y[training_samples:]

    training_size    = min(params.training_size, 1.0)
    training_samples = int(np.floor(training_samples * training_size))

    x_train = x[0:training_samples]
    y_train = y[0:training_samples]

    # Estimate linear cancellation parameters and perform linear cancellation
    h_lin        = poly.si_estimation_linear(x_train, y_train, params)
    y_canc_train = poly.si_cancellation_linear(x_train, h_lin, params)
    y_canc_test  = poly.si_cancellation_linear(x_test, h_lin, params)

    # Generate data
    x_train, y_train, y_train_orig, y_var = gen_data(x_train, y_train, y_canc_train, None, params)
    x_test, y_test, y_test_orig, _        = gen_data(x_test, y_test, y_canc_test, y_var, params)

    # Shuffling negatively asffects the results, seems like we need the
    # dependdency between samples a bitselect_optimizer
    # Makes sense as we do a form of online learning
    if params.shuffle:
        x_train, y_train = shuffle(x_train, y_train, random_state=params.seed)

    # Remove part not entirely in window as done in the gen_data() function
    y_canc_train = y_canc_train[params.hsi_len:]
    y_canc_test  = y_canc_test[params.hsi_len:]
    y_train_orig = y_train_orig[params.hsi_len:]
    y_test_orig  = y_test_orig[params.hsi_len:]

    noise                = noise
    measured_noise_power = measured_noise_power

    return x_train, y_train, y_train_orig, x_test, y_test, y_test_orig, y_var, h_lin, y_canc_train, y_canc_test, noise, measured_noise_power


def load_data(fileName, params):
    """Load the full-duplex dataset from the mat file.
    """

    # Get parameters
    data_offset = params.data_offset
    chan_len    = params.hsi_len
    offset      = np.maximum(data_offset - int(np.ceil(chan_len / 2)), 1)

    # Load the file
    matFile = scipy.io.loadmat(fileName)

    x = matFile['txSamples'].flatten()
    y = matFile['analogResidual'].flatten()

    x           = np.squeeze(matFile['txSamples'], axis=1)[:-offset]
    y           = np.squeeze(matFile['analogResidual'], axis=1)[offset:]
    y           = y - np.mean(y)
    noise       = np.squeeze(matFile['noiseSamples'], axis=1)
    noise_power = np.squeeze(matFile['noisePower'], axis=1)

    return x, y, noise, noise_power


def gen_data(x, y, y_canc, y_var, params):
    """Post-process the loaded dataset depending on the different options.

    Parameters
    ----------
    x                   : :class:`numpy.ndarray`
        Input data.
    y                   : :class:`numpy.ndarray`
        Target data.
    y_canc              : :class:`numpy.ndarray`
        Cancellation values from the linear canceller.
    y_var               : float
        Variance to scale target with.
    params.hsi_len      : int
        Number of channels.
    params.dtype        : str
        If data_option="complex", keep the input data as complex-valued,
        otherwise, reshape such that we have the real- and imaginary-parts split.
    params.data_format  : str
        If data_format="ffnn", return the data in the format (time-steps,) if
        data_format="rnn, return the data in the format (time-steps, features)
    params.fit_option   : str
        If "all", then fit to the pure data, otherwise, subtract the linear
        cancellation and train on the remaining part.

    Returns
    -------
    x       : :class:`numpy.ndarray`
        Input data restructured.
    y       : :class:`numpy.ndarray`
        Target data restructured.
    y_orig  : :class:`numpy.ndarray`
        Original y input.
    y_var   : float
        Variance to scale target with.
    """

    if params.dtype == "complex128" or params.dtype == "float64":
        x      = x.astype(np.complex128)
        y      = y.astype(np.complex128)
        y_canc = y_canc.astype(np.complex128)
    elif params.dtype == "complex64" or params.dtype == "float32":
        x      = x.astype(np.complex64)
        y      = y.astype(np.complex64)
        y_canc = y_canc.astype(np.complex64)

    y_orig = y

    # Only fit to non-linear part
    if params.fit_option == "nl":
        y = y - y_canc
    elif params.fit_option != "all":
        raise NotImplementedError("The provided fitting option does not match the supported ones".format(params.fit_option))

    # Align to window
    y = np.reshape(y[params.hsi_len:], (y.size - params.hsi_len, 1))

    if y_var is None:
        y_var = np.var(y)

    y = y / np.sqrt(y_var)

    x = np.reshape(
        np.array([x[i:i + params.hsi_len] for i in range(x.size - params.hsi_len)]),
        (x.size - params.hsi_len, params.hsi_len)
    )

    # RNN expects data in shape (batch, time_step, features)
    if "complex" in params.dtype and params.data_format == "rnn":
        x = np.reshape(x, (*x.shape, 1))

    elif "float" in params.dtype:
        x_real = x.real
        x_imag = x.imag

        if params.data_format == "ffnn":

            x = np.empty((x_real.shape[0], 2 * params.hsi_len))

            x[:, 0:params.hsi_len]                   = x_real
            x[:, params.hsi_len: 2 * params.hsi_len] = x_imag

        elif params.data_format == "rnn":

            x = np.empty((x_real.shape[0], params.hsi_len, 2))

            x[:, :, 0] = x_real
            x[:, :, 1] = x_imag

    elif "complex" not in params.dtype:
        raise NotImplementedError("The data-option you requested is not supported {}".format(params.data_format))

    return x, y, y_orig, y_var


def basic_parser():
    """Contains the defaults arguments used for the application.
    """

    parser = argparse.ArgumentParser(description='Parameters for full duplex')

    # Hardware settings
    parser.add_argument(
        "--sampling-freq-MHz", dest="sampling_freq_MHz", type=int, default=20,
        help="Sampling frequency, required for correct scaling of PSD",
    )

    parser.add_argument(
        "--training-ratio", dest="training_ratio", type=float, default=0.9,
        help="Ratio of total samples to use for training",
    )

    parser.add_argument(
        "--training-size", dest="training_size", type=float, default=1.0,
        help="Percentage of training set to use.",
    )

    parser.add_argument(
        "--data-scaled", dest="data_scaled", type=int, default=1,
        help="If 1, then the data is scaled by e.g. dividing targets with std. deviation.",
    )

    parser.add_argument(
        "--data-offset", dest="data_offset", type=int, default=14,
        help="Data offset to take transmitter-receiver misalignment into account",
    )

    parser.add_argument(
        "--dtype", dest="dtype", type=str, default="complex",
        help="Specifies if we use complex-valued or split training data (complex, real).",
    )

    parser.add_argument(
        "--data-format", dest="data_format", type=str, default="ffnn",
        help="Specifies if we return the data in the form needed for FFNNs or RNNS (ffnn, rnn)",
    )

    parser.add_argument(
        "--fit-option", dest="fit_option", type=str, default="all",
        help="Specifies if we fit to the full data or only the non-linear part (all, nl)",
    )

    parser.add_argument(
        "--hsi-len", dest="hsi_len", type=int, default=13,
        help="Self-interference channel length",
    )

    parser.add_argument(
        "--min-power", dest="min_power", type=int, default=1,
        help="Minimum PA non-linearity order",
    )

    parser.add_argument(
        "--max-power", dest="max_power", type=int, default=5,
        help="Maximum PA non-linearity order",
    )

    parser.add_argument(
        "--htnn-struct", dest="htnn_struct", type=str, default="C-C",
        help="Indicate the network structure. First layer has options B=bypass, T1, T2, E, C, R and second layer only has option C. Defaults to R-C .",
    )

    parser.add_argument(
        "--ffnn-struct", dest="ffnn_struct", type=str, default="1",
        help="Indicate number of layers and number of neurons in each, e.g. 1-1 indicates 2 hidden layers and 1 neuron in each. The input and output are fixed",
    )

    parser.add_argument(
        "--activation", dest="activation", type=str, default="relu",
        help="Indicates which activation function to use",
    )

    parser.add_argument(
        "--rnn-cell", dest="rnn_cell", type=str, default="simple",
        help="Define the cell type for the RNN",
    )

    parser.add_argument(
        "--rnn-stateful", dest="rnn_stateful", type=int, default=0,
        help="If 1, last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch",
    )

    parser.add_argument(
        "--rnn-unroll", dest="rnn_unroll", type=int, default=0,
        help="If 1, the network is unrolled which can speed up an RNN, although it may be more memory intensive. Mainly suitable for short sequences",
    )

    parser.add_argument(
        "--optimizer", dest="optimizer", type=str, default="ftrl",
        help="The optimizer used to generate results (ftrl, adam). Please remember to change your parameters when using a different optimizer",
    )

    parser.add_argument(
        "--initializer", dest="initializer", type=str, default="normal",
        help="One of (normal or uniform) for normal neural networks or (normal, uniform, hammerstein_normal, hammerstein_rayleigh, hammerstein_uniform) for Hammerstein network.",
    )

    parser.add_argument(
        "--seed", dest="seed", type=int, default=42,
        help="The default seed",
    )

    parser.add_argument(
        "--n-seeds", dest="n_seeds", type=int, default=1,
        help="Number of runs with different seeds",
    )

    parser.add_argument(
        "--n-epochs", dest="n_epochs", type=int, default=10,
        help="Number of training epochs for NN training",
    )

    parser.add_argument(
        "--shuffle", dest="shuffle", type=int, default=0,
        help="If 1, then shuffle the training data",
    )

    parser.add_argument(
        "--lr-schedule", dest="lr_schedule", type=int, default=0,
        help="If 1, enable learning-rate scheduling",
    )
    parser.add_argument(
        "--learning-rate", dest="learning_rate", type=float, default=0.25,
        help="Learning rate for NN training",
    )

    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=8,
        help="Batch size for NN training",
    )

    parser.add_argument(
        "--regularizer", dest="regularizer", type=str, default=None,
        help="If 1, use weight regularization, either l1 or l2.",
    )

    parser.add_argument(
        "--weight-decay", dest="weight_decay", type=float, default=1e-9,
        help="Weight decay",
    )

    parser.add_argument(
        "--momentum", dest="momentum", type=float, default=0.0,
        help="Momentum",
    )

    parser.add_argument(
        "--gradient-clipping-param1", dest="gradient_clipping_param1", type=float, default=0.0,
        help="First defined parameter in the Hammerstein model (either K1, or gain).",
    )

    parser.add_argument(
        "--gradient-clipping-param2", dest="gradient_clipping_param2", type=float, default=0.0,
        help="First defined parameter in the Hammerstein model (either K2, or phase).",
    )

    parser.add_argument(
        "--gradient-clipping", dest="gradient_clipping", type=float, default=0.0,
        help="General gradient clipping for neural network weights or the h parameters in the Hammersterin model.",
    )

    parser.add_argument(
        "--search", dest="search", type=int, default=0,
        help="If 1, perform a grid-search for the optimal parameters, otherwise just fit using defaults",
    )

    parser.add_argument(
        "--n-search-points", dest="n_search_points", type=int, default=10,
        help="Number of initialization points for hyperparameter tuning",
    )

    parser.add_argument(
        "--min-batch-size", dest="min_batch_size", type=int, default=8,
        help="Minimum batch size for search",
    )

    parser.add_argument(
        "--max-batch-size", dest="max_batch_size", type=int, default=64,
        help="Maximum batch size for search",
    )

    parser.add_argument(
        "--min-learning-rate", dest="min_learning_rate", type=float, default=1e-2,
        help="Minimum learning-rate for search",
    )

    parser.add_argument(
        "--max-learning-rate", dest="max_learning_rate", type=float, default=0.5,
        help="Maximum learning-rate for search",
    )

    parser.add_argument(
        "--cv", dest="cv", type=int, default=0,
        help="If 1, do cross-validation, the selection of the best hyper-parameters are then based on the average from the cross-validation, and we then retrain the model on the whole data-set using the best hyper-parameters.",
    )

    parser.add_argument(
        "--cv-folds", dest="cv_folds", type=int, default=5,
        help="Specifies number of cross-validation folds",
    )

    parser.add_argument(
        "--n-cv-seeds", dest="n_cv_seeds", type=int, default=5,
        help="Specifies number of seeds to use for each cross-validation fold",
    )

    parser.add_argument(
        "--n-cv-epochs", dest="n_cv_epochs", type=int, default=20,
        help="Number of epochs to train for during cross-validation",
    )

    parser.add_argument(
        "--mp", dest="mp", type=int, default=0,
        help="If 1, enable multiprocessing",
    )

    parser.add_argument(
        "--n-jobs", dest="n_jobs", type=int, default=1,
        help="Number of parallel jobs to run at a time",
    )

    parser.add_argument(
        "--search-width", dest="search_width", type=int, default=0,
        help="If 1, search the width in an interval with some steps",
    )

    parser.add_argument(
        "--min-width", dest="min_width", type=int, default=1,
        help="Minimum width for width search for NN",
    )

    parser.add_argument(
        "--max-width", dest="max_width", type=int, default=10,
        help="Maximum width for width search for NN",
    )

    parser.add_argument(
        "--step-width", dest="step_width", type=int, default=1,
        help="Step in width search for NN",
    )

    parser.add_argument(
        "--search-training-size", dest="search_training_size", type=int, default=0,
        help="If 1, then after selecting optimal hyper opt, will also check different training set sizes. Runs after CV so requires CV=1",
    )

    parser.add_argument(
        "--min-training-size", dest="min_training_size", type=int, default=25,
        help="Minimum percentage of training set used for training set sizes",
    )

    parser.add_argument(
        "--max-training-size", dest="max_training_size", type=int, default=100,
        help="Maximum percentage of training set used for training set sizes",
    )

    parser.add_argument(
        "--step-training-size", dest="step_training_size", type=int, default=25,
        help="Step of percentage of training set used for training set sizes",
    )

    parser.add_argument(
        "--save", dest="save", type=int, default=0,
        help="If 1, save the results",
    )

    parser.add_argument(
        "--exp-name", dest="exp_name", type=str, default=None,
        help="Additional naming for experiment",
    )

    parser.add_argument(
        "--verbose", dest="verbose", type=int, default=0,
        help="Verbosity level, only used in a few places",
    )

    return parser
