#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fit the real-valued recurrent neural network to the full-duplex training set.
"""

import os
import uuid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disable GPU as data-set is small
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np

if tf.__version__[0] != "2":
    tf.enable_eager_execution()

from gen_models import gen_rnn
from helpers.flop import flop_rnn
from helpers.preprocess import basic_parser, set_seeds
from helpers.training import parameter_search_parallel, single_run


def train(params, batch_size, learning_rate, x_train, x_test, y_train, y_test, seed):
    """Training loop for the real-valued recurrent neural-network.
    """

    set_seeds(seed)

    if params.search or params.mp:
        verbosity = 0
    else:
        verbosity = 1

    model = gen_rnn(params, batch_size, learning_rate)

    # Information on distribution of |r(n)|^2, mean and variance
    train_loss_list = np.zeros(params.n_epochs) # Normal loss is just MEAN squared of residuals
    train_var_list  = np.zeros(params.n_epochs) # Variance of the |r(n)|^2 values

    test_loss_list = np.zeros(params.n_epochs)
    test_var_list  = np.zeros(params.n_epochs)

    # Restructure input to match batch_size (required for a statefull network)
    if params.rnn_stateful:
        x_train = x_train[: x_train.shape[0] - (x_train.shape[0]) % batch_size]
        y_train = y_train[: y_train.shape[0] - (y_train.shape[0]) % batch_size]

    history = model.fit(
        x_train,
        np.hstack((y_train.real, y_train.imag)),
        epochs          = params.n_epochs,
        batch_size      = batch_size,
        verbose         = verbosity,
        validation_data = (x_test, np.hstack((y_test.real, y_test.imag))),
    )

    # Compensate for the additional counts performed for mean when split to real and imag
    # That is, mean is computed for array of size 2N so reduced too much
    train_loss_list = 2 * np.array(history.history["loss"])
    test_loss_list  = 2 * np.array(history.history["val_loss"])

    if not params.search and not params.mp:
        print("Train Loss", train_loss_list[-1])
        print("Test Loss", test_loss_list[-1])
        print()

    if not os.path.exists("tmp_models"):
        os.makedirs("tmp_models")

    path = "tmp_models" + os.sep + str(uuid.uuid4().hex)
    model.save_weights(path, save_format="tf")

    result_dict = {
        "train_loss": train_loss_list,
        "train_var": train_var_list,
        "test_loss": test_loss_list,
        "test_var": test_var_list,
        "model": path,
    }

    return result_dict


def wrapper_train_parallel(args):
    """Unpack the arguments to the train function. Used for multi-processing.
    """
    return train(*args)


def main(params):

    set_seeds(params.seed)

    if params.rnn_stateful:
        postfix = "_stateful"
    else:
        postfix = ""

    shared_params = ("rnn" + "_" + params.rnn_cell + postfix, "nn", params, gen_rnn, flop_rnn)

    if params.search:
        network_depth = len([int(layer) for layer in params.ffnn_struct.split("-") if layer.isdigit()])

        if params.search_width:
            for n_hidden in range(params.min_width, params.max_width + 1, params.step_width):
                params.ffnn_struct = (network_depth * (str(n_hidden) + "-"))[:-1]
                print("NETWORK STRUCTURE:", params.ffnn_struct)
                parameter_search_parallel(*shared_params, wrapper_train_parallel)

        else:
            print("NETWORK STRUCTURE:", params.ffnn_struct)
            parameter_search_parallel(*shared_params, wrapper_train_parallel)

    else:
        print("NETWORK STRUCTURE:", params.ffnn_struct)
        train_func = wrapper_train_parallel if params.mp else train
        single_run(*shared_params, train_func)


if __name__ == '__main__':

    parser = basic_parser()

    parser.set_defaults(n_epochs=50)
    parser.set_defaults(learning_rate=0.0025)
    parser.set_defaults(batch_size=158)
    parser.set_defaults(optimizer="adam")
    parser.set_defaults(ffnn_struct="20")
    parser.set_defaults(dtype="float32")
    parser.set_defaults(data_format="rnn")
    parser.set_defaults(fit_option="nl")

    params, unknown = parser.parse_known_args()

    if unknown:
        raise ValueError("This argument {} in unknown".format(unknown))

    main(params)
