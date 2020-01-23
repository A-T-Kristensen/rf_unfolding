#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit the complex-valued neural network to the full-duplex training set.
"""

import sys
import os
import uuid
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disable GPU as data-set is small
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np

if tf.__version__[0] != "2":
    tf.enable_eager_execution()

from gen_models import gen_complex_ffnn
from complex_nn.complex_losses import complex_mean_squared_error

from helpers.preprocess import basic_parser, set_seeds, select_optimizer
from helpers.flop import flop_complexnn
from helpers.training import single_run, parameter_search_parallel


def train(params, batch_size, learning_rate, x_train, x_test, y_train, y_test, seed):
    """Training loop for the complex-valued feed-forward neural-network.
    """

    set_seeds(seed)

    optimizer = select_optimizer(params)
    model     = gen_complex_ffnn(params)
    f_train   = tf.function(step)
    n_batches = x_train.shape[0] // batch_size

    # Information on distribution of |r(n)|^2, mean and variance
    train_loss_list = np.zeros(params.n_epochs) # Normal loss is just MEAN squared of residuals
    train_var_list  = np.zeros(params.n_epochs) # Variance of the |r(n)|^2 values

    test_loss_list = np.zeros(params.n_epochs)
    test_var_list  = np.zeros(params.n_epochs)

    for epoch in range(0, params.n_epochs):
        train_loss = 0

        for batch in range(0, n_batches):

            first_index  = batch * batch_size
            second_index = first_index + batch_size

            input  = x_train[first_index:second_index]
            target = y_train[first_index:second_index]

            loss = f_train(model, optimizer, params.gradient_clipping, input, target)

            train_loss += loss

        test_out  = model(x_test)
        test_loss = complex_mean_squared_error(y_test, test_out)

        train_loss /= n_batches

        train_loss_list[epoch] = train_loss
        test_loss_list[epoch] = test_loss

        train_var_list[epoch] = np.var(np.abs(y_train - model(x_train).numpy())**2)
        test_var_list[epoch]  = np.var(np.abs(y_test - test_out.numpy())**2)

        if not params.search and not params.mp:
            print("Epoch", epoch + 1)
            print("Train Loss", train_loss_list[epoch])
            print("Test Loss", test_loss_list[epoch])
            print()

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


def step(model, optimizer, gradient_clipping, input, target):

    with tf.GradientTape() as tape:

        loss  = complex_mean_squared_error(target, model(input))

        grads = tape.gradient(loss, model.trainable_variables)
        grads = [(tf.clip_by_value(grad, -gradient_clipping, gradient_clipping)) for grad in grads]

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def wrapper_train_parallel(args):
    """Unpack the arguments to the train function. Used for multi-processing.
    """
    return train(*args)


def main(params):

    set_seeds(params.seed)
    shared_params = ("complex_ffnn", "nn", params, gen_complex_ffnn, flop_complexnn)

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

    # File specific defaults
    parser.set_defaults(n_epochs=50)
    parser.set_defaults(learning_rate=0.0045)
    parser.set_defaults(batch_size=62)
    parser.set_defaults(optimizer="adam")
    parser.set_defaults(ffnn_struct="10")
    parser.set_defaults(dtype="complex64")
    parser.set_defaults(data_format="ffnn")
    parser.set_defaults(fit_option="nl")
    parser.set_defaults(gradient_clipping=1.0)
    parser.set_defaults(initializer="rayleigh")

    params, unknown = parser.parse_known_args()

    if unknown:
        raise ValueError("This argument {} in unknown".format(unknown))

    main(params)
