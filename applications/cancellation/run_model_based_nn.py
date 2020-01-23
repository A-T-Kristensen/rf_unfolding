#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fit the model-based neural network to the full-duplex training set.
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

from gen_models import gen_model_based_nn
from complex_nn.complex_losses import complex_mean_squared_error

from helpers.preprocess import basic_parser, set_seeds, select_optimizer
from helpers.training import single_run, parameter_search_parallel
from helpers.flop import flop_model_based_nn


def train(params, batch_size, learning_rate, x_train, x_test, y_train, y_test, seed):
    """Training loop for the complex-valued model-based neural-network.
    """

    set_seeds(seed)

    network_array = params.htnn_struct.split("-")
    network_array = [layer for layer in network_array if layer != "-"]

    optimizer = select_optimizer(params)
    model     = gen_model_based_nn(params)
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

            loss = f_train(model, optimizer, params.gradient_clipping_param1, params.gradient_clipping_param2, params.gradient_clipping, network_array, input, target)

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
            print("Test output variance", np.var(test_out.numpy()))
            print("Test output mean", np.mean(test_out.numpy()))
            print()

    if True in np.isnan(test_loss_list):
        print("Warning: NaNs present in loss")

    if not os.path.exists("tmp_models"):
        os.makedirs("tmp_models")

    path = "tmp_models" + os.sep + str(uuid.uuid4().hex)
    model.save_weights(path, save_format="tf")

    for layer in model.layers:
        if layer.__class__.__name__ == "IQMixer":

            if network_array[0] == "T1":
                K1_real = np.cos(layer.phase.numpy())
                K1_imag = layer.gain.numpy() * np.sin(layer.phase.numpy())
                K1 = K1_real + 1j * K1_imag

                K2_real = layer.gain.numpy() * np.cos(layer.phase.numpy())
                K2_imag = -np.sin(layer.phase.numpy())
                K2 = K2_real + 1j * K2_imag

            elif network_array[0] == "T2":
                K1_real = np.cos(layer.phase.numpy())
                K1_imag = -layer.gain.numpy() * np.sin(layer.phase.numpy())
                K1 = K1_real + 1j * K1_imag

                K2_real = layer.gain.numpy() * np.cos(layer.phase.numpy())
                K2_imag = -np.sin(layer.phase.numpy())
                K2 = K2_real + 1j * K2_imag

            elif network_array[0] == "E" :
                K1_real = (1 + layer.gain.numpy() * np.cos(layer.phase.numpy())) / 2
                K1_imag = (layer.gain.numpy() * np.sin(layer.phase.numpy())) / 2
                K1 = K1_real + 1j * K1_imag

                K2_real = (1 - layer.gain.numpy() * np.cos(layer.phase.numpy())) / 2
                K2_imag = -(layer.gain.numpy() * np.sin(layer.phase.numpy())) / 2
                K2 = K2_real + 1j * K2_imag

            elif network_array[0] == "C" or network_array[0] == "R":
                K1_real = layer.K1_real.numpy()
                K1_imag = layer.K1_imag.numpy()
                K1 = K1_real + 1j * K1_imag

                K2_real = layer.K2_real.numpy()
                K2_imag = layer.K2_imag.numpy()
                K2 = K2_real + 1j * K2_imag

        if layer.__class__.__name__ == "Hammerstein":

            weights_real = layer.kernel_real.numpy()
            weights_imag = layer.kernel_imag.numpy()

            weights = weights_real + 1j * weights_imag

        if network_array[0] == "B":
            K1 = np.array([1.0])
            K2 = np.array([0.0])

    result_dict = {
        "train_loss": train_loss_list,
        "train_var": train_var_list,
        "test_loss": test_loss_list,
        "test_var": test_var_list,
        "model": path,
        "K1": K1,
        "K2": K2,
        "weights": weights,
    }

    return result_dict


def step(model, optimizer, gradient_clipping_param1, gradient_clipping_param2, gradient_clipping, network_array, input, target):

    with tf.GradientTape() as tape:

        loss  = complex_mean_squared_error(target, model(input))
        grads = tape.gradient(loss, model.trainable_variables)

        if gradient_clipping_param1 != 0.0 and network_array[0] != "B":
            if network_array[0] == "C":
                grads[0] = (tf.clip_by_value(grads[0], -gradient_clipping_param1, gradient_clipping_param1)) # K1 real
                grads[1] = (tf.clip_by_value(grads[1], -gradient_clipping_param1, gradient_clipping_param1)) # K1 imag
            else:
                grads[0] = (tf.clip_by_value(grads[0], -gradient_clipping_param1, gradient_clipping_param1)) # gain or K1

        if gradient_clipping_param2 != 0.0 and network_array[0] != "B":
            if network_array[0] == "C":
                grads[2] = (tf.clip_by_value(grads[2], -gradient_clipping_param2, gradient_clipping_param2)) # K2
                grads[3] = (tf.clip_by_value(grads[3], -gradient_clipping_param2, gradient_clipping_param2))
            else:
                grads[1] = (tf.clip_by_value(grads[1], -gradient_clipping_param2, gradient_clipping_param2)) # phase or K2

        if gradient_clipping != 0.0:
            if network_array[0] == "B":
                grads[0] = (tf.clip_by_value(grads[0], -gradient_clipping, gradient_clipping))
            elif network_array[0] == "C":
                grads[4] = (tf.clip_by_value(grads[4], -gradient_clipping, gradient_clipping))
            else:
                grads[2] = (tf.clip_by_value(grads[2], -gradient_clipping, gradient_clipping)) # gain or K2

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def wrapper_train_parallel(args):
    """Unpack the arguments to the train function. Used for multi-processing.
    """
    return train(*args)


def main(params):

    set_seeds(params.seed)
    shared_params = ("model_based_nn", "model_based_nn", params, gen_model_based_nn, flop_model_based_nn)
    print("NETWORK STRUCTURE:", params.htnn_struct)

    if params.search:
        if params.search_width:
            for max_power in range(params.min_width, params.max_width + 1, params.step_width):
                print("Max Power", max_power)
                params.max_power = max_power
                parameter_search_parallel(*shared_params, wrapper_train_parallel)
        else:
            parameter_search_parallel(*shared_params, wrapper_train_parallel)
    else:
        train_func = wrapper_train_parallel if params.mp else train
        single_run(*shared_params, train_func)


if __name__ == '__main__':

    parser = basic_parser()

    parser.set_defaults(n_epochs=50)
    parser.set_defaults(learning_rate=0.25)
    parser.set_defaults(batch_size=6)
    parser.set_defaults(optimizer="ftrl")
    parser.set_defaults(htnn_struct="C-C")
    parser.set_defaults(dtype="complex64")
    parser.set_defaults(data_format="ffnn")
    parser.set_defaults(fit_option="all")
    parser.set_defaults(initializer="hammerstein_rayleigh")

    params, unknown = parser.parse_known_args()

    if unknown:
        raise ValueError("This argument {} in unknown".format(unknown))

    main(params)
