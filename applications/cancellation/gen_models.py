#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Funtions for generating the different models.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from complex_nn.complex_layers import ComplexDense
from complex_nn.complex_layers import SimpleComplexRNN

from complex_nn.complex_layers import IQMixer
from complex_nn.complex_layers import Hammerstein

from complex_nn.complex_activations import complex_amp_phase_exp
from complex_nn.complex_activations import complex_cardioid
from complex_nn.complex_activations import complex_mod_relu
from complex_nn.complex_activations import complex_split_relu
from complex_nn.complex_activations import complex_z_relu

from complex_nn.complex_initializers import ComplexGlorotNormal, ComplexGlorotUniform
from complex_nn.complex_initializers import ComplexHeNormal, ComplexHeUniform

from helpers.preprocess import select_optimizer

import numpy as np


def gen_model(params, name):
    """Generate the model based on the name.

    Parameters
    ----------
    params  : :obj:
        Parameters.
    name    : str
        Name of the model (model_based_nn, complex_rnn, rnn, complex_ffnn, ffnn).

    Returns
    -------
    model : :obj:
        The Tensorflow model.
    """

    if "model_based_nn" in name:
        model = gen_model_based_nn(params)
    elif "complex_rnn" in name:
        model = gen_complex_rnn(params)
    elif "rnn" in name:
        model = gen_rnn(params)
    elif "complex_ffnn" in name:
        model = gen_complex_ffnn(params)
    elif "ffnn" in name:
        model = gen_ffnn(params)
    else:
        raise NotImplementedError("The model {} is not supported".format(name))

    return model


def gen_ffnn(params, learning_rate=0.01):
    """Generate the feed-forward neural network.

    Parameters
    ----------
    params          : :obj:
        Parameters.
    learning_rate   : str
        Learning-rate.

    Returns
    -------
    model : :obj:
        The Tensorflow model.
    """

    model = tf.keras.models.Sequential()

    network_array = params.ffnn_struct.split("-")
    network_array = [int(layer) for layer in network_array if layer.isdigit()]

    input_layer  = tf.keras.layers.InputLayer(input_shape=(2 * params.hsi_len))
    model.add(input_layer)

    if params.initializer not in ["normal", "uniform"]:
        raise ValueError("The provided initializer {} is not valid".format(params.initializer))

    if params.activation == "relu":
        initializer = tf.keras.initializers.he_normal() if params.initializer == "normal" else tf.keras.initializers.he_uniform()
    else:
        initializer = tf.keras.initializers.GlorotNormal if params.initializer == "normal" else tf.keras.initializers.GlorotUniform

    for n_hidden in network_array:
        model.add(tf.keras.layers.Dense(n_hidden, activation=tf.nn.relu, kernel_initializer=initializer))

    # Initializer for output layer
    initializer = tf.keras.initializers.GlorotNormal if params.initializer == "normal" else tf.keras.initializers.GlorotUniform

    output_layer = tf.keras.layers.Dense(2, kernel_initializer=initializer)
    model.add(output_layer)

    optimizer = select_optimizer(params)

    model.compile(loss="mse", optimizer=optimizer)

    return model


def gen_complex_ffnn(params):
    """Return the complex-valued neural network model.

    Parameters
    ----------
    params  : :obj:
        Parameters.

    Returns
    -------
    model : :obj:
        The Tensorflow model.
    """

    model = tf.keras.models.Sequential()

    network_array = params.ffnn_struct.split("-")
    network_array = [int(layer) for layer in network_array if layer.isdigit()]

    input_layer = tf.keras.layers.InputLayer(input_shape=(params.hsi_len), dtype=tf.complex64)
    model.add(input_layer)

    if params.activation == "relu":
        activation = complex_split_relu
    elif params.activation == "amp_phase":
        activation = complex_amp_phase_exp
    elif params.activation == "cardioid":
        activation = complex_cardioid
    elif params.activation == "mrelu":
        activation = complex_mod_relu
    elif params.activation == "zrelu":
        activation = complex_z_relu
    else:
        raise ValueError("The provided activation function {} is not recognized.".format(params.activation))

    if params.initializer not in ["normal", "rayleigh", "uniform"]:
        raise ValueError("The provided initializer {} is not valid".format(params.initializer))

    if params.activation == "relu":
        initializer = ComplexHeNormal if params.initializer == "normal" else ComplexHeUniform
    else:
        initializer = ComplexGlorotNormal if params.initializer == "normal" else ComplexGlorotUniform

    if params.regularizer == "l1":
        kernel_regularizer = tf.keras.regularizers.l1(params.weight_decay)
    elif params.regularizer == "l2":
        kernel_regularizer = tf.keras.regularizers.l2(params.weight_decay)
    else:
        kernel_regularizer = None

    # Build network from structure
    for n_hidden in network_array:
        model.add(
            ComplexDense(
                n_hidden,
                activation         = activation,
                use_bias           = True,
                kernel_initializer = initializer,
                kernel_regularizer = kernel_regularizer
            )
        )

    initializer = ComplexGlorotNormal if params.initializer == "normal" else ComplexGlorotUniform
    output_layer = ComplexDense(1, use_bias=True, kernel_initializer=initializer)
    model.add(output_layer)

    return model


def gen_complex_rnn(params, batch_size=16, learning_rate=0.01):
    """Generate the complex-valued RNN model.

    Parameters
    ----------
    params          : :obj:
        Parameters.
    batch_size      : str
        Batch-size.
    learning_rate   : str
        Learning-rate.

    Returns
    -------
    model : :obj:
        The Tensorflow model.
    """

    model = tf.keras.models.Sequential()

    network_array = params.ffnn_struct.split("-")
    network_array = [int(layer) for layer in network_array if layer.isdigit()]
    network_depth = len(network_array)

    stateful = True if params.rnn_stateful else False
    unroll   = True if params.rnn_unroll else False

    if stateful:
        raise NotImplementedError("Statefulness of the complex-valued RNN is not supported at the moment.")

    # Build network from structure
    for layer in range(0, network_depth):
        # When stacking RNN layers, output full sequences until last layer
        if network_depth > 1 and layer < (network_depth - 1):
            output_sequence = True
        else:
            output_sequence = False

        if params.rnn_cell == "simple":
            model.add(
                SimpleComplexRNN(
                    network_array[layer],
                    stateful         = stateful,
                    unroll           = unroll,
                    return_sequences = output_sequence
                )
            )

        else:
            raise ValueError("The provided cell type {} is not supported".format(params.rnn_cell))

    output_layer = ComplexDense(1, use_bias=True, kernel_initializer=ComplexGlorotUniform)
    model.add(output_layer)

    return model


def gen_rnn(params, batch_size=16, learning_rate=0.01):
    """Generate the real-valued RNN model.

    Parameters
    ----------
    params          : :obj:
        Parameters.
    batch_size      : str
        Batch-size.
    learning_rate   : str
        Learning-rate.

    Returns
    -------
    model : :obj:
        The Tensorflow model.
    """

    model = tf.keras.models.Sequential()

    network_array = params.ffnn_struct.split("-")
    network_array = [int(layer) for layer in network_array if layer.isdigit()]
    network_depth = len(network_array)

    input_layer  = tf.keras.layers.InputLayer(input_shape=(params.hsi_len, 2), batch_size=batch_size)
    model.add(input_layer)

    stateful = True if params.rnn_stateful else False
    unroll   = True if params.rnn_unroll else False

    if stateful:
        raise NotImplementedError("Statefulness of the RNN is not supported at the moment.")

    # Build network from structure
    for layer in range(0, network_depth):
        # When stacking RNN layers, output full sequences until last layer
        if network_depth > 1 and layer < (network_depth - 1):
            output_sequence = True
        else:
            output_sequence = False

        if params.rnn_cell == "simple":
            model.add(
                tf.keras.layers.SimpleRNN(
                    network_array[layer],
                    stateful         = stateful,
                    unroll           = unroll,
                    return_sequences = output_sequence
                )
            )

        elif params.rnn_cell == "gru":
            model.add(
                tf.keras.layers.GRU(
                    network_array[layer],
                    stateful         = stateful,
                    unroll           = unroll,
                    return_sequences = output_sequence
                )
            )

        elif params.rnn_cell == "lstm":
            model.add(
                tf.keras.layers.LSTM(
                    network_array[layer],
                    stateful         = stateful,
                    unroll           = unroll,
                    return_sequences = output_sequence
                )
            )

        else:
            raise ValueError("The provided cell type {} is not supported".format(params.rnn_cell))

    output_layer = tf.keras.layers.Dense(2)
    model.add(output_layer)

    optimizer = select_optimizer(params)

    model.compile(loss="mse", optimizer=optimizer)

    return model


def gen_model_based_nn(params):
    """Generates the model-based neural network.

    Parameters
    ----------
    params : :obj:
        Parameters.

    Returns
    -------
    model : :obj:
        The Tensorflow model.
    """

    network_array = params.htnn_struct.split("-")
    network_array = [layer for layer in network_array if layer != "-"]

    if params.regularizer == "l1":
        kernel_regularizer = tf.keras.regularizers.l1(params.weight_decay)
    elif params.regularizer == "l2":
        kernel_regularizer = tf.keras.regularizers.l2(params.weight_decay)
    else:
        kernel_regularizer = None

    model = tf.keras.models.Sequential()

    # Input layer required to make model restore easier
    input_layer = tf.keras.layers.InputLayer(input_shape=(params.hsi_len), dtype=tf.complex64)
    model.add(input_layer)

    layer1 = IQMixer(number_type = network_array[0])

    # Bypass IQ layer by using a B
    if network_array[0] != "B":
        K1_init = layer1.K1_init
        model.add(layer1)
    else:
        K1_init = None

    layer2 = Hammerstein(
        network_array[1],
        params.hsi_len,
        params.min_power,
        params.max_power,
        K1=K1_init,
        kernel_initializer=params.initializer,
        kernel_regularizer = kernel_regularizer,
    )

    model.add(layer2)

    return model
