#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complex-Valued layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys
import types as python_types
import warnings

import numpy as np
import math

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export

from complex_nn import complex_initializers
from complex_nn.complex_misc import calc_odd_powers
from complex_nn import complex_activations


class ComplexDense(layers.Layer):
    r"""Complex-valued equivalent of the dense layer.

    Arguments
    ---------
    units                   : int
        Positive integer, dimensionality of the output space.
    activation              : :obj:
        Activation function to use. If you don't specify anything, no activation
        is applied (ie. "linear" activation: `a(x) = x`).
    use_bias                : bool
        Boolean, whether the layer uses a bias vector.
    kernel_initializer      : :obj:
        Initializer for the `kernel` weights matrix.
    bias_initializer        : :obj:
        Initializer for the bias vector.
    kernel_regularizer      : :obj:
        Regularizer function applied to the `kernel` weights matrix.
    bias_regularizer        : :obj:
        Regularizer function applied to the bias vector.
    activity_regularizer    : :obj:
        Regularizer function applied to the output of the layer (its "activation").
    kernel_constraint       : :obj:
        Constraint function applied to the `kernel` weights matrix.
    bias_constraint         : :obj:
        Constraint function applied to the bias vector.

    Attributes
    ----------
    kernel_real : :class:`tf.Tensor`
        The real part of the weights.
    kernel_imag : :class:`tf.Tensor`
        The imaginary part of the weights.
    bias_real   : :class:`tf.Tensor`
        The real part of the biases.
    bias_imag   : :class:`tf.Tensor`
        The imaginary part of the biases.
    """

    def __init__(
            self,
            units,
            activation           = None,
            use_bias             = True,
            kernel_initializer   = complex_initializers.ComplexGlorotNormal,
            bias_initializer     = 'zeros',
            kernel_regularizer   = None,
            bias_regularizer     = None,
            activity_regularizer = None,
            kernel_constraint    = None,
            bias_constraint      = None,
            **kwargs,
    ):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ComplexDense, self).__init__(**kwargs)

        self.units              = int(units)
        self.activation         = activations.get(activation)
        self.use_bias           = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer   = regularizers.get(bias_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        self.bias_constraint    = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):

        if self.dtype == "complex128" or self.dtype == "float64":
            init_dtype = "float64"
        elif self.dtype == "complex64" or self.dtype == "float32":
            init_dtype = "float32"

        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` should be defined. Found `None`.')

        last_dim        = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.kernel_real = self.add_weight(
            'kernel_real',
            shape       = (last_dim, self.units),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            constraint  = self.kernel_constraint,
            dtype       = init_dtype,
            trainable   = True
        )

        self.kernel_imag = self.add_weight(
            'kernel_imag',
            shape       = (last_dim, self.units),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            constraint  = self.kernel_constraint,
            dtype       = init_dtype,
            trainable   = True
        )

        if self.use_bias:
            self.bias_real = self.add_weight(
                'bias_real',
                shape       = (self.units,),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                constraint  = self.bias_constraint,
                dtype       = init_dtype,
                trainable   = True
            )

            self.bias_imag = self.add_weight(
                'bias_imag',
                shape       = (self.units,),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                constraint  = self.bias_constraint,
                dtype       = init_dtype,
                trainable   = True
            )

        else:
            self.bias_real = None
            self.bias_imag = None

        self.built = True

    @tf.function
    def call(self, inputs):
        self.input_dim = inputs.shape[1]

        W = tf.complex(self.kernel_real, self.kernel_imag)

        output = tf.matmul(inputs, W)

        if self.use_bias:
            output += tf.complex(self.bias_real, self.bias_imag)

        if self.activation is not None:
            return self.activation(output)
        else:
            return output

        # Alternatively, this can be calculated as shown below (there is
        # also the option where people build the matrices and then stack
        # the real and imaginary parts).
        # Benchmarking the above solution with the one below reveal very small
        # differences in general, with the one above winning, usually.
        # Moreover, the above approach uses the complex-valued operators directly
        # y_real = tf.matmul(tf.math.real(inputs), self.kernel_real) - tf.matmul(tf.math.imag(inputs), self.kernel_imag)
        # y_imag = tf.matmul(tf.math.imag(inputs), self.kernel_real) + tf.matmul(tf.math.real(inputs), self.kernel_imag)

        # if self.use_bias:
        #     y_real += self.bias_real
        #     y_imag += self.bias_imag

        # output = tf.complex(y_real, y_imag)

        # if self.activation is not None:
        #     return self.activation(output)
        # else:
        #     return output

    def get_config(self):
        config = {
            'units'                : self.units,
            'activation'           : activations.serialize(self.activation),
            'use_bias'             : self.use_bias,
            'kernel_initializer'   : initializers.serialize(self.kernel_initializer),
            'bias_initializer'     : initializers.serialize(self.bias_initializer),
            'kernel_regularizer'   : regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer'     : regularizers.serialize(self.bias_regularizer),
            'activity_regularizer' : regularizers.serialize(self.activity_regularizer),
            'kernel_constraint'    : constraints.serialize(self.kernel_constraint),
            'bias_constraint'      : constraints.serialize(self.bias_constraint)
        }

        base_config = super(ComplexDense, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class IQMixer(layers.Layer):
    r"""Class defining the IQ-mixer layer. Let the complex digital transmitted signal at time instant :math:`n` be denoted by :math:`x[n]`.
    Assuming an ideal DAC, the digital baseband equivalent of the signal after the IQ imbalance introduced by the IQ mixer can be modeled as.

    .. math::
        x_{IQ}[n] = K_1 x[n] + K_2 x[n]^*

    We have multiple options for defining the parameters :math:`K_1, K_2`.

    **Option: number_type = T1 (trig)**

    .. math::
        K_1 = \cos( \phi ) + j g \sin(\phi) \qquad K_2 = g \cos( \phi ) - j \sin(\phi) .

    If no IQ-imbalance is present, then :math:`g = \phi = 0` and :math:`K_1 = 1`
    and :math:`K_2 = 0`.

    * `Joint Adaptive Compensation of Transmitter and Receiver IQ Imbalance Under Carrier Frequency Offset in OFDM-Based Systems <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4803780>`_.
    * `OFDM Systems With Both Transmitter and Receiver IQ Imbalances <https://ieeexplore.ieee.org/document/1506237>`_.
    * `Baseband Receiver Design for Wireless MIMO-OFDM Communications <https://ieeexplore.ieee.org/book/6218878>`_ (page 107).

    **Option: number_type = T2 (trig)**

    .. math::
        K_1 = \cos( \phi ) - j g \sin(\phi) \qquad K_2 = g \cos( \phi ) - j \sin(\phi) .

    The distortion parameters :math:`g` and :math:`\phi` are the amplitude and
    phase orthogonality mismatch between the I and Q branches in the RF modulation
    process at the transmitter.

    `Digital Compensation of Frequency-Dependent Joint Tx/Rx I/Q Imbalance in OFDM Systems Under High Mobility <https://ieeexplore.ieee.org/document/4939424>`_

    **Option: number_type = E (exponential)**

    We can write it in terms of exponentials

    .. math::
        K_1 = \frac{1}{2} (1+g \exp(j \phi)) \qquad K_2 =  \frac{1}{2} (1-g \exp(j \phi)) .

    Typically :math:`|K_1| \gg |K_2|`.

    * `Nonlinear self-interference cancellation in MIMO full-duplex transceivers under crosstalk <https://link.springer.com/article/10.1186/s13638-017-0808-4>`_
    * Equation (2.28) in `Digital Front-End Signal Processing with Widely-Linear Signal Models in Radio Devices <https://tutcris.tut.fi/portal/files/5453822/anttila.pdf>`_

    **Option: number_type = C (complex)**

    We simply have :math:`K_1, K_2 \in \mathbb{C}` and :math:`|K_1| \gg |K_2|`.
    In this case, we do not define them based on a common phase and amplitude.

    **Option: number_type = R (real)**

    Ideally, there is no phase imbalance, so :math:`\phi = 0` for the exponential
    option and :math:`K_1, K_2 \in \mathbb{R}`.
    We typically set :math:`|K_1| \gg |K_2|`.

    Given a memory length of :math:`M` and a batch-size of :math:`B`,
    the output from this layer will be a matrix of shape
    :math:`B \times M` with the results per batch along the rows.

    Arguments
    ---------
    number_type : str
        The number type (either real or complex) for the parameters
        :math:`K_1` and :math:`K_2`.

    Attributes
    ----------
    gain    : :class:`tf.Tensor`
        Amplitude distortion.
    phase   : :class:`tf.Tensor`
        Phase distortion.
    K1_real : :class:`tf.Tensor`.
        The real part of :math:`K_1`.
    K1_imag : :class:`tf.Tensor`
        The imaginary part :math:`K_1`.
    K2_real : :class:`tf.Tensor`
        The real part of :math:`K_2`.
    K2_imag : :class:`tf.Tensor`
        The imaginary part of :math:`K_2`.
    """

    def __init__(
        self,
        number_type = "R",
        **kwargs
    ):

        super(IQMixer, self).__init__()

        self.number_type = number_type

        self.phase = np.random.uniform(low = 0.001, high = 0.1)

        if self.number_type == "T1":
            self.gain    = np.random.uniform(low = 0.001, high = 0.01)
            self.K1_init = np.cos(self.phase) + 1j * self.gain * np.sin(self.phase)

        elif self.number_type == "T2":
            self.gain    = np.random.uniform(low = 0.001, high = 0.01)
            self.K1_init = np.cos(self.phase) - 1j * self.gain * np.sin(self.phase)

        # For the next 3, important to initialize such that output variance < 1.0!
        # The high range is selected because for higher powers, it is better
        # to have smaller numbers
        elif self.number_type == "E" :
            self.gain    = np.random.uniform(low = 0.5, high = 0.99)
            self.K1_init = (1 + self.gain * np.cos(self.phase)) / 2 + 1j * (self.gain * np.sin(self.phase)) / 2

        elif self.number_type == "C":
            self.gain    = np.random.uniform(low = 0.5, high = 0.99)
            self.K1_init = self.gain * np.exp(1j * self.phase)

        elif self.number_type == "R":
            self.K1_init = np.random.uniform(low = 0.5, high = 0.99)

    def build(self, input_shape):

        if self.dtype == "complex128" or self.dtype == "float64":
            init_dtype = "float64"
        elif self.dtype == "complex64" or self.dtype == "float32":
            init_dtype = "float32"

        if self.number_type == "E" or self.number_type == "T1" or self.number_type == "T2":

            self.gain_initializer  = tf.constant_initializer(self.gain)
            self.phase_initializer = tf.constant_initializer(self.phase)

            self.gain = self.add_weight(
                "gain",
                shape       = (1, 1),
                initializer = self.gain_initializer,
                trainable   = True
            )

            self.phase = self.add_weight(
                "phase",
                shape       = (1, 1),
                initializer = self.phase_initializer,
                trainable   = True
            )

        elif self.number_type == "C":

            self.K1_real_initializer = tf.constant_initializer(self.K1_init.real)
            self.K1_imag_initializer = tf.constant_initializer(self.K1_init.imag)

            self.K1_real = self.add_weight(
                "K1_real",
                shape       = (1, 1),
                initializer = self.K1_real_initializer,
                trainable   = True
            )

            self.K1_imag = self.add_weight(
                "K1_imag",
                shape       = (1, 1),
                initializer = self.K1_imag_initializer,
                trainable   = True
            )

            phase     = np.random.uniform(low = 0.001, high = 0.01)
            magnitude = np.random.uniform(low = 0.001, high = 0.01)

            K2 = magnitude * np.exp(1j * phase)

            self.K2_real_initializer = tf.constant_initializer(K2.real)
            self.K2_imag_initializer = tf.constant_initializer(K2.imag)

            self.K2_real = self.add_weight(
                "K2_real",
                shape       = (1, 1),
                initializer = self.K2_real_initializer,
                trainable   = True
            )

            self.K2_imag = self.add_weight(
                "K2_imag",
                shape       = (1, 1),
                initializer = self.K2_imag_initializer,
                trainable   = True
            )

        elif self.number_type == "R":

            K2 = self.K1_init * 0.01

            self.K1_initializer = tf.constant_initializer(self.K1_init)
            self.K2_initializer = tf.constant_initializer(K2)

            self.K1_real = self.add_weight(
                "K1_real",
                shape       = (1, 1),
                initializer = self.K1_initializer,
                dtype       = init_dtype,
                trainable   = True
            )

            self.K2_real = self.add_weight(
                "K2_real",
                shape       = (1, 1),
                initializer = self.K2_initializer,
                dtype       = init_dtype,
                trainable   = True
            )

            # NB: It is more efficient to have the zero matrices and instead
            # of doing real-valued calculations with real and imaginary parts
            self.K1_imag = self.add_weight(
                "K1_imag",
                shape       = (1, 1),
                initializer = tf.zeros_initializer,
                dtype       = init_dtype,
                trainable   = False
            )

            self.K2_imag = self.add_weight(
                "K2_imag",
                shape       = (1, 1),
                initializer = tf.zeros_initializer,
                dtype       = init_dtype,
                trainable   = False
            )

        else:
            raise ValueError("Invalid value to number_type passed")

        self.built = True

    @tf.function
    def call(self, inputs):
        r"""
        Parameters
        ----------
        input : :class:`tf.Tensor`
            The complex-valued input of shape :math:`B \times M`. That is,
            we assume the inner-most dimension corresponds to different
            time-steps.

        Returns
        -------
        output : :class:`tf.Tensor`
            The complex-valued output of shape :math:`B \times M`.
        """

        if self.number_type == "T1":
            K1 = tf.complex(tf.math.cos(self.phase), self.gain * tf.math.sin(self.phase))
            K2 = tf.complex(self.gain * tf.math.cos(self.phase), -tf.math.sin(self.phase))

        elif self.number_type == "T2":
            K1 = tf.complex(tf.math.cos(self.phase), -self.gain * tf.math.sin(self.phase))
            K2 = tf.complex(self.gain * tf.math.cos(self.phase), -tf.math.sin(self.phase))

        elif self.number_type == "E":
            K1 = tf.complex((1 + self.gain * tf.math.cos(self.phase)) / 2, (self.gain * tf.math.sin(self.phase)) / 2)
            K2 = tf.complex((1 - self.gain * tf.math.cos(self.phase)) / 2, -(self.gain * tf.math.sin(self.phase)) / 2)

        # Seems to work the best
        elif self.number_type == "C" or self.number_type == "R":
            K1 = tf.complex(self.K1_real, self.K1_imag)
            K2 = tf.complex(self.K2_real, self.K2_imag)

        return K1 * inputs + K2 * tf.math.conj(inputs)

    def get_config(self):
        config = {
            'number_type': self.number_type,
        }
        base_config = super(IQMixer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class Hammerstein(layers.Layer):
    r"""Class defining the Hammerstein layer. For the Hammerstein model, we
    first have the non-linearity with branches for each power, then we have the
    LTI system.

    Provided the :math:`x_{IQ}` signal, this is passed through a power-amplifier,
    the effect of which can be modeled using a parallel Hammerstein model.

    .. math::
        x_{PA}(n) = \sum^P_{p=1, p \, \mathrm{odd}} \sum^M_{m=1} h_{PA,p}(m) x_{IQ}(m-1) |x_{IQ}(m-1)|^{p-1}

    In this layer, we multiply the impulse response values with the input

    Arguments
    ---------
    number_type         : str
        The number type (either real or complex) for the parameters.
    n_channels          : int
        The amount of memory.
    min_power           : int
        The minimum power to consider for the Hammerstein non-linearity.
    max_power           : int
        The minimum power to consider for the Hammerstein non-linearity.
    K1                  : :class:`tf.Tensor`
        The :math:`K_1` parameter from the IQ-mixer.
    kernel_initializer  : :obj:
        Initializer for the `kernel` weights matrix.

    Attributes
    ----------
    kernel_real : :class:`tf.Tensor`
        The real part of the weights.
    kernel_imag : :class:`tf.Tensor`
        The imaginary part of the weights.
    """

    def __init__(
        self,
        number_type,
        n_channels,
        min_power,
        max_power,
        K1                 = None,
        kernel_initializer = 'hammerstein_rayleigh',
        kernel_regularizer = None,
        **kwargs
    ):

        super(Hammerstein, self).__init__()

        self.number_type        = number_type
        self.n_channels         = n_channels
        self.min_power          = min_power
        self.max_power          = max_power
        self.K1                 = K1
        self.odd_powers         = calc_odd_powers(self.min_power, self.max_power)
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        if "hammerstein" in kernel_initializer:
            init = complex_initializers.hammerstein_initializer(
                self.n_channels,
                self.min_power,
                self.max_power,
                K1 = self.K1,
                distribution=kernel_initializer
            )

            self.kernel_initializer_real = tf.constant_initializer(init.real)
            self.kernel_initializer_imag = tf.constant_initializer(init.imag)

        else:

            if kernel_initializer == "normal":
                self.kernel_initializer_real = complex_initializers.ComplexGlorotNormal
                self.kernel_initializer_imag = complex_initializers.ComplexGlorotNormal
            elif kernel_initializer == "uniform":
                self.kernel_initializer_real = complex_initializers.ComplexGlorotUniform
                self.kernel_initializer_imag = complex_initializers.ComplexGlorotUniform
            else:
                raise ValueError("The initializer {} is unknown".format(kernel_initializer))

        self.activation = complex_activations.complex_hammerstein

    def build(self, input_shape):

        if self.dtype == "complex128" or self.dtype == "float64":
            init_dtype = "float64"
        elif self.dtype == "complex64" or self.dtype == "float32":
            init_dtype = "float32"

        self.kernel_real = self.add_weight(
            "kernel_real",
            shape       = (self.odd_powers * self.n_channels, 1),
            dtype       = init_dtype,
            initializer = self.kernel_initializer_real,
            regularizer = self.kernel_regularizer,
            trainable   = True
        )

        self.kernel_imag = self.add_weight(
            "kernel_imag",
            shape       = (self.odd_powers * self.n_channels, 1),
            dtype       = init_dtype,
            initializer = self.kernel_initializer_imag,
            regularizer = self.kernel_regularizer,
            trainable   = True
        )

    @tf.function
    def call(self, inputs):
        r"""
        Parameters
        ----------
        input : :class:`tf.Tensor`
            The complex-valued output of shape :math:`B \times (M \times P)`
            where :math:`M` is the memory length and :math:`P` is the number
            of odd-powers considered.

        Returns
        -------
        output : :class:`tf.Tensor`
            The complex-valued output of shape :math:`B \times 1`, that is, a
            scalar value.
        """

        x_pow  = self.activation(inputs, self.min_power, self.max_power)
        output = tf.matmul(x_pow, tf.complex(self.kernel_real, self.kernel_imag))

        return output

    def get_config(self):

        config = {
            'number_type': self.number_type,
            'n_channels': self.n_channels,
            'min_power': self.min_power,
            'max_power': self.max_power,
            'K1': self.K1,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'activation': self.activaton,
        }
        base_config = super(Hammerstein, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class SimpleComplexRNNCell(layers.Layer):
    """Cell class for SimpleComplexRNN.

    Arguments
    ---------
    units                   : int
        Positive integer, dimensionality of the output space.
    activation              : :obj:
        Activation function to use. Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias                : bool
        Whether the layer uses a bias vector.
    kernel_initializer      : :obj:
        Initializer for the `kernel` weights matrix, used for the linear
        transformation of the inputs.
    recurrent_initializer   : :obj:
        Initializer for the `recurrent_kernel weights matrix, used for the
        linear transformation of the recurrent state.
    bias_initializer        : :obj:
        Initializer for the bias vector.
    kernel_regularizer      : :obj:
        Regularizer function applied to the `kernel` weights matrix.
    recurrent_regularizer   : :obj:
        Regularizer function applied to the `recurrent_kernel` weights matrix.
    bias_regularizer        : :obj:
        Regularizer function applied to the bias vector.
    kernel_constraint       : :obj:
        Constraint function applied to the `kernel` weights matrix.
    recurrent_constraint    : :obj:
        Constraint function applied to the `recurrent_kernel` weights matrix.
    bias_constraint         : :obj:
        Constraint function applied to the bias vector.
    dropout                 : :obj:
        Float between 0 and 1. Fraction of the units to drop for the linear
        transformation of the inputs.
    recurrent_dropout       : :obj:
        Float between 0 and 1. Fraction of the units to drop for the linear
        transformation of the recurrent state.

    Attributes
    ----------
    kernel_real : :class:`tf.Tensor`
        The real part of the weights.
    kernel_imag : :class:`tf.Tensor`
        The imaginary part of the weights.
    bias_real   : :class:`tf.Tensor`
        The real part of the biases.
    bias_imag   : :class:`tf.Tensor`
        The imaginary part of the biases.
    """

    def __init__(
        self,
        units,
        activation            = complex_activations.complex_amp_phase_exp,
        use_bias              = True,
        kernel_initializer    = complex_initializers.ComplexGlorotNormal,
        recurrent_initializer = complex_initializers.ComplexGlorotNormal,
        bias_initializer      = 'zeros',
        kernel_regularizer    = None,
        recurrent_regularizer = None,
        bias_regularizer      = None,
        activity_regularizer  = None,
        kernel_constraint     = None,
        recurrent_constraint  = None,
        bias_constraint       = None,
        dropout               = 0.,
        recurrent_dropout     = 0.,
        return_sequences      = False,
        return_state          = False,
        go_backwards          = False,
        stateful              = False,
        unroll                = False,
        **kwargs
    ):

        super(SimpleComplexRNNCell, self).__init__(**kwargs)

        self.units      = units
        self.activation = activations.get(activation)
        self.use_bias   = use_bias

        self.kernel_initializer    = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer      = initializers.get(bias_initializer)

        self.kernel_regularizer    = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer      = regularizers.get(bias_regularizer)

        self.kernel_constraint    = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint      = constraints.get(bias_constraint)

        self.dropout           = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size        = self.units
        self.output_size       = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):

        if self.dtype == "complex128" or self.dtype == "float64":
            init_dtype = "float64"
        elif self.dtype == "complex64" or self.dtype == "float32":
            init_dtype = "float32"

        self.last_dim = tensor_shape.dimension_value(input_shape[-1])

        self.kernel_real = self.add_weight(
            'kernel_real',
            shape       = (self.last_dim, self.units),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            constraint  = self.kernel_constraint,
            dtype       = init_dtype,
            trainable   = True
        )

        self.kernel_imag = self.add_weight(
            'kernel_imag',
            shape       = (self.last_dim, self.units),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            constraint  = self.kernel_constraint,
            dtype       = init_dtype,
            trainable   = True
        )

        self.recurrent_kernel_real = self.add_weight(
            'recurrent_kernel_real',
            shape       = (self.units, self.units),
            initializer = self.recurrent_initializer,
            regularizer = self.recurrent_regularizer,
            constraint  = self.recurrent_constraint,
            dtype       = init_dtype,
            trainable   = True
        )

        self.recurrent_kernel_imag = self.add_weight(
            'recurrent_kernel_imag',
            shape       = (self.units, self.units),
            initializer = self.recurrent_initializer,
            regularizer = self.recurrent_regularizer,
            constraint  = self.recurrent_constraint,
            dtype       = init_dtype,
            trainable   = True
        )

        if self.use_bias:

            self.bias_real = self.add_weight(
                'bias_real',
                shape       = (self.units,),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                constraint  = self.bias_constraint,
                dtype       = init_dtype,
                trainable   = True
            )

            self.bias_imag = self.add_weight(
                'bias_imag',
                shape       = (self.units,),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                constraint  = self.bias_constraint,
                dtype       = init_dtype,
                trainable   = True
            )

        else:
            self.bias_real = None
            self.bias_imag = None

        self.built = True

    def call(self, inputs, states, training=None):
        """
        states:
            List of state tensors corresponding to the previous timestep.
        training:
            Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
        """
        prev_output    = states[0]
        self.input_dim = inputs.shape[1]

        kernel           = tf.complex(self.kernel_real, self.kernel_imag)
        recurrent_kernel = tf.complex(self.recurrent_kernel_real, self.recurrent_kernel_imag)

        h = tf.matmul(inputs, kernel)

        if self.use_bias is not None:
            h += tf.complex(self.bias_real, self.bias_imag)

        output = h + tf.matmul(prev_output, recurrent_kernel)

        if self.activation is not None:
            output = self.activation(output)

        return output, [output]

    # def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

    #     return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }

        base_config = super(SimpleRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SimpleComplexRNN(tf.keras.layers.RNN):
    """Fully-connected RNN where the output is to be fed back to input.

    Arguments
    ---------
    units                   : int
        Positive integer, dimensionality of the output space.
    activation              : :obj:
        Activation function to use. Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias                : bool
        Whether the layer uses a bias vector.
    kernel_initializer      : :obj:
        Initializer for the `kernel` weights matrix, used for the linear
        transformation of the inputs.
    recurrent_initializer   : :obj:
        Initializer for the `recurrent_kernel weights matrix, used for the
        linear transformation of the recurrent state.
    bias_initializer        : :obj:
        Initializer for the bias vector.
    kernel_regularizer      : :obj:
        Regularizer function applied to the `kernel` weights matrix.
    recurrent_regularizer   : :obj:
        Regularizer function applied to the `recurrent_kernel` weights matrix.
    bias_regularizer        : :obj:
        Regularizer function applied to the bias vector.
    kernel_constraint       : :obj:
        Constraint function applied to the `kernel` weights matrix.
    recurrent_constraint    : :obj:
        Constraint function applied to the `recurrent_kernel` weights matrix.
    bias_constraint         : :obj:
        Constraint function applied to the bias vector.
    dropout                 : :obj:
        Float between 0 and 1. Fraction of the units to drop for the linear
        transformation of the inputs.
    recurrent_dropout       : :obj:
        Float between 0 and 1. Fraction of the units to drop for the linear
        transformation of the recurrent state.
    return_sequences        : bool
        Whether to return the last output in the output sequence, or the full
        sequence.
    return_state            : bool
        Whether to return the last state in addition to the output.
    go_backwards            : bool
        (default False). If True, process the input sequence backwards
        and return the reversed sequence.
    stateful                : bool
        (default False). If True, the last state for each sample at index i in
        a batch will be used as initial state for the sample of index i in the
        following batch.
    unroll                  : bool
        (default False). If True, the network will be unrolled, else a symbolic
        loop will be used. Unrolling can speed-up a RNN, although it tends to
        be more memory-intensive. Unrolling is only suitable for short sequences.
    """

    def __init__(
        self,
        units,
        activation            = complex_activations.complex_amp_phase_exp,
        use_bias              = True,
        kernel_initializer    = complex_initializers.ComplexGlorotNormal,
        recurrent_initializer = complex_initializers.ComplexGlorotNormal,
        bias_initializer      = 'zeros',
        kernel_regularizer    = None,
        recurrent_regularizer = None,
        bias_regularizer      = None,
        activity_regularizer  = None,
        kernel_constraint     = None,
        recurrent_constraint  = None,
        bias_constraint       = None,
        dropout               = 0.,
        recurrent_dropout     = 0.,
        return_sequences      = False,
        return_state          = False,
        go_backwards          = False,
        stateful              = False,
        unroll                = False,
        **kwargs
    ):

        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `SimpleRNN` has been deprecated. '
                            'Please remove it from your layer call.')

        cell = SimpleComplexRNNCell(
            units,
            activation            = activation,
            use_bias              = use_bias,
            kernel_initializer    = kernel_initializer,
            recurrent_initializer = recurrent_initializer,
            bias_initializer      = bias_initializer,
            kernel_regularizer    = kernel_regularizer,
            recurrent_regularizer = recurrent_regularizer,
            bias_regularizer      = bias_regularizer,
            kernel_constraint     = kernel_constraint,
            recurrent_constraint  = recurrent_constraint,
            bias_constraint       = bias_constraint,
            dropout               = dropout,
            recurrent_dropout     = recurrent_dropout,
            dtype                 = kwargs.get('dtype')
        )

        super(SimpleComplexRNN, self).__init__(
            cell,
            return_sequences = return_sequences,
            return_state     = return_state,
            go_backwards     = go_backwards,
            stateful         = stateful,
            unroll           = unroll,
            **kwargs
        )

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        """
        Parameters
        ----------
        input           : :class:`tf.Tensor`
            A 3D tensor.
        mask            : :class:`tf.Tensor`
            Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
        training        : bool
        Python boolean indicating whether the layer should behave in training
        mode or in inference mode. This argument is passed to the cell when
        calling it. This is only relevant if `dropout` or `recurrent_dropout`
        is used.
        initial_state   : :obj:
        List of initial state tensors to be passed to the first
        call of the cell.

        Returns
        -------
        output : :class:`tf.Tensor`
            The complex-valued output.
        """

        # self.cell.reset_dropout_mask()
        # self.cell.reset_recurrent_dropout_mask()

        return super(SimpleComplexRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }

        base_config = super(SimpleRNN, self).get_config()
        del base_config['cell']

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)