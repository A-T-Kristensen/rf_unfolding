#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source code defining activations functions for complex-valued neural networks.

See https://arxiv.org/pdf/1802.08026.pdf and
https://www.cs.dartmouth.edu/~trdata/reports/TR2018-859.pdf for some examples.
"""

import tensorflow as tf
import math


@tf.function
def complex_amp_phase_exp(input):
    r"""Function defining the amplitude-phase-type activation function

    .. math::
        f(z) = \mathrm{tanh}(|z|) \exp(i \theta_z) .

    See `Complex-Valued Neural Networks <https://www.springer.com/gp/book/9783642276316>`_.

    Parameters
    ----------
    input : :class:`tf.Tensor`
        The complex-valued input :math:`z`.

    Returns
    -------
    output : :class:`tf.Tensor`
        The complex-valued output :math:`f(z)`.
    """

    z_abs      = tf.math.abs(input)
    tanh_z_abs = tf.math.tanh(z_abs)

    phase     = tf.math.angle(input)
    exp_phase = tf.complex(tf.math.cos(phase), tf.math.sin(phase))

    output = exp_phase * tf.cast(tanh_z_abs, dtype=exp_phase.dtype)

    return output


@tf.function
def complex_cardioid(input):
    r"""Defines the cardioid function

    .. math::
        f(z) = \frac{1}{2} (1+\cos(\theta_z)) z .

    See `Better than Real: Complex-valued Neural Nets for MRI Fingerprinting <https://arxiv.org/abs/1707.00070>`_.

    Parameters
    ----------
    input : :class:`tf.Tensor`
        The complex-valued input :math:`z`.

    Returns
    -------
    output : :class:`tf.Tensor`
        The complex-valued output :math:`f(z)`.
    """

    phase  = tf.math.angle(input)
    scaler = tf.cast(1 / 2 * (1 + tf.math.cos(phase)), dtype=input.dtype)

    output = scaler * input

    return output


@tf.function
def complex_hammerstein(input, min_power, max_power):
    r"""Defines the non-linearity function for the Hammerstein Neural Network.

    Provided the :math:`x_{IQ}` signal, this is passed through a power-amplifier,
    the effect of which can be modeled using a parallel Hammerstein model with
    the output from the IQ-mixer as input.

    .. math::
        x_{PA}(n) = \sum^P_{p=1, p \, \mathrm{odd}} \sum^M_{m=1} h_{PA,p}(m) x_{IQ}(m-1) |x_{IQ}(m-1)|^{p-1}

    In this function, we generate the power values
    :math:`|x_{IQ}(n-1)|^{p-1}`.

    Parameters
    ----------
    input       : :class:`tf.Tensor`
        The complex-valued input of shape :math:`B \times M` where :math:`M`
        is the memory length.
    min_power   : int
        The minimum power to consider for the Hammerstein non-linearity.
    max_power   : int
        The minimum power to consider for the Hammerstein non-linearity.

    Returns
    -------
    output : :class:`tf.Tensor`
        The complex-valued output of shape :math:`B \times (M \times P)`
        where :math:`M` is the memory length and :math:`P` is the number
        of odd-powers considered.
    """

    input_abs = tf.cast(tf.abs(input), dtype=input.dtype)

    # Generates arrays for different powers and stacks them, that is, if we have
    # 13 memory taps, the first 13 values of the output will be for power 1,
    # the next 13 for power 3 and so forth.
    output = tf.concat(
        [input * tf.math.pow(input_abs, n - 1) for n in range(min_power, max_power + 1, 2)]
        , axis=-1
    )

    return output


@tf.function
def complex_mod_relu(input, b=0.5):
    r"""Defines the mod-relu function

    .. math::
        f(z) = \max(0, |z| + b ) \exp(i \theta_z )

    See `Unitary Evolution Recurrent Neural Networks <https://arxiv.org/abs/1511.06464>`_.

    Parameters
    ----------
    input : :class:`tf.Tensor`
        The complex-valued input :math:`z`.

    Returns
    -------
    output : :class:`tf.Tensor`
        The complex-valued output :math:`f(z)`.
    """

    phase  = tf.math.angle(input)
    relu   = tf.nn.relu(tf.math.abs(input) + b)

    exp_phase = tf.complex(tf.math.cos(phase), tf.sin(phase))

    output = tf.cast(relu, dtype=input.dtype) * exp_phase

    return output


@tf.function
def complex_split_relu(input):
    r"""Defines the SplitReLU (or CReLU) activation function

    .. math::
        f_{sr}(z) = \max(0, \Re(z)) + i \max(0, \Im(z))

    Parameters
    ----------
    input : :class:`tf.Tensor`
        The complex-valued input :math:`z`.

    Returns
    -------
    output : :class:`tf.Tensor`
        The complex-valued output :math:`f(z)`.
    """

    output_real = tf.nn.relu(tf.math.real(input))
    output_imag = tf.nn.relu(tf.math.imag(input))

    output = tf.complex(output_real, output_imag)

    return output


@tf.function
def complex_z_relu(input):
    r"""Defines the z-relu function

    .. math::
        f(z) = \begin{cases}
        z & \arg z \in [0, \pi / 2] \\
        0
        \end{cases}

    See `On Complex Valued Convolutional Neural Networks <https://arxiv.org/abs/1602.09046>`_.

    Parameters
    ----------
    input : :class:`tf.Tensor`
        The complex-valued input :math:`z`.

    Returns
    -------
    output : :class:`tf.Tensor`
        The complex-valued output :math:`f(z)`.
    """

    zeros = tf.zeros_like(input)
    phase = tf.math.angle(input)

    c1  = tf.greater(phase, 0.0)
    c2  = tf.less(phase, math.pi / 2.0)
    c_f = tf.logical_and(c1, c2) # 0 < arg z < pi/2

    output = tf.where(c_f, input, zeros)

    return output
