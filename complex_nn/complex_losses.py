#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source code defining cost-functions for the complex-valued neural networks

The losses in tensorflow are defined here
https://www.tensorflow.org/api_docs/python/tf/losses
"""

import tensorflow as tf


@tf.function
def complex_cosh_log_loss(target, estimate, a=2):
    r"""Class defining the complex-valued CoshLog Loss function

    In the 1D case,

    .. math::
        \mathrm{C}(z,t) = \frac{1}{2} \ln \left( \frac{\exp(|t-z|) + \exp(-|t-z|)}{2} \right)

    See: https://openreview.net/pdf?id=rkglvsC9Ym

    Parameters
    ----------
    target      : :class:`tf.Tensor`
        The target of the model.
    estimate    : :class:`tf.Tensor`
        The estimate of the model.
    a           : float
        Scaling parameter for the loss.

    Returns
    -------
    loss : :class:`tf.Tensor`
        The loss value (real-valued).
    """

    residuals = tf.math.abs(target - estimate)
    log_expr  = tf.math.exp(a * residuals) + tf.math.exp(-a * residuals)
    loss      = 1 / a * tf.reduce_mean(tf.math.log(log_expr / 2))

    return loss


@tf.function
def complex_mean_log_loss(target, estimate):
    r"""Class defining the logarithmic performance index

    .. math::
        \mathrm{C}(z,t) = \frac{1}{2N} \log(t/z)^H \log(t/z)

    when t, z has size N

    in the 1D case, this simplifies to

    .. math::
        \mathrm{C}(z,t) = 1/2 \log(t/z)* \log(t/z)^*

    Parameters
    ----------
    target      : :class:`tf.Tensor`
        The target of the model.
    estimate    : :class:`tf.Tensor`
        The estimate of the model.

    Returns
    -------
    loss : :class:`tf.Tensor`
        The loss value (real-valued).
    """

    loss = complex_mean_squared_error(tf.math.log(target), tf.math.log(estimate))

    return loss


@tf.function
def complex_mean_squared_error(target, estimate):
    r"""Function defining the complex-valued mean square error

    .. math::
        \mathrm{C}(z,t) = \frac{1}{2N} (t-z)^H (t-z)

    when t, z has size N.

    In the 1D case, this simplifies to

    .. math::
        \mathrm{C}(z,t) = \frac{1}{2} (t-z)(t-z)^* =  \frac{1}{2} |t-z|^2

    Parameters
    ----------
    target      : :class:`tf.Tensor`
        The target of the model.
    estimate    : :class:`tf.Tensor`
        The estimate of the model.

    Returns
    -------
    loss : :class:`tf.Tensor`
        The loss value (real-valued).
    """

    # DO NOT MODIFY: Other code relies on this not being divided with e.g. 2
    residuals = tf.math.abs(target - estimate)
    loss      = tf.reduce_mean(tf.square(residuals))

    return loss


@tf.function
def complex_psedu_huber_loss(target, estimate, delta=2.0):
    r"""Class defining the complex-valued pseudo Huber loss

    In the 1D case,

    .. math::
        \mathrm{C}(z,t) = \delta^2 \left( \sqrt{1+ \left(\frac{|t-z|}{\delta} \right)^2} -1 \right)

    Parameters
    ----------
    target      : :class:`tf.Tensor`
        The target of the model.
    estimate    : :class:`tf.Tensor`
        The estimate of the model.
    delta       : float
        Scaling parameter for the loss.

    Returns
    -------
    loss : :class:`tf.Tensor`
        The loss value (real-valued).
    """

    residuals = tf.math.abs(target - estimate)
    loss      = tf.reduce_mean(tf.square(delta) * (tf.sqrt(1 + (residuals / delta)**2) - 1))

    return loss
