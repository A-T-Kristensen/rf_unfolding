#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source code defining initializers for the complex-valued and model-based neural
networks. Note that for the case of a feed-forward network, you can simply use
the normal weight initializers and we simply scale the variance by 1/2.

Initializers are based on
https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/initializers/VarianceScaling
"""

from tensorflow.keras.initializers import VarianceScaling

import math
import numpy as np

import scipy.integrate as integrate

from complex_nn.complex_misc import calc_odd_powers


class ComplexGlorotNormal(VarianceScaling):
    r"""Complex-valued equivalent of the glorot uniform initializer.
    This intitializer is used for the real- and imaginary-parts independently.

    The variance of a complex-valued random variable is a real number
    (non-negative), equal to the sum of the variances of the real- and imaginary-
    parts of the complex random variable.

    .. math::
        \mathrm{Var}[Z]=\mathrm{Var} [\Re {(Z)}] + \mathrm{Var} [\Im {(Z)}]

    Considering the derivation in `Glorot et al., 2010 <http://proceedings.mlr.press/v9/glorot10a.html>`_, the real- and imaginary-parts are sampled from Gaussians with :math:`\mu=0` and :math:`\sigma^2=1 / \sqrt{(fan_{in} + fan_{out})}`.
    Therefore, the normal intializers can simply be used, with a modification to the scaling parameter.

    Arguments
    ---------
    scale           : float
        Scaling factor (positive float). This is the variance.
    mode            : str
        One of "fan_in", "fan_out", "fan_avg".
    distribution    : str
        Random distribution to use. One of "truncated_normal", "untruncated_normal" and "uniform".
    seed            : int
        A Python integer. Used to create random seeds.
    """

    def __init__(self, seed=None):
        super(ComplexGlorotNormal, self).__init__(
            scale        = 0.5,
            mode         = "fan_avg",
            distribution = "truncated_normal",
            seed         = seed
        )

    def get_config(self):
        return {"seed": self.seed}


class ComplexGlorotUniform(VarianceScaling):
    """Complex-valued equivalent of the glorot uniform initializer.

    Arguments
    ---------
    scale           : float
        Scaling factor (positive float). This is the variance.
    mode            : str
        One of "fan_in", "fan_out", "fan_avg".
    distribution    : str
        Random distribution to use. One of "truncated_normal", "untruncated_normal" and "uniform".
    seed            : int
        A Python integer. Used to create random seeds.
    """

    def __init__(self, seed=None):
        super(ComplexGlorotUniform, self).__init__(
            scale        = 0.5,
            mode         = "fan_avg",
            distribution = "uniform",
            seed         = seed
        )

    def get_config(self):
        return {"seed": self.seed}


class ComplexHeNormal(VarianceScaling):
    """Complex-valued equivalent of the glorot uniform initializer.

    Arguments
    ---------
    scale           : float
        Scaling factor (positive float). This is the variance.
    mode            : str
        One of "fan_in", "fan_out", "fan_avg".
    distribution    : str
        Random distribution to use. One of "truncated_normal", "untruncated_normal" and "uniform".
    seed            : int
        A Python integer. Used to create random seeds.
    """

    def __init__(self, seed=None):
        super(ComplexHeNormal, self).__init__(
            scale        = 1.0,
            mode         = "fan_in",
            distribution = "truncated_normal",
            seed         = seed
        )

    def get_config(self):
        return {"seed": self.seed}


class ComplexHeUniform(VarianceScaling):
    """Complex-valued equivalent of the glorot uniform initializer.

    Arguments
    ---------
    scale           : float
        Scaling factor (positive float).
    mode            : str
        One of "fan_in", "fan_out", "fan_avg".
    distribution    : str
        Random distribution to use. One of "truncated_normal", "untruncated_normal" and "uniform".
    seed            : int
        A Python integer. Used to create random seeds.
    """

    def __init__(self, seed=None):
        super(ComplexHeUniform, self).__init__(
            scale        = 1.0,
            mode         = "fan_in",
            distribution = "uniform",
            seed         = seed
        )

    def get_config(self):
        return {"seed": self.seed}


def hammerstein_initializer(
    n_channels,
    min_power,
    max_power,
    K1           = None,
    distribution = "hammerstein_rayleigh",
    dtype        = np.complex64
):
    r"""Hammerstein type neural network weight initializer

    We ensure that the variance remains close to the input variance by
    initializing the weights for this, requiring that the variance of each term
    in the sum is the same therefore we need to scale the weights accordingly.

    Parameters
    ----------
    n_channels      : int
        Number of channels for the Hammerstein network
    min_power       : int
        Minimum power for the network
    max_power       : int
        Maximum power for the network
    distribution    : str
        Probability density to sample from (rayleigh, normal, uniform)
    dtype           : :obj:
        Datatype of the weights

    Returns
    -------
    weights : :class:`numpy.ndarray`
        Numpy array of initialized weights
    """
    odd_powers = calc_odd_powers(min_power, max_power)

    weights = np.empty((odd_powers, n_channels), dtype=dtype)

    if distribution == "hammerstein_rayleigh":
        weights = sample_rayleigh(weights, n_channels, min_power, max_power, K1)
    elif distribution == "hammerstein_normal":
        weights = sample_normal(weights, n_channels, min_power, max_power, K1)
    elif distribution == "hammerstein_uniform":
        weights = sample_uniform(weights, n_channels, min_power, max_power, K1)

    weights = weights.flatten()
    weights = np.expand_dims(weights, axis=0) # Shape (1, *weights.size)

    return weights


def sample_rayleigh(weights, n_channels, min_power, max_power, K1):
    r"""Initialize weights in the parallel Hammerstein layer by sampling from a
    Rayleigh distribution.

    The variance of a complex-valued random variable :math:`z=x+iy` with mean 0
    is

    .. math::
        \mathrm{Var}[Z]= \mathrm{E}[|Z|^2]

    where :math:`z\sim \mathrm{Rayleigh}(\sigma)` with
    :math:`X\sim N(0, \sigma^2)` and :math:`Y\sim N(0, \sigma^2)` as indepdendent
    normal random variables.

    Using this, we have

    .. math::
        \mathrm{Var}[|Z|] = \mathrm{Var}[Z] - \mathrm{E}[|Z|]^2

    and the magnitude of the complex-variables can be sampled from a Rayleigh
    distribution. The phases are sampled from a uniform distribution.

    Parameters
    ----------
    n_channels  : int
        Number of channels for the Hammerstein network,
    min_power   : int
        Minimum power for the network,
    max_power   : int
        Maximum power for the network,
    K1          : :class:`numpy.ndarray`
        Value of K1 (complex-valued).
    dtype       : :obj:
        Datatype of the weights,

    Returns
    -------
    weights : :class:`numpy.ndarray`
        Numpy array of initialized weights,
    """

    for p in range(min_power, max_power + 1, 2):
        # Sample the magnitude from the Rayleigh distribution
        std_re = compute_std_re(weights.size, p, K1)

        idx = int((p + 1) / 2) - int((min_power + 1) / 2)
        weights[idx] = np.random.rayleigh(std_re, n_channels)

        # Add the phases from a uniform distribution from -Pi to Pi
        phases = np.random.uniform(low=-math.pi, high=math.pi, size=n_channels)
        weights[idx] *= np.exp(1j * phases)

    return weights


def sample_normal(weights, n_channels, min_power, max_power, K1):
    """Initialize weights in the parallel Hammerstein layer by sampling from a
    normal distribution.

    Parameters
    ----------
    n_channels  : int
        Number of channels for the Hammerstein network,
    min_power   : int
        Minimum power for the network,
    max_power   : int
        Maximum power for the network,
    K1          : :class:`numpy.ndarray`
        Value of K1 (complex-valued).
    dtype       : :obj:
        Datatype of the weights,

    Returns
    -------
    weights : :class:`numpy.ndarray`
        Numpy array of initialized weights
    """

    for p in range(min_power, max_power + 1, 2):
        std_re = compute_std_re(weights.size, p, K1)

        tmp_re = np.random.normal(loc=0, scale=std_re, size=n_channels)
        tmp_im = np.random.normal(loc=0, scale=std_re, size=n_channels)

        idx = int((p + 1) / 2) - int((min_power + 1) / 2)

        weights[idx] = tmp_re + 1j * tmp_im

    return weights


def sample_uniform(weights, n_channels, min_power, max_power, K1):
    """Initialize weights in the parallel Hammerstein layer by sampling from a
    uniform distribution.

    Parameters
    ----------
    n_channels  : int
        Number of channels for the Hammerstein network,
    min_power   : int
        Minimum power for the network,
    max_power   : int
        Maximum power for the network,
    K1          : :class:`numpy.ndarray`
        Value of K1 (complex-valued).
    dtype       : :obj:
        Datatype of the weights,

    Returns
    -------
    weights : :class:`numpy.ndarray`
        Numpy array of initialized weights
    """

    for p in range(min_power, max_power + 1, 2):
        std_re = compute_std_re(weights.size, p, K1)

        unif_lim = np.sqrt(3) * std_re

        tmp_re = np.random.uniform(low=-unif_lim, high=unif_lim, size=n_channels)
        tmp_im = np.random.uniform(low=-unif_lim, high=unif_lim, size=n_channels)

        idx = int((p + 1) / 2) - int((min_power + 1) / 2)

        weights[idx] = tmp_re + 1j * tmp_im

    return weights


def compute_std_re(size, power, K1):
    r"""Returns the standard-deviation of the real/imaginary components,
    assuming these are equal.

    The variance of a complex-valued random variable is a real number
    (non-negative).
    It is equal to the sum of the variances of the real and imaginary part
    of the complex random variable.

    .. math::
        \mathrm{Var}[Z]=\mathrm{Var} [\Re {(Z)}] + \mathrm{Var} [\Im {(Z)}]

    Therefore, if we want e.g. the variance of the complex-valued random
    variable to be 1, we have to scale the variance of the real and imaginary
    part by 1/2.

    Parameters
    ----------
    size    : int
        Number of weights to generate (N_powers*N_channels).
    power   : int
        Current power.
    K1      : :class:`numpy.ndarray`
        Value of K1 (complex-valued).

    Returns
    -------
    std_re : float
        We return the standard-deviation of the real-part divided by the
        square-root of two.
    """

    # In case we do not have an IQ-mixer layer
    if K1 is None:
        e_xiq_p = math.factorial(power)
    else:
        e_xiq_p = math.factorial(power) * np.abs(K1)**(2 * power)

    std_z  = np.sqrt(1 / size * 1 / e_xiq_p) # For complex
    std_re = std_z / np.sqrt(2) # For real/imaginary part correction

    return std_re
