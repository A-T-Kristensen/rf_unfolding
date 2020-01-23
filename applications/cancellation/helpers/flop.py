#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions related to calculating the FLOP values.

For the flop calculations please see:
https://mediatum.ub.tum.de/doc/625604/625604
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
from complex_nn.complex_misc import calc_odd_powers


def flop_convert(n_cadd, n_cmult, algo="default"):
    """Converts a given number of complex-valued additions and complex-valued
    multiplications into the number of real-valued additions and multiplications.

    Parameters
    ----------
    n_cadd  : int
        Number of complex-valued additions.
    n_cmult : int
        Number of complex-valued multiplications.
    algo    : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add   : int
        Number of real-valued additions.
    n_mult  : int
        Number of real-valued multiplications.
    """

    # Every complex mult requires 4 real mult and 2 real add, and every complex
    # addition requires 2 real additions
    if algo == "default":
        n_mult = 4 * n_cmult
        n_add  = 2 * n_cmult + 2 * n_cadd

    # See: https://en.wikipedia.org/wiki/Multiplication_algorithm#Complex_multiplication_algorithm
    # Lower number of multiply operations but overall more operations
    elif algo == "reduced_cmult":
        n_mult = 3 * n_cmult
        n_add  = 5 * n_cmult + 2 * n_cadd
    else:
        raise ValueError("The provided algorithm {} is not supported".format(algo))

    return n_add, n_mult


def flop_complex_layer(n_hidden, n_in, use_bias=False, algo="default"):
    r"""Computes the number of FLOPs for the complex-valued dense layer.
    The number of FLOPs in terms of complex-valued operations is then simply
    given by

    .. math::
        n_{cadd} = n_{hidden} + n_{hidden} * (n_{in} - 1)
        n_{cmult} = n_{hidden} * n_{in}

    For the complex-valued valued network, we can use that every multiplication
    costs four multiplications and two additions (adding the real and imaginary)

    .. math::
        (a+bi)(c+di) = (ac - bd) + (bc + ad)i

    We can then get the result in real by using that for every complex
    multiplication, we need 4 real multiplications and for every complex addition
    we need two additions.

    .. math::
        n_{add}  = 2 n_{cadd} + 2 n_{cmult}
        n_{mult} = 4 n_{cmult}

    If we then compare e.g. a real-valued neural network with 1 hidden layer,
    against a complex-valued equivalent, the number of input nodes to the
    real-valued one will have to be twice as big (one extra for imaginary).

    Considering the formula, we then require that the complex valued network
    uses half the hidden units to get the same overall arithmetic complexity.

    Parameters
    ----------
    n_hidden    : int
        Number of neurons.
    n_in        : int
        Number of inputs.
    use_bias    : bool
        Boolean indicating if bias is used.
    algo        : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions.
    """

    n_cmult = n_hidden * n_in
    n_cadd  = n_hidden * (n_in - 1)

    if use_bias:
        n_cadd += n_hidden

    n_add, n_mult = flop_convert(n_cadd, n_cmult, algo)
    n_activation  = 2 * n_hidden

    return n_add, n_mult, n_activation


def flop_linear_polynomial(params, algo="default"):
    """Computes the number of FLOPs for the linear model.

    Parameters
    ----------
    params  : :obj:
        Parameters.
    algo    : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions (always 0).
    """
    n_add, n_mult, n_act = flop_complex_layer(
        n_hidden = 1,
        n_in     = params.hsi_len,
        use_bias = False,
        algo     = algo
    )

    return n_add, n_mult, 0


def flop_IQMixer(params, calculation="optimistic", algo="default"):
    """Computes the number of FLOPs for the IQ-mixer layer.

    Parameters
    ----------
    params      : :obj:
        Parameters.
    calculation : str
        Specifies whether the calculation is "optimistic" or "pessimistic".
    algo        : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions (always 0).
    """

    # Bypass option
    if params.htnn_struct[0] == "B":
        n_mult = 0
        n_add  = 0

    # Real-valued parameters
    # Calculation is (k1+k2)Re(x) + j(k1-k2)Im(x), we reuse over time
    # so no multiplication with number of channels
    elif params.htnn_struct[0] == "R":
        n_mult = 2
        n_add  = 2

    # For all other complex-valued parameters, assume whatever conversions
    # we have to complex K1, K2 is free.
    else:
        n_cmult = 2
        n_cadd  = 1

        n_add, n_mult = flop_convert(n_cadd, n_cmult, algo)

    # If pessimistic, all results have be computed per channel everytime
    if calculation == "pessimistic":
        n_add  = params.hsi_len * n_add
        n_mult = params.hsi_len * n_mult

    return n_add, n_mult, 0


def flop_hammerstein_basis_functions(params, calculation="optimistic"):
    """Number of FLOPs for the powers of the Hammerstein model.

    We assume the following for calculating powers

    p = 1 --> x_iq, so n_add = 0, n_mult = 0

    p = 3 --> x_iq |x_iq|^2 = x_iq*(Re(x_iq)^2 + Im(x_iq)^2),
    so n_add = 1, n_mult = 2 + 2 (2 for |x_iq| powers and 2 for mult with complex)

    p = 5 --> (x_iq |x_iq|^2) |x_iq|^2 (reuse above) , so n_add = 0, n_mult = 2

    Therefore, after p=3 we have a contribution for 2 mult for every odd power.

    Parameters
    ----------
    params      : :obj:
        Parameters.
    calculation : str
        Specifies whether the calculation is "optimistic" or "pessimistic".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions (always 0).
    """

    if params.max_power == 1:
        n_add  = 0
        n_mult = 0
    else:
        n_add  = 1
        n_mult = 2 + params.max_power - 1

    if calculation == "pessimistic":
        n_add  = params.hsi_len * n_add
        n_mult = params.hsi_len * n_mult

    return n_add, n_mult, 0


def flop_hammerstein(params, algo="default"):
    """Calculate the number of FLOPs when multiplying and adding all the
    power values with the coefficients.

    We simply consider this step as a normal complex-valued feed forward layer,
    with an input vector of size number of odd powers times the channel memory
    and no bias.

    Parameters
    ----------
    params      : :obj:
        Parameters.
    algo        : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions (always 0).
    """

    odd_powers = calc_odd_powers(params.min_power, params.max_power)

    n_hidden = 1
    n_in     = params.hsi_len * odd_powers

    n_cmult = n_hidden * n_in
    n_cadd  = n_hidden * (n_in - 1)

    n_add, n_mult = flop_convert(n_cadd, n_cmult, algo)

    return n_add, n_mult, 0


def flop_model_based_nn(params, calculation="optimistic", algo="default"):
    """Function computing the total number of FLOPs for the Hammerstein model.

    Parameters
    ----------
    params      : :obj:
        Parameters.
    calculation : str
        Specifies whether the calculation is "optimistic" or "pessimistic".
    algo        : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions (always 0).
    """

    l1_add, l1_mult, l1_act = flop_IQMixer(params, calculation, algo)
    l2_add, l2_mult, l2_act = flop_hammerstein_basis_functions(params, calculation)
    l3_add, l3_mult, l3_act = flop_hammerstein(params, algo)

    n_add        = l1_add + l2_add + l3_add
    n_mult       = l1_mult + l2_mult + l3_mult
    n_activation = l1_act + l2_act + l3_act

    return n_add, n_mult, n_activation


def flop_realnn(model, **kwargs):
    """Computes the total number of FLOPs for the real-valued FFNN.
    For the real valued network, the number of flop is simply given by

    .. math::
        n_{add} = n_{hidden} + n_{hidden} * (n_{in} - 1)
        n_{mult} = n_{hidden} * n_{in}

    For every layer, for n_add the extra n_hidden term is for the bias

    Parameters
    ----------
    model : :obj:
        The Tensorflow model.

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions.
    """

    flop_add        = 0
    flop_mult       = 0
    flop_activation = 0

    for layer in model.layers:
        if layer.__class__.__name__ == "Dense":

            n_hidden = layer.units
            n_in     = layer.input_shape[-1]

            flop_mult += n_hidden * n_in
            flop_add  += n_hidden * (n_in - 1)

            if layer.use_bias:
                flop_add += n_hidden

            if layer.activation is not None:
                flop_activation += n_hidden

    # Subtract the last activation as linear
    flop_activation -= n_hidden

    return flop_add, flop_mult, flop_activation


def flop_complexnn(model, algo="default"):
    """Computes the complexity of the complex-valued NN.

    Parameters
    ----------
    model : :obj:
        The Tensorflow model.
    algo  : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions.
    """

    flop_add        = 0
    flop_mult       = 0
    flop_activation = 0

    for layer in model.layers:
        if layer.__class__.__name__ == "ComplexDense":

            tmp_add, tmp_mult, tmp_activation = flop_complex_layer(layer.units, layer.input_dim, layer.use_bias, algo)

            flop_add        += tmp_add
            flop_mult       += tmp_mult
            flop_activation += tmp_activation

    # Subtract the last activation as linear layer
    flop_activation -= tmp_activation

    return flop_add, flop_mult, flop_activation


def flop_rnn_simple_layer(n_hidden, n_in, use_bias, activation):
    r"""Computes the total number of FLOPs for a simple RNN layer.

    Parameters
    ----------
    n_hidden    : int
        Number of neurons.
    n_in        : int
        Number of inputs.
    use_bias    : bool
        Boolean indicating if bias is used.
    algo        : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions.
    """

    # We have the matrix for inputs and the recurrent kernel
    n_add  = (n_hidden * (n_in - 1)) + (n_hidden * (n_hidden - 1))
    n_mult = (n_hidden * n_in) + (n_hidden * n_hidden)

    # We add x_t*K to h_{t-1}*K_rec
    n_add += n_hidden

    # There is only one bias in this implementation
    if use_bias:
        n_add += n_hidden

    if activation is not None:
        n_activation = n_hidden
    else:
        n_activation = 0

    return n_add, n_mult, n_activation


def flop_rnn(model, **kwargs):
    r"""Computes the total number of FLOPs for an RNN.

    Parameters
    ----------
    model : :obj:
        The Tensorflow model.

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions.
    """

    flop_add        = 0
    flop_mult       = 0
    flop_activation = 0

    for layer in model.layers:
        if layer.__class__.__name__ == "Dense":
            n_hidden = layer.units
            n_in     = layer.input_shape[-1]

            flop_mult_tmp = n_hidden * n_in
            flop_add_tmp  = n_hidden * (n_in - 1)

            if layer.use_bias:
                flop_add_tmp += n_hidden

            if layer.activation is not None:
                flop_activation_tmp = n_hidden

        elif layer.__class__.__name__ == "GRU":
            n_hidden = layer.units
            n_in     = layer.input_shape[-1]

            # Count for candidate, reset gate and update gate, we have
            # two matrix multiplications for each
            # So 3 gates with two matrix multiplications each
            flop_add_tmp  = 3 * (n_hidden * (n_in - 1)) + 3 * (n_hidden * (n_hidden - 1))
            flop_mult_tmp = 3 * (n_hidden * n_in) + 3 * (n_hidden * n_hidden)

            # Add flop mult and add for weighted average of candidate and input
            # We do two hadamard products and one subtraction and one addition
            flop_add_tmp  += 2 * n_hidden
            flop_mult_tmp += 3 * n_hidden

            # Each gate and the candiate has a bias
            if layer.use_bias:
                flop_add_tmp += 3 * n_hidden

            if layer.activation is not None:
                flop_activation_tmp = 3 * n_hidden

        # Simple RNN is simply a feed-forward unrolled
        # https://keras.io/layers/recurrent/
        # https://peterroelants.github.io/posts/rnn-implementation-part01/
        elif layer.__class__.__name__ == "SimpleRNN":
            n_hidden   = layer.units
            n_in       = layer.input_shape[-1]
            use_bias   = layer.use_bias
            activation = layer.activation

            flop_add_tmp, flop_mult_tmp, flop_activation_tmp = flop_rnn_simple_layer(n_hidden, n_in, use_bias, activation)

        flop_add        += flop_add_tmp
        flop_mult       += flop_mult_tmp
        flop_activation += flop_activation_tmp

    # Subtract the last activation as linear
    flop_activation -= flop_activation_tmp

    return flop_add, flop_mult, flop_activation


def flop_complexrnn(model, algo="default", **kwargs):
    r"""Computes the total number of FLOPs for a comples-valued RNN.

    Parameters
    ----------
    model : :obj:
        The Tensorflow model.
    algo  : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".

    Returns
    -------
    n_add           : int
        Number of real-valued additions.
    n_mult          : int
        Number of real-valued multiplications.
    n_activation    : int
        Number of applications of activation functions.
    """

    flop_add        = 0
    flop_mult       = 0
    flop_activation = 0

    for layer in model.layers:
        if layer.__class__.__name__ == "ComplexDense":

            tmp_add, tmp_mult, tmp_activation = flop_complex_layer(layer.units, layer.input_dim, layer.use_bias, algo)

            flop_add        += tmp_add
            flop_mult       += tmp_mult
            flop_activation += tmp_activation

        elif layer.__class__.__name__ == "SimpleComplexRNN":
            n_hidden   = layer.units

            n_in       = layer.cell.input_dim
            use_bias   = layer.use_bias
            activation = layer.activation

            flop_add_tmp, flop_mult_tmp, flop_activation_tmp = flop_rnn_simple_layer(n_hidden, n_in, use_bias, activation)
            flop_add_tmp, flop_mult_tmp = flop_convert(flop_add_tmp, flop_mult_tmp, algo)
            flop_activation_tmp *= 2

        flop_add        += flop_add_tmp
        flop_mult       += flop_mult_tmp
        flop_activation += flop_activation_tmp

    # Subtract the last activation as linear
    flop_activation -= flop_activation_tmp

    return flop_add, flop_mult, flop_activation
