#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for polynomial cancellation.
"""

import numpy as np
import sys


def si_estimation_linear(x, y, params):
    """Estimates parameters for linear cancellation.

    Parameters
    ----------
    x       : :class:`numpy.ndarray`
        Input array.
    y       : :class:`numpy.ndarray`
        Output array.
    params  : :obj:
        Parameters.

    Returns
    -------
    h : :class:`numpy.ndarray`
        Parameters for the linear canceller.
    """

    chan_len = params.hsi_len

    # Construct LS problem
    A = np.reshape(
        [np.flip(x[i + 1:i + chan_len + 1], axis=0) for i in range(x.size - chan_len)],
        (x.size - chan_len, chan_len)
    )

    # Solve LS problem
    h = np.linalg.lstsq(A, y[chan_len:], rcond=None)[0]

    return h


def si_estimation_nonlinear(x, y, params):
    """Estimates parameters for non-linear cancellation (memory polynomial +
    IQ mixer).

    Parameters
    ----------
    x       : :class:`numpy.ndarray`
        Input array.
    y       : :class:`numpy.ndarray`
        Output array.
    params  : :obj:
        Parameters.

    Returns
    -------
    h : :class:`numpy.ndarray`
        Parameters for the non-linear canceller.
    """

    print("Self-interference channel estimation:")

    # Get PA non-linearity parameters
    pamaxorder        = params.max_power
    chan_len          = params.hsi_len
    n_basis_functions = int((pamaxorder + 1) / 2 * ((pamaxorder + 1) / 2 + 1))

    # Apply PA non-linearities
    A = np.zeros((x.size - chan_len, n_basis_functions * chan_len), dtype=np.complex128)

    mat_ind = 0

    for i in range(1, pamaxorder + 1, 2):
        for j in range(0, i + 1):
            sys.stdout.write("\r1. Constructing basis functions... ({:d}/{:d})".format(int(mat_ind + 1), n_basis_functions))
            sys.stdout.flush()
            xnl = np.power(x, j) * np.power(np.conj(x), i - j)
            A[:, mat_ind * chan_len:(mat_ind + 1) * chan_len] = np.reshape([np.flip(xnl[i + 1:i + chan_len + 1], axis=0) for i in range(xnl.size - chan_len)], (xnl.size - chan_len, chan_len))
            mat_ind += 1

    sys.stdout.write("\r1. Constructing basis functions... done!         \n")

    # Solve LS problem
    sys.stdout.write("2. Doing channel estimation... ")
    sys.stdout.flush()
    h = np.linalg.lstsq(A, y[chan_len:])[0]
    sys.stdout.write("done! Estimated a total of {:d} parameters.\n".format(h.size * 2))

    return h


def si_estimation_hammerstein(x, y, params):
    """Estimates parameters for non-linear cancellation for the Hammerstein
    model, that is, we do not consider the effect of the IQ-mixer.

    Parameters
    ----------
    x       : :class:`numpy.ndarray`
        Input array.
    y       : :class:`numpy.ndarray`
        Output array.
    params  : :obj:
        Parameters.

    Returns
    -------
    h : :class:`numpy.ndarray`
        Parameters for the non-linear canceller.
    """

    print("Self-interference channel estimation:")

    # Get PA non-linearity parameters
    pamaxorder        = params.max_power
    chan_len          = params.hsi_len
    n_basis_functions = int((pamaxorder + 1) / 2)

    # Apply PA non-linearities
    A = np.zeros((x.size - chan_len, n_basis_functions * chan_len), dtype=np.complex128)

    mat_ind = 0

    for i in range(1, pamaxorder + 1, 2):
        sys.stdout.write("\r1. Constructing basis functions... ({:d}/{:d})".format(int(mat_ind + 1), n_basis_functions))
        sys.stdout.flush()
        xnl = x * np.abs(x)**(i - 1)
        A[:, mat_ind * chan_len:(mat_ind + 1) * chan_len] = np.reshape([np.flip(xnl[i + 1:i + chan_len + 1], axis=0) for i in range(xnl.size - chan_len)], (xnl.size - chan_len, chan_len))
        mat_ind += 1

    sys.stdout.write("\r1. Constructing basis functions... done!         \n")

    # Solve LS problem
    sys.stdout.write("2. Doing channel estimation... ")
    sys.stdout.flush()
    h = np.linalg.lstsq(A, y[chan_len:])[0]
    sys.stdout.write("done! Estimated a total of {:d} parameters.\n".format(h.size * 2))

    return h


def si_cancellation_linear(x, h, params):
    """Perform linear cancellation based on estimated parameters.

    Parameters
    ----------
    x       : :class:`numpy.ndarray`
        Input array.
    h       : :class:`numpy.ndarray`
        Parameter for the linear canceller.
    params  : :obj:
        Parameters.

    Returns
    -------
    x_can : :class:`numpy.ndarray`
        Linear cancellation results.
    """

    # Calculate the cancellation signal
    x_can = np.convolve(x, h, mode='full')
    x_can = x_can[0:x.size]

    return x_can


def si_cancellation_nonlinear(x, h, params):
    """Perform non-linear cancellation based on estimated parameters.

    Parameters
    ----------
    x       : :class:`numpy.ndarray`
        Input array.
    h       : :class:`numpy.ndarray`
        Parameters for the non-linear canceller.
    params  : :obj:
        Parameters.

    Returns
    -------
    x_can : :class:`numpy.ndarray`
        Non-linerar cancellation results.
    """

    # Get parameters
    pamaxorder        = params.max_power
    chan_len          = params.hsi_len
    n_basis_functions = int((pamaxorder + 1) / 2 * ((pamaxorder + 1) / 2 + 1))

    # Calculate the cancellation signal
    xnl   = x
    x_can = np.zeros(x.size + chan_len - 1, dtype=np.complex128)

    chanInd = 0
    for i in range(1, pamaxorder + 1, 2):
        for j in range(0, i + 1):
            sys.stdout.write("\r1. Constructing basis functions and cancellation signal... ({:d}/{:d})".format(int(chanInd + 1), int(n_basis_functions)))
            sys.stdout.flush()
            xnl = np.power(x, j) * np.power(np.conj(x), i - j)
            x_can += np.convolve(xnl, h[chanInd * chan_len:(chanInd + 1) * chan_len])
            chanInd += 1

    x_can = x_can[0:x.size]

    return x_can


def si_cancellation_hammerstein(x, h, params):
    """Perform non-linear cancellation based on estimated parameters.
    This uses the memory polynomial.

    Parameters
    ----------
    x       : :class:`numpy.ndarray`
        Input array.
    h       : :class:`numpy.ndarray`
        Parameters for the non-linear canceller.
    params  : :obj:
        Parameters.

    Returns
    -------
    x_can : :class:`numpy.ndarray`
        Non-linerar cancellation results.
    """

    # Get parameters
    pamaxorder        = params.max_power
    chan_len          = params.hsi_len
    n_basis_functions = int((pamaxorder + 1) / 2)

    # Calculate the cancellation signal
    xnl   = x
    x_can = np.zeros(x.size + chan_len - 1, dtype=np.complex128)

    chanInd = 0
    for i in range(1, pamaxorder + 1, 2):
        sys.stdout.write("\r1. Constructing basis functions and cancellation signal... ({:d}/{:d})".format(int(chanInd + 1), int(n_basis_functions)))
        sys.stdout.flush()
        xnl = x * np.abs(x)**(i - 1)
        x_can += np.convolve(xnl, h[chanInd * chan_len:(chanInd + 1) * chan_len])
        chanInd += 1

    x_can = x_can[0:x.size]

    return x_can
