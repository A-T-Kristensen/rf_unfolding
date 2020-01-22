#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions related to post-processing of data for the full-duplex application,
generating different statistics and so forth.
"""

import numpy as np
import os


def gen_new_path(path):
    """Generates the new path for all the plots
    """
    # Remove extension and separate by os seperator
    new_path = os.path.splitext(path)[0]
    new_path = new_path.split(os.sep)

    # Get folder and remove, will be part of new file namee
    folder = new_path[-2]
    del new_path[-2]

    file_name = new_path[-1]
    new_file_name = file_name.split("_")
    new_file_name.insert(1, folder)

    # Join new file name into path and join path
    new_path[-1] = "_".join(new_file_name)
    new_path = os.sep.join(new_path) + ".png"

    return new_path


def check_array(input_array):
    """Verifies array has correct type and shape for use with gen_stat(),
    that is, the array has at-least 2 dimensions and is a numpy array.

    Parameters
    ----------
    input_array : :class:`numpy.ndarray`
        Input array.

    Returns
    -------
    y : :class:`numpy.ndarray`
        Modified or original input array.
    """

    input_array = np.array(input_array)

    if len(input_array.shape) > 2:
        raise ValueError("The provided input array has more than 2 dimensions.")
    elif len(input_array.shape) < 2:
        return np.expand_dims(input_array, axis=0)
    else:
        return input_array


def gen_array(input_array, field):
    """Takes an array of dictionaries with each dictionary containing arrays.
    The function generates an array of values with the specified field/key, that
    is, it extracts all sub-arrays of same type form the different dictionaries.

    Parameters
    ----------
    input_array : :obj:
        Input array of dictionaries.
    field       : str
        Field to search for.
    Returns
    -------
    y : :class:`numpy.ndarray`
        Collection of data of same field.
    """

    return np.array([res[field] for res in input_array])


def gen_stat(loss_array, var_array, target, seed_list):
    """Function which generates the statistics used for the results in the CSV
    files.

    Expects result arrays to be given for multiple seeds (N_seeds,N_epochs)
    (or array structured to have shape (1,N_epochs) for 1 seed)

    Parameters
    ----------
    loss_array : :class:`numpy.ndarray`
        Array of mean loss values on test-set across seeds and epochs.
    var_array : :class:`numpy.ndarray`
        Array for the variance of the loss values on test-set across seeds and epochs.
    target : :class:`numpy.ndarray`
        Input array of dictionaries.
    seed_list : :obj:
        List of seeds used for initialization.
    Returns
    -------
    y : :obj:
        Dictionary containing main results.
    """

    # Check arrays have correct type
    loss_array = check_array(loss_array)
    var_array  = check_array(var_array)

    # Compute the average mean for last epoch (final result) over seeds
    mean_loss = np.mean(loss_array[:, -1])
    min_loss  = np.min(loss_array[:, -1])

    # Compute the variance of the loss per seed (variance of mean of |r|^2)
    # Variance 0 would indicate the seed has no effect
    var_mean_loss = np.var(loss_array[:, -1])

    # Compute the variance of the magnitude of the residuals squared
    # Quantifies amount of variation of |t-y|^2
    # This formula works for both different variances and means
    var_loss = np.mean(loss_array[:, -1]**2 + var_array[:, -1] - mean_loss**2)

    # Cancellation for all results
    canc_array = 10 * np.log10(np.mean(np.abs(target)**2)) - 10 * np.log10(loss_array)

    # Average cancellation over seeds
    mean_canc = np.mean(canc_array[:, -1])
    max_canc  = np.max(canc_array[:, -1])

    # Variance of cancellation per seed.
    # Quantifies effect of seed on mean performance
    var_canc = np.var(canc_array[:, -1])

    # Mean cancellation by first averaging loss in linear domain and then converting to dB
    mean_canc_lin = 10 * np.log10(np.mean(np.abs(target)**2) / np.mean(mean_loss))

    # Optimal seed gives the lowest loss for last epochs
    opt_seed = seed_list[np.nanargmin(loss_array[:, -1])]

    # Let r denote the residual values
    result_dict = {
        "loss_array"   : loss_array,    # Mean of |r|^2 for different seeds and epochs
        "var_array"    : var_array,     # Variation of |r|^2 for different seeds and epochs
        "canc_array"   : canc_array,    # Cancellation for different seeds and epochs
        "min_loss"     : min_loss,      # Best possible loss for a seed at last epochs
        "mean_loss"    : mean_loss,     # Average loss over seeds at last epochs
        "var_mean_loss": var_mean_loss, # Variation of the mean loss across seeds
        "var_loss"     : var_loss,      # Variation of the mixture distribution of |r|^2 for different seeds
        "max_canc"     : max_canc,      # Best possible cancellation
        "mean_canc"    : mean_canc,     # Average cancellation over seeds
        "var_mean_canc": var_canc,      # Variation of mean cancellation accross seeds
        "mean_canc_lin": mean_canc_lin, # Mean cancellation when avg computed in linear
        "opt_seed"     : opt_seed
    }

    return result_dict
