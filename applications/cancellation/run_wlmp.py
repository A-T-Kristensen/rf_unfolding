#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fit the polynomial the full-duplex training set
"""

import os
from collections import OrderedDict

import numpy as np
import helpers.signal_processing as sp
import helpers.polynomial_cancellation as poly

from helpers.preprocess import basic_parser, load_data
from helpers.flop import flop_linear_polynomial, flop_convert

import pandas as pd


def fit(params):
    """Fit the polynomial model which includes the IQ mixer.
    """

    ##### Load and prepare data #####
    x, y, noise, measured_noise_power = load_data('data/fdTestbedData' + str(params.sampling_freq_MHz) + 'MHz10dBm', params)

    # Print total number of real parameters to be estimated
    # This number includes the linear as well
    n_poly = int(params.hsi_len * ((params.max_power + 1) / 2) * ((params.max_power + 1) / 2 + 1))
    print("Total number of real parameters to estimate for polynomial based canceller: {:d}".format(2 * n_poly))

    # Split into training and test sets
    training_samples = int(np.floor(x.size * params.training_ratio))

    x_train = x[0:training_samples]
    y_train = y[0:training_samples]
    x_test  = x[training_samples:]
    y_test  = y[training_samples:]

    # Remove samples when training with less samples
    training_samples = int(np.floor(training_samples * params.training_size))

    x_train = x[0: training_samples]
    y_train = y[0: training_samples]

    ##### Training #####
    # Estimate linear and non-linear cancellation parameters
    h_lin    = poly.si_estimation_linear(x_train, y_train, params)
    # This actually also estimates the linear! So we get both
    h_nonlin = poly.si_estimation_nonlinear(x_train, y_train, params)

    ##### Test #####
    # Do linear and non-linear cancellation
    y_canc        = poly.si_cancellation_linear(x_test, h_lin, params)
    # NB: This actually contains the linear cancellation AND the non-linear
    # cancellation.
    y_canc_nonlin = poly.si_cancellation_nonlinear(x_test, h_nonlin, params)

    ##### Evaluation #####
    # Scale signals according to known noise power
    y_test, y_canc, y_canc_nonlin, noise = sp.compute_scaling(noise, measured_noise_power, y_test, y_canc, y_canc_nonlin)

    if params.save:
        path = "results/results_wlmp"
    else:
        path = "results/tmp_wlmp"

    if not os.path.exists(path):
        os.makedirs(path)

    # Plot PSD and get signal powers
    noise_power, y_test_power, y_test_lin_canc_power, y_test_nonlin_canc_power = sp.plotPSD(
        params,
        y_test[params.hsi_len:],
        y_canc[params.hsi_len:],
        y_canc_nonlin[params.hsi_len:],
        noise,
        y_var=1,
        path=path
    )

    # We actually have that it performs the total cancellation!
    # model_canc then refers to the difference in performance between the linear
    # and non-linear canceller (where non-linear includes linear)
    model_canc = y_test_lin_canc_power - y_test_nonlin_canc_power
    # Total cancellation of model
    total_canc = y_test_power - y_test_nonlin_canc_power

    # Complexity (where we subtract the linear! and then we add the linear back later)
    n_cadd = n_poly - params.hsi_len - 1
    n_cmult = n_poly - params.hsi_len

    # Take result in terms of complex-valued additions, multiplications and convert
    # to total real-valued ops.
    model_add, model_mult = flop_convert(n_cadd, n_cmult, algo="reduced_cmult")
    model_act = 0

    model_flop = model_add + model_mult

    total_add  = model_add
    total_mult = model_mult
    total_act  = 0

    lin_add, lin_mult, lin_act = flop_linear_polynomial(params, algo="reduced_cmult")

    total_add  += lin_add + 2
    total_mult += lin_mult
    total_act  += lin_act

    total_flop = total_add + total_mult + total_act

    # Convert to real
    total_params = 2 * n_poly

    data = OrderedDict([
        ('total_flop', total_flop),
        ('total_add', total_add),
        ('total_mult', total_mult),
        ('total_canc_max', total_canc),
        ('total_params', total_params),
        ('model_flop', model_flop),
        ('model_add', model_add),
        ('model_mult', model_mult),
        ('model_act', model_act),
        ('model_canc', model_canc),
        ('training_size', params.training_size),
        ('min_power', params.min_power),
        ('max_power', params.max_power),
    ])

    file_path = path + os.sep + "wlmp.csv"

    df = pd.DataFrame(data, columns=data.keys(), index=[0])

    if os.path.exists(file_path):
        df_restored = pd.read_csv(file_path)

        df = df.append(df_restored)
        df = df.sort_values(by=['total_flop'])

    df.to_csv(file_path, index=False)


def main(params):

    np.random.seed(params.seed)

    if params.search_width:
        for max_power in range(params.min_width, params.max_width + 1, params.step_width):
            print("Max Power", max_power)
            params.max_power = max_power
            fit(params)
    else:
        fit(params)


if __name__ == '__main__':

    parser = basic_parser()
    parser.set_defaults(data_scaled=0)

    params, unknown = parser.parse_known_args()

    if unknown:
        raise ValueError("This argument {} in unknown".format(unknown))

    main(params)
