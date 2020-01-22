#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Go through all directories and generate the dat files used for:
* Complexity plot: Show the performances in terms of # parameters and # FLOPs.
"""

import os
from glob import iglob
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import csv


def flop_files_split_nn(path_in, path_out):
    """Extract the total number of flops and performance for neural networks.
    """

    df = pd.read_csv(path_in)
    df = df.astype({"ffnn_struct": str})
    df = df.astype({"training_size": float})

    shallow = df[df["ffnn_struct"].str.count("-") == 0]
    shallow = shallow[shallow["training_size"] == 1.0]

    deep = df[df["ffnn_struct"].str.count("-") == 2]
    deep = deep[deep["training_size"] == 1.0]

    x_shallow = shallow["total_flop"].values
    y_shallow = shallow["total_canc_mean"].values

    x_deep = deep["total_flop"].values
    y_deep = deep["total_canc_mean"].values

    directory = path_out.split(os.sep)
    del directory[-1]
    directory = os.sep.join(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path_out + "_shallow.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_shallow, y_shallow))

    with open(path_out + "_deep.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_deep, y_deep))


def flop_files_split_hammerstein(path_in, path_out):
    """Extract the total number of flops and performance for Hammerstein
    neural networks.
    """

    df = pd.read_csv(path_in)
    df = df.astype({"training_size": float})

    rc = df[df["htnn_struct"] == "R-C"]
    rc = rc[rc["training_size"] == 1.0]

    cc = df[df["htnn_struct"] == "C-C"]
    cc = cc[cc["training_size"] == 1.0]

    bc = df[df["htnn_struct"] == "B-C"]
    bc = bc[bc["training_size"] == 1.0]

    x_rc = rc["total_flop"].values
    y_rc = rc["total_canc_mean"].values

    x_cc = cc["total_flop"].values
    y_cc = cc["total_canc_mean"].values

    x_bc = bc["total_flop"].values
    y_bc = bc["total_canc_mean"].values

    directory = path_out.split(os.sep)
    del directory[-1]
    directory = os.sep.join(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path_out + "_rc.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_rc, y_rc))

    with open(path_out + "_cc.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_cc, y_cc))

    with open(path_out + "_bc.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_bc, y_bc))

    x_rc = rc["total_flop_pessimistic"].values
    x_cc = cc["total_flop_pessimistic"].values

    with open(path_out + "_rc_pessimistic.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_rc, y_rc))

    with open(path_out + "_cc_pessimistic.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_cc, y_cc))


def flop_files_split_poly(path_in, path_out):
    """Extract the total number of flops and performance for the polynomial.
    """

    df = pd.read_csv(path_in)
    df = df.astype({"training_size": float})
    df = df[df["training_size"] == 1.0]

    x = df["total_flop"].values
    y = df["total_canc_max"].values

    directory = path_out.split(os.sep)
    del directory[-1]
    directory = os.sep.join(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path_out + ".dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x, y))


def mem_files_split_nn(path_in, path_out):
    """Extract the total number of parameters and performance for neural
    networks.
    """

    df = pd.read_csv(path_in)
    df = df.astype({"ffnn_struct": str})
    df = df.astype({"training_size": float})

    shallow = df[df["ffnn_struct"].str.count("-") == 0]
    shallow = shallow[shallow["training_size"] == 1.0]

    deep = df[df["ffnn_struct"].str.count("-") == 2] # Deep NN has 3 hidden layers
    deep = deep[deep["training_size"] == 1.0]

    x_shallow = shallow["total_params"].values
    y_shallow = shallow["total_canc_mean"].values

    x_deep = deep["total_params"].values
    y_deep = deep["total_canc_mean"].values

    directory = path_out.split(os.sep)
    del directory[-1]
    directory = os.sep.join(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path_out + "_shallow.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_shallow, y_shallow))

    with open(path_out + "_deep.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_deep, y_deep))


def mem_files_split_hammerstein(path_in, path_out):
    """Extract the total number of parameters and performance for Hammerstein
    neural networks.
    """

    df = pd.read_csv(path_in)
    df = df.astype({"training_size": float})

    rc = df[df["htnn_struct"] == "R-C"]
    rc = rc[rc["training_size"] == 1.0]

    cc = df[df["htnn_struct"] == "C-C"]
    cc = cc[cc["training_size"] == 1.0]

    bc = df[df["htnn_struct"] == "B-C"]
    bc = bc[bc["training_size"] == 1.0]

    x_rc = rc["total_params"].values
    y_rc = rc["total_canc_mean"].values

    x_cc = cc["total_params"].values
    y_cc = cc["total_canc_mean"].values

    x_bc = bc["total_params"].values
    y_bc = bc["total_canc_mean"].values

    directory = path_out.split(os.sep)
    del directory[-1]
    directory = os.sep.join(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path_out + "_rc.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_rc, y_rc))

    with open(path_out + "_cc.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_cc, y_cc))

    with open(path_out + "_bc.dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x_bc, y_bc))


def mem_files_split_poly(path_in, path_out):
    """Extract the total number of parameters and performance for the polynomial.
    """

    df = pd.read_csv(path_in)
    df = df.astype({"training_size": float})

    df = df[df["training_size"] == 1.0]

    x = df["total_params"].values
    y = df["total_canc_max"].values

    directory = path_out.split(os.sep)
    del directory[-1]
    directory = os.sep.join(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path_out + ".dat", 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(x, y))


def main():
    """Specify input files and output directories for extracted data.
    """

    flop_files_split_hammerstein("results/results_ftrl/hammerstein_all_test_reduced_cmult.csv", "results/results_ftrl/pgf_dat_complexity/flop_hammerstein_all_complexity_mult_algo_reduced_cmult")
    mem_files_split_hammerstein("results/results_ftrl/hammerstein_all_test_reduced_cmult.csv", "results/results_ftrl/pgf_dat_complexity/mem_hammerstein_all_complexity_mult_algo_reduced_cmult")

    flop_files_split_nn("results/results_adam/complex_ffnn_nl_test_reduced_cmult.csv", "results/results_adam/pgf_dat_complexity/flop_complex_ffnn_nl_complexity_mult_algo_reduced_cmult")
    flop_files_split_nn("results/results_adam/ffnn_nl_test_reduced_cmult.csv", "results/results_adam/pgf_dat_complexity/flop_ffnn_nl_complexity_mult_algo_reduced_cmult")
    flop_files_split_nn("results/results_adam/rnn_simple_nl_test_reduced_cmult.csv", "results/results_adam/pgf_dat_complexity/flop_rnn_simple_nl_complexity_mult_algo_reduced_cmult")

    mem_files_split_nn("results/results_adam/complex_ffnn_nl_test_reduced_cmult.csv", "results/results_adam/pgf_dat_complexity/mem_complex_ffnn_nl_complexity_mult_algo_reduced_cmult")
    mem_files_split_nn("results/results_adam/ffnn_nl_test_reduced_cmult.csv", "results/results_adam/pgf_dat_complexity/mem_ffnn_nl_complexity_mult_algo_reduced_cmult")
    mem_files_split_nn("results/results_adam/rnn_simple_nl_test_reduced_cmult.csv", "results/results_adam/pgf_dat_complexity/mem_rnn_simple_nl_complexity_mult_algo_reduced_cmult")

    flop_files_split_poly("results/results_wlmp/wlmp.csv", "results/results_wlmp/pgf_dat_complexity/flop_polynomial_nl_complexity_mult_algo_reduced_cmult")
    mem_files_split_poly("results/results_wlmp/wlmp.csv", "results/results_wlmp/pgf_dat_complexity/mem_polynomial_nl_complexity_mult_algo_reduced_cmult")


if __name__ == '__main__':
    main()
