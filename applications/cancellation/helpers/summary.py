#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions related to saving results.
"""
import numpy as np
import sys
import os
import json
import glob
import shutil

import copy
import pandas as pd

from helpers.flop import flop_linear_polynomial
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")


def save_data_nn(
    name,
    params,
    model,
    model_path,
    flop_func,
    train_dict,
    test_dict,
    canc_lin,
    seed_list
):
    """Function which saves all the data for neural networks by dumping the
    performance per seed per epoch as .out files, saving the parameters used
    for the training and or grid-searching and collecting main results in
    .csv files.

    Parameters
    ----------
    name        : str
        Name of the model (hammerstein, complex_rnn, rnn, complex_ffnn, ffnn).
    params      : :obj:
        Parameters.
    model       : :obj:
        The Tensorflow model.
    model_path  : str
        Path to tempory storage of model.
    flop_func   : :obj:
        Function to calculate FLOPs for model.
    train_dict  : :obj:
        Dictionary of training performance.
    test_dict   : :obj:
        Dictionary of test performance.
    canc_lin    : float
        Linear cancellation in dB.
    seed_list   : :obj:
        List of seeds.
    """

    output_dir = "results" + os.sep

    if params.save:
        output_dir += "results" + "_" + params.optimizer
    else:
        output_dir += "tmp" + "_" + params.optimizer

    if params.momentum != 0.0:
        output_dir += "_" + "momentum" + "_" + str(params.momentum).replace(".", "_")

    if params.exp_name is not None:
        output_dir += "_" + params.exp_name

    prefix = output_dir \
        + os.sep + name \
        + "_" + params.fit_option \
        + os.sep + "struct" + "_" + params.ffnn_struct

    postfix = "_" + "n_epochs" + "_" + str(params.n_epochs) + "_" \
              + "train_size" + "_" + str(params.training_size).replace(".", "_") + "_" \
              + "batch_size" + "_" + str(params.batch_size) + "_" \
              + "lr" + "_" + str(params.learning_rate).replace(".", "_") + "_" \
              + "initializer" + "_" + str(params.initializer)

    if params.gradient_clipping != 0.0:
        postfix += "_gradclip"

    if params.regularizer is not None:
        postfix += "_" + params.regularizer + "_" + str(params.weight_decay).replace(".", "_")

    # Generate the .out files
    save_results(params, model_path, train_dict, test_dict, seed_list, prefix, postfix)

    # Generate CSV files
    common_str = output_dir + os.sep + name + "_" + params.fit_option + "_"

    path = common_str + "train" + "_" + "default" + ".csv"
    save_csv_nn(path, params, model, flop_func, train_dict, canc_lin, "default")

    path = common_str + "test" + "_" + "default" + ".csv"
    save_csv_nn(path, params, model, flop_func, test_dict, canc_lin, "default")

    path = common_str + "train" + "_" + "reduced_cmult" + ".csv"
    save_csv_nn(path, params, model, flop_func, train_dict, canc_lin, "reduced_cmult")

    path = common_str + "test" + "_" + "reduced_cmult" + ".csv"
    save_csv_nn(path, params, model, flop_func, test_dict, canc_lin, "reduced_cmult")


def save_data_hammerstein(
    name,
    params,
    model,
    model_path,
    flop_func,
    train_dict,
    test_dict,
    canc_lin,
    seed_list,
    K1_array,
    K2_array,
    weights_array
):
    """Function which saves all the data for Hammerstein networks by dumping the
    performance per seed per epoch as .out files, saving the parameters used
    for the training and or grid-searching and collecting main results in
    .csv files.

    Parameters
    ----------
    name        : str
        Name of the model (hammerstein, complex_rnn, rnn, complex_ffnn, ffnn).
    params      : :obj:
        Parameters.
    model       : :obj:
        The Tensorflow model.
    model_path  : str
        Path to tempory storage of model.
    flop_func   : :obj:
        Function to calculate FLOPs for model.
    train_dict  : :obj:
        Dictionary of training performance.
    test_dict   : :obj:
        Dictionary of test performance.
    canc_lin    : float
        Linear cancellation in dB.
    seed_list   : :obj:
        List of seeds.
    """

    output_dir = "results" + os.sep

    if params.save:
        output_dir += "results" + "_" + params.optimizer
    else:
        output_dir += "tmp" + "_" + params.optimizer

    if params.momentum != 0.0:
        output_dir += "_" + "momentum" + "_" + str(params.momentum).replace(".", "_")

    if params.exp_name is not None:
        output_dir += "_" + params.exp_name

    prefix = output_dir \
        + os.sep + name \
        + "_" + params.fit_option \
        + os.sep + "struct" + "_" + params.htnn_struct + "_min_power_" + str(params.min_power) + "_max_power_" + str(params.max_power)

    postfix = "_" + "n_epochs" + "_" + str(params.n_epochs) + "_" \
              + "train_size" + "_" + str(params.training_size).replace(".", "_") + "_" \
              + "batch_size" + "_" + str(params.batch_size) + "_" \
              + "lr" + "_" + str(params.learning_rate).replace(".", "_") + "_" \
              + "initializer" + "_" + str(params.initializer)

    if params.gradient_clipping_param1 != 0.0 or params.gradient_clipping_param2 != 0.0 or params.gradient_clipping != 0.0:
        postfix += "_gradclip"

    if params.regularizer is not None:
        postfix += "_" + params.regularizer + "_" + str(params.weight_decay).replace(".", "_")

    # Generate the .out files
    save_results(params, model_path, train_dict, test_dict, seed_list, prefix, postfix)

    # Generate CSV files for train and test sets and different algorithms
    common_str = output_dir + os.sep + name + "_" + params.fit_option + "_"

    path = common_str + "train" + "_" + "default" + ".csv"
    save_csv_hammerstein(path, params, model, flop_func, train_dict, canc_lin, "default")

    path = common_str + "test" + "_" + "default" + ".csv"
    save_csv_hammerstein(path, params, model, flop_func, test_dict, canc_lin, "default")

    path = common_str + "train" + "_" + "reduced_cmult" + ".csv"
    save_csv_hammerstein(path, params, model, flop_func, train_dict, canc_lin, "reduced_cmult")

    path = common_str + "test" + "_" + "reduced_cmult" + ".csv"
    save_csv_hammerstein(path, params, model, flop_func, test_dict, canc_lin, "reduced_cmult")

    save_weights_hammerstein(params, prefix, postfix, K1_array, K2_array, weights_array)


def convert_np_to_int(input):
    """As we mainly use numpy for calculations, we may occasionally have
    numpy datatypes, check for this before saving to JSON.
    """

    if 'float' in str(type(input)):
        return float(input)
    elif 'int' in str(type(input)):
        return int(input)
    elif 'complex' in str(type(input)):
        return str(input)
    else:
        raise TypeError


def save_results(params, model_path, train_dict, test_dict, seed_list, prefix, postfix):
    """Save the results per seed over all epochs, such as loss and cancellation.

    Parameters
    ----------
    params      : :obj:
        Parameters.
    model_path  : str
        Path to tempory storage of model.
    train_dict  : :obj:
        Dictionary of training performance.
    test_dict   : :obj:
        Dictionary of test performance.
    seed_list   : :obj:
        List of seeds.
    prefix      : str
        Prefix for output files.
    postfix     : str
        Postfix for output files.
    """

    path       = prefix + os.sep
    path_train = path + "train" + "_"
    path_test  = path + "test" + "_"

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Dump parameters
    with open(path + 'params' + postfix + '.json', 'w') as f:
        json.dump(params.__dict__, f, default=convert_np_to_int)

    # Copy model parameters from tmp
    files = glob.glob(model_path + '*')
    for file in files:
        file_ext = os.path.splitext(file)[-1]
        shutil.move(file, path + "model" + postfix + file_ext)

    # Dump seeds
    with open(path + "seed" + postfix + ".out", 'w') as f:
        np.savetxt(f, seed_list, fmt="%d")

    # Dump cancellation and loss performance
    with open(path_train + "cancellation" + postfix + ".out", 'w') as f:
        np.savetxt(f, train_dict["canc_array"], fmt='%1.6f')

    with open(path_test + "cancellation" + postfix + ".out", 'w') as f:
        np.savetxt(f, test_dict["canc_array"], fmt='%1.6f')

    with open(path_train + "loss" + postfix + ".out", 'w') as f:
        np.savetxt(f, train_dict["loss_array"], fmt='%1.5e')

    with open(path_test + "loss" + postfix + ".out", 'w') as f:
        np.savetxt(f, test_dict["loss_array"], fmt='%1.5e')

    # Prepare to save the rest of the data by removing the arrays
    tmp_dict = copy.deepcopy(train_dict)
    del tmp_dict["loss_array"]
    del tmp_dict["var_array"]
    del tmp_dict["canc_array"]

    with open(path + 'train_dict' + postfix + '.json', 'w') as f:
        json.dump(tmp_dict, f, default=convert_np_to_int)

    tmp_dict = copy.deepcopy(test_dict)
    del tmp_dict["loss_array"]
    del tmp_dict["var_array"]
    del tmp_dict["canc_array"]

    with open(path + 'test_dict' + postfix + '.json', 'w') as f:
        json.dump(tmp_dict, f, default=convert_np_to_int)


def save_csv_nn(
    path,
    params,
    model,
    flop_func,
    result_dict,
    canc_lin,
    algo="default",
):
    """Save summary for Neural Networks.

    Parameters
    ----------
    path        : str
        Path to store file.
    params      : :obj:
        Parameters.
    model       : :obj:
        The Tensorflow model.
    flop_func   : :obj:
        Function to calculate FLOPs for model.
    result_dict : :obj:
        Result dictionary for .csv file.
    canc_lin    : float
        Linear cancellation in dB.
    algo        : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".
    """

    model_add, model_mult, model_act = flop_func(model, algo=algo)

    total_add  = model_add
    total_mult = model_mult
    total_act  = model_act

    # Add cost of linear
    if params.fit_option == "nl":
        lin_add, lin_mult, lin_act = flop_linear_polynomial(params, algo=algo)

        total_add  += lin_add + 2
        total_mult += lin_mult
        total_act  += lin_act

    total_flop = total_add + total_mult + total_act
    model_flop = model_add + model_mult + model_act

    total_canc_max      = result_dict["max_canc"]
    total_canc_mean     = result_dict["mean_canc"]
    total_canc_mean_lin = result_dict["mean_canc_lin"]

    if params.fit_option == "nl":
        total_canc_max      += canc_lin
        total_canc_mean     += canc_lin
        total_canc_mean_lin += canc_lin

    model_params = model.count_params()
    total_params = model_params

    # Add number of parameters for the linear model
    if params.fit_option == "nl":
        total_params += 2 * params.hsi_len

    data = OrderedDict([
        ('total_flop', total_flop),
        ('total_add', total_add),
        ('total_mult', total_mult),
        ('total_act', total_act),
        ('total_canc_max', total_canc_max),
        ('total_canc_mean', total_canc_mean),
        ('total_canc_var_mean', result_dict["var_mean_canc"]),
        ('total_canc_mean_lin', total_canc_mean_lin),
        ('total_params', total_params),
        ('model_flop', model_flop),
        ('model_add', model_add),
        ('model_mult', model_mult),
        ('model_act', model_act),
        ('model_canc_max', result_dict["max_canc"]),
        ('model_canc_mean', result_dict["mean_canc"]),
        ('model_canc_var_mean', result_dict["var_mean_canc"]),
        ('model_canc_mean_lin', result_dict["mean_canc_lin"]),
        ('model_loss_min', result_dict["min_loss"]),
        ('model_loss', result_dict["mean_loss"]),
        ('model_loss_var_mean', result_dict["var_mean_loss"]),
        ('model_loss_var', result_dict["var_loss"]),
        ('model_params', model_params),
        ('ffnn_struct', params.ffnn_struct),
        ('shuffle', params.shuffle),
        ('training_size', params.training_size),
        ('optimizer', params.optimizer),
        ('momentum', params.momentum),
        ('n_epochs', params.n_epochs),
        ('batch_size', params.batch_size),
        ('lr', params.learning_rate),
        ('gradient_clipping', params.gradient_clipping),
        ('regularizer', params.regularizer),
        ('weight_decay', params.weight_decay),
        ('seed', params.seed),
        ('n_seeds', params.n_seeds),
        ('search', params.search),
        ('cv', params.cv),
        ('cv_folds', params.cv_folds),
        ('n_cv_seeds', params.n_cv_seeds),
        ('n_cv_epochs', params.n_cv_epochs),
        ('opt_seed', result_dict["opt_seed"]),
        ('n_search_points', params.n_search_points),
        ('hsi_len', params.hsi_len),
        ('min_batch_size', params.min_batch_size),
        ('max_batch_size', params.max_batch_size),
        ('min_learning_rate', params.min_learning_rate),
        ('max_learning_rate', params.max_learning_rate),
    ])

    df = pd.DataFrame(data, columns=data.keys(), index=[0])

    if os.path.exists(path):
        df_restored = pd.read_csv(path)

        df = df.append(df_restored)
        df = df.sort_values(by=['total_flop'])

    df.to_csv(path, index=False)


def save_csv_hammerstein(
    path,
    params,
    model,
    flop_func,
    result_dict,
    canc_lin,
    algo="default",
):
    """Summary for Hammerstein neural network in .csv file.

    Parameters
    ----------
    path        : str
        Path to store file.
    params      : :obj:
        Parameters.
    model       : :obj:
        The Tensorflow model.
    flop_func   : :obj:
        Function to calculate FLOPs for model.
    result_dict : :obj:
        Result dictionary for .csv file.
    canc_lin    : float
        Linear cancellation in dB.
    algo        : str
        The algorithm to compute the number of real-valued operations.
        One of "default" or "reduced_cmult".
    """

    model_add, model_mult, model_act = flop_func(params, algo=algo)

    total_add  = model_add
    total_mult = model_mult
    total_act  = model_act

    model_add_pessimistic, model_mult_pessimistic, model_act_pessimistic = flop_func(params, calculation="pessimistic", algo=algo)

    total_add_pessimistic  = model_add_pessimistic
    total_mult_pessimistic = model_mult_pessimistic
    total_act_pessimistic  = model_act_pessimistic

    # Add cost of linear
    if params.fit_option == "nl":
        lin_add, lin_mult, lin_act = flop_linear_polynomial(params, algo=algo)

        total_add  += lin_add + 2
        total_mult += lin_mult
        total_act  += lin_act

        total_add_pessimistic  += lin_add
        total_mult_pessimistic += lin_mult
        total_act_pessimistic  += lin_act

    total_flop = total_add + total_mult + total_act
    model_flop = model_add + model_mult + model_act

    total_flop_pessimistic = total_add_pessimistic + total_mult_pessimistic + total_act_pessimistic
    model_flop_pessimistic = model_add_pessimistic + model_mult_pessimistic + model_act_pessimistic

    total_canc_max      = result_dict["max_canc"]
    total_canc_mean     = result_dict["mean_canc"]
    total_canc_mean_lin = result_dict["mean_canc_lin"]

    if params.fit_option == "nl":
        total_canc_max      += canc_lin
        total_canc_mean     += canc_lin
        total_canc_mean_lin += canc_lin

    model_params = model.count_params()
    total_params = model_params

    # Add number of parameters for the linear model
    if params.fit_option == "nl":
        total_params += 2 * params.hsi_len

    data = OrderedDict([
        ('total_flop', total_flop),
        ('total_add', total_add),
        ('total_mult', total_mult),
        ('total_act', total_act),
        ('total_flop_pessimistic', total_flop_pessimistic),
        ('total_add_pessimistic', total_add_pessimistic),
        ('total_mult_pessimistic', total_mult_pessimistic),
        ('total_act_pessimistic', total_act_pessimistic),
        ('total_canc_max', total_canc_max),
        ('total_canc_mean', total_canc_mean),
        ('total_canc_var_mean', result_dict["var_mean_canc"]),
        ('total_canc_mean_lin', total_canc_mean_lin),
        ('total_params', total_params),
        ('model_flop', model_flop),
        ('model_add', model_add),
        ('model_mult', model_mult),
        ('model_act', model_act),
        ('model_flop_pessimistic', model_flop_pessimistic),
        ('model_add_pessimistic', model_add_pessimistic),
        ('model_mult_pessimistic', model_mult_pessimistic),
        ('model_act_pessimistic', model_act_pessimistic),
        ('model_canc_max', result_dict["max_canc"]),
        ('model_canc_mean', result_dict["mean_canc"]),
        ('model_canc_var_mean', result_dict["var_mean_canc"]),
        ('model_canc_mean_lin', result_dict["mean_canc_lin"]),
        ('model_loss_min', result_dict["min_loss"]),
        ('model_loss', result_dict["mean_loss"]),
        ('model_loss_var_mean', result_dict["var_mean_loss"]),
        ('model_loss_var', result_dict["var_loss"]),
        ('model_params', model_params),
        ('htnn_struct', params.htnn_struct),
        ('min_power', params.min_power),
        ('max_power', params.max_power),
        ('shuffle', params.shuffle),
        ('training_size', params.training_size),
        ('optimizer', params.optimizer),
        ('momentum', params.momentum),
        ('initializer', params.initializer),
        ('n_epochs', params.n_epochs),
        ('batch_size', params.batch_size),
        ('lr', params.learning_rate),
        ('gradient_clipping_param1', params.gradient_clipping_param1),
        ('gradient_clipping_param2', params.gradient_clipping_param2),
        ('gradient_clipping', params.gradient_clipping),
        ('regularizer', params.regularizer),
        ('weight_decay', params.weight_decay),
        ('seed', params.seed),
        ('n_seeds', params.n_seeds),
        ('search', params.search),
        ('cv', params.cv),
        ('cv_folds', params.cv_folds),
        ('n_cv_seeds', params.n_cv_seeds),
        ('n_cv_epochs', params.n_cv_epochs),
        ('opt_seed', result_dict["opt_seed"]),
        ('n_search_points', params.n_search_points),
        ('hsi_len', params.hsi_len),
        ('min_batch_size', params.min_batch_size),
        ('max_batch_size', params.max_batch_size),
        ('min_learning_rate', params.min_learning_rate),
        ('max_learning_rate', params.max_learning_rate),
    ])

    df = pd.DataFrame(data, columns=data.keys(), index=[0])

    if os.path.exists(path):
        df_restored = pd.read_csv(path)

        df = df.append(df_restored)
        df = df.sort_values(by=['total_flop'])

    df.to_csv(path, index=False)


def save_weights_hammerstein(
    params,
    prefix,
    postfix,
    K1_array,
    K2_array,
    weights_array
):
    """Function to save the weights of the Hammerstein model

    Parameters
    ----------
    params          : :obj:
        Parameters.
    prefix          : str
        Prefix for output files.
    postfix         : str
        Postfix for output files.
    K1_array        : :class:`numpy.ndarray`
        Array of K1 values for different seeds.
    K2_array        : :class:`numpy.ndarray`
        Array of K2 values for different seeds.
    weights_array   : :class:`numpy.ndarray`
        Array of weights in the Hammerstein layer.
    """

    weights_array = np.reshape(weights_array, (params.n_seeds, -1, params.hsi_len))
    weights_array = np.swapaxes(weights_array, 0, 1)

    K1_array = np.squeeze(K1_array, axis = -1)
    K2_array = np.squeeze(K2_array, axis = -1)

    path = prefix + os.sep

    path_weights = path + "weights"
    path_K1      = path + "k1"
    path_K2      = path + "k2"

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    with open(path_weights + postfix + ".out", 'wb') as f:
        np.save(f, weights_array, allow_pickle=True)

    with open(path_K1 + postfix + ".out", 'w') as f:
        np.savetxt(f, K1_array, delimiter=",")

    with open(path_K2 + postfix + ".out", 'w') as f:
        np.savetxt(f, K2_array, delimiter=",")
