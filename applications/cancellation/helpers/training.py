#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions related to either doing a single run of training over multiple seeds
or generating multiple tasks from hyperparameter search points and seeds
and executing these in parallel.
"""
import multiprocessing as mp
import os
import sys
import shutil

from multiprocessing import get_context
import tqdm

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

if tf.__version__[0] != "2":
    tf.enable_eager_execution()

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from gen_models import gen_model

from helpers.postprocess import gen_array, gen_stat
from helpers.preprocess import gen_seeds, initialize, set_seeds
from helpers.summary import save_data_hammerstein, save_data_nn

from sklearn.model_selection import KFold


def single_run(name, model_type, params, model_func, flop_func, train_func):
    """Perform a single run of the model, possibly over multiple seeds and in
    parallel if mp is enabled.

    Parameters
    ----------
    name        : str
        Name of the model (hammerstein, complex_rnn, rnn, complex_ffnn, ffnn).
    model_type  : str
        Type of model (nn, hammerstein).
    params      : :obj:
        Parameters.
    model_func  : :obj:
        Model generator function
    flop_func   : :obj:
        Function to calculate FLOPs for model.
    train_func  : :obj:
        Model training function.
    """

    if model_type not in ["nn", "hammerstein"]:
        raise ValueError("Specified model_type of model is incorrect")

    x_train, y_train, y_train_orig, x_test, y_test, y_test_orig, y_var, h_lin, y_canc_train, y_canc_test, noise, measured_noise_power = initialize(
        params
    )

    canc_lin = 10 * np.log10(np.mean(np.abs(y_test_orig)**2) / np.mean(np.abs(y_test_orig - y_canc_test)**2))

    result_array = []
    seed_list    = gen_seeds(params.seed, params.n_seeds)

    # Tasks shared across seeds
    shared_tasks = (
        params,
        params.batch_size,
        params.learning_rate,
        x_train,
        x_test,
        y_train,
        y_test
    )

    if params.mp:
        tasks = []

        for seed in seed_list:
            tasks.append([*shared_tasks, seed])

        result_array = parallel_train(params, tasks, train_func)

    else:

        for seed in seed_list:
            tmp = train_func(*shared_tasks, seed)
            result_array.append(tmp)

    train_dict = gen_stat(
        gen_array(result_array, "train_loss"),
        gen_array(result_array, "train_var"),
        y_train,
        seed_list
    )

    test_dict = gen_stat(
        gen_array(result_array, "test_loss"),
        gen_array(result_array, "test_var"),
        y_test,
        seed_list
    )

    test_loss_array = gen_array(result_array, "test_loss")
    test_loss_array = test_loss_array[..., -1] # Get only final epoch

    opt_seed_idx = np.nanargmin(test_loss_array, axis = -1)

    model_path   = result_array[opt_seed_idx]["model"]

    model = gen_model(params, name)
    model.load_weights(model_path)

    y_hat = model(x_test)

    if model_type == "nn":
        save_data_nn(name, params, model, model_path, flop_func, train_dict, test_dict, canc_lin, seed_list)
    else:

        K1_array = gen_array(result_array, "K1")
        K2_array = gen_array(result_array, "K2")
        weights_array = gen_array(result_array, "weights")

        save_data_hammerstein(name, params, model, model_path, flop_func, train_dict, test_dict, canc_lin, seed_list, K1_array, K2_array, weights_array)

    # Remove dir for temporary models
    try:
        shutil.rmtree("tmp_models")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def gen_cv_data(params, x_train, y_train):
    """Generate data for cross-validation.

    Parameters
    ----------
    params  : :obj:
        Parameters.
    x_train : :class:`numpy.ndarray`
        Input training data.
    y_train : :class:`numpy.ndarray`
        Output training data.

    Returns
    -------
    x_train_cv : :class:`numpy.ndarray`
        Input training sets for cross-validation.
    y_train_cv : :class:`numpy.ndarray`
        Output training sets for cross-validation.
    x_val_cv : :class:`numpy.ndarray`
        Input validation sets for cross-validation.
    y_val_cv : :class:`numpy.ndarray`
        Output validation sets for cross-validation.
    """

    cv_train_size = (x_train.shape[0] // params.cv_folds) * (params.cv_folds - 1)
    cv_val_size   = (x_train.shape[0] // params.cv_folds)

    x_train_tmp = x_train[: cv_val_size * params.cv_folds]
    y_train_tmp = y_train[: cv_val_size * params.cv_folds]

    kf = KFold(n_splits=params.cv_folds)

    x_train_cv = np.array([x_train_tmp[train_index] for train_index, _ in kf.split(x_train_tmp)])
    x_val_cv   = np.array([x_train_tmp[test_index] for _, test_index in kf.split(x_train_tmp)])

    y_train_cv = np.array([y_train_tmp[train_index] for train_index, _ in kf.split(y_train_tmp)])
    y_val_cv   = np.array([y_train_tmp[test_index] for _, test_index in kf.split(y_train_tmp)])

    return x_train_cv, y_train_cv, x_val_cv, y_val_cv


def parallel_train(params, tasks, train_func):
    """Parallel training, returns the result array for all the parallel
    processes.

    Parameters
    ----------
    params      : :obj:
        Parameters.
    tasks       : :obj:
        Parameters specific to this training run, e.g., used for grid-search.
    train_func  : :obj:
        Model training function.

    Returns
    -------
    output : :class:`numpy.ndarray`
        Array of results from training.
    """

    if params.n_jobs < 1:
        n_processes = mp.cpu_count()
    else:
        n_processes = min(params.n_jobs, mp.cpu_count() - 1)

    pool = mp.Pool(n_processes)

    print("PARALLEL WITH {} PROCESSES".format(n_processes))

    with get_context("spawn").Pool(n_processes) as pool:
        results = list(tqdm.tqdm(pool.imap(train_func, tasks), total=len(tasks)))

    result_array = np.array(results)

    return result_array


def parameter_search_parallel(name, model_type, params, model_func, flop_func, train_func):
    """Perform the parameter search in parallel by constructing all parameter
    points and seeds as a set of tasks.

    Parameters
    ----------
    name        : str
        Name of the model (hammerstein, complex_rnn, rnn, complex_ffnn, ffnn).
    model_type  : str
        Type of model (nn, hammerstein).
    params      : :obj:
        Parameters.
    model_func  : :obj:
        Model generator function
    flop_func   : :obj:
        Function to calculate FLOPs for model.
    train_func  : :obj:
        Model training function.
    """

    if model_type not in ["nn", "hammerstein"]:
        raise ValueError("Specified model_type of model is incorrect")

    x_train, y_train, y_train_orig, x_test, y_test, y_test_orig, y_var, h_lin, y_canc_train, y_canc_test, noise, measured_noise_power = initialize(
        params
    )

    canc_lin = 10 * np.log10(np.mean(np.abs(y_test_orig)**2) / np.mean(np.abs(y_test_orig - y_canc_test)**2))

    # If cross-validation, generate array for the folds
    if params.cv:
        x_train_cv, y_train_cv, x_val_cv, y_val_cv = gen_cv_data(params, x_train, y_train)
        seed_list = gen_seeds(params.seed, params.n_cv_seeds)

        # Swap to use fewer epochs
        n_epochs        = params.n_epochs
        params.n_epochs = params.n_cv_epochs

    else:
        seed_list = gen_seeds(params.seed, params.n_seeds)

    # Generate all the tasks
    tasks              = []
    batch_size_list    = []
    learning_rate_list = []

    if params.cv:
        print("PARAMETER SEARCH USING CV STARTING")
    else:
        print("PARAMETER SEARCH STARTING")

    set_seeds(params.seed)
    batch_size_list = np.random.randint(
        low=params.min_batch_size,
        high=params.max_batch_size,
        size=params.n_search_points
    )

    set_seeds(params.seed)
    learning_rate_list = np.random.uniform(
        low=params.min_learning_rate,
        high=params.max_learning_rate,
        size=params.n_search_points
    )

    for point in range(0, params.n_search_points):
        if params.cv:
            for fold in range(0, params.cv_folds):
                for seed in seed_list:
                    tasks.append(
                        (
                            params,
                            batch_size_list[point],
                            learning_rate_list[point],
                            x_train_cv[fold],
                            x_val_cv[fold],
                            y_train_cv[fold],
                            y_val_cv[fold],
                            seed
                        )
                    )
        else:
            for seed in seed_list:
                tasks.append(
                    (
                        params,
                        batch_size_list[point],
                        learning_rate_list[point],
                        x_train,
                        x_test,
                        y_train,
                        y_test,
                        seed
                    )
                )

    result_array = parallel_train(params, tasks, train_func)

    # Get the mean test loss across multiple seeds to determine best hyperparameter
    test_loss_array = gen_array(result_array, "test_loss")

    if params.cv:
        test_loss_array = np.reshape(
            test_loss_array,
            (params.n_search_points, params.cv_folds, params.n_cv_seeds, params.n_epochs)
        )

        results_last = test_loss_array[..., -1] # All results but only last epoch
        results_last = np.mean(results_last, axis=-1) # Mean across CV seeds for final epochs
        results_last = np.mean(results_last, axis=-1) # Mean across folds for final epochs

        # Find the optimal points in the parameter space based on minimum loss
        opt_point_idx = np.nanargmin(results_last)

        params.batch_size    = batch_size_list[opt_point_idx]
        params.learning_rate = learning_rate_list[opt_point_idx]

        params.n_epochs = n_epochs

        print("CROSS-VALIDATION COMPLETE: PARAMETERS FOUND")
        print("\tBatch-Size", params.batch_size)
        print("\tLearning-Rate", params.learning_rate)
        print()

        if params.search_training_size:

            training_sizes = np.arange(
                params.min_training_size,
                params.max_training_size + params.step_training_size,
                params.step_training_size
            ) / 100

            print("Training set sizes", training_sizes)

            for training_size in training_sizes:
                params.training_size = training_size

                # Rerun with these optimal parameters and other amount of seeds
                single_run(name, model_type, params, model_func, flop_func, train_func)

        else:
            # Rerun with these optimal parameters and other amount of seeds
            single_run(name, model_type, params, model_func, flop_func, train_func)

    else:
        test_loss_array = np.reshape(test_loss_array, (params.n_search_points, params.n_seeds, params.n_epochs))

        results_last_epoch = test_loss_array[..., -1] # All results but only last epoch
        results_avg        = np.mean(results_last_epoch, axis = -1) # Mean across seeds for final epochs for each parameter

        # Find the optimal points in the parameter space based on minimum loss
        opt_point_idx = np.nanargmin(results_avg)
        opt_seed_idx  = np.nanargmin(results_last_epoch[opt_point_idx], axis=-1)

        params.batch_size    = batch_size_list[opt_point_idx]
        params.learning_rate = learning_rate_list[opt_point_idx]

        print("PARAMETER SEARCH COMPLETE: PARAMETERS FOUND")
        print("\tBatch-Size", params.batch_size)
        print("\tLearning-Rate", params.learning_rate)
        print()

        # Get results and parameters for optimal result
        result_array = np.reshape(result_array, (params.n_search_points, params.n_seeds))
        result_array = result_array[opt_point_idx]

        train_dict = gen_stat(
            gen_array(result_array, "train_loss"),
            gen_array(result_array, "train_var"),
            y_train,
            seed_list
        )

        test_dict  = gen_stat(
            gen_array(result_array, "test_loss"),
            gen_array(result_array, "test_var"),
            y_test,
            seed_list
        )

        model_path = result_array[opt_seed_idx]["model"]
        model      = gen_model(params, name)

        model.load_weights(model_path)
        y_hat = model(x_test)

        if model_type == "nn":
            save_data_nn(name, params, model, model_path, flop_func, train_dict, test_dict, canc_lin, seed_list)
        else:
            K1_array = gen_array(result_array, "K1")
            K2_array = gen_array(result_array, "K2")
            weights_array = gen_array(result_array, "weights")

            save_data_hammerstein(name, params, model, model_path, flop_func, train_dict, test_dict, canc_lin, seed_list, K1_array, K2_array, weights_array)

        # Remove dir
        try:
            shutil.rmtree("tmp_models")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
