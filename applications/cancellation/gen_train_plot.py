#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Go through all directories and generate the training plots.
"""

import os
from glob import iglob
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv


from helpers.postprocess import gen_new_path


def gen_train_plot(path, result_type):
    """Generate the training plots showing performance over epochs.
    """

    results = np.loadtxt(path)
    new_path = gen_new_path(path)

    if len(results.shape) < 2:
        results = np.expand_dims(results, axis=0)

    std  = np.sqrt(np.var(results, axis=0))
    mean = np.mean(results, axis=0)
    x    = np.arange(1, len(results[0]) + 1)

    # Plot learning curve
    plt.plot(x, mean, "r-", linewidth=4)

    plt.fill_between(x, mean - std, mean + std,
                     color='gray', alpha=0.2)

    if result_type == "db":
        plt.ylabel("Self-Interference Cancellation (dB)")
    elif result_type == "loss":
        plt.ylabel("Loss")
    else:
        raise TypeError("Given result_type is not valid {}".format(result_type))

    plt.xlabel("Training Epoch")
    plt.legend(["Mean on Frame", "All Results"], loc="best")
    plt.grid(which="major", alpha=0.25)
    plt.xticks(range(1, results.shape[1], 2))

    axes = plt.gca()

    if result_type == "db":
        # if results[0][0] > 10:
        #     axes.set_ylim([30, 46])
        # else:
        #     axes.set_ylim([0, 9])

        axes.set_ylim([0, 46])

    plt.savefig(new_path, bbox_inches="tight")
    plt.close()


def main():

    file_list = [f for f in iglob("results/**/**cancellation*train_size_1_0*.out", recursive=True) if os.path.isfile(f)]
    for f in file_list:
        print("Processing:", f)
        gen_train_plot(f, result_type="db")

    # file_list = [f for f in iglob("results/results_adam*/**/**cancellation*train_size_1_0*.out", recursive=True) if os.path.isfile(f)]
    # for f in file_list:
    #     print("Processing:", f)
    #     gen_train_plot(f, result_type="db")


if __name__ == '__main__':

    main()
