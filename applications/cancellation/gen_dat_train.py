#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Go through all directories and generate the training plot files for latex.
"""

import os
from glob import iglob
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import csv


def gen_dat_train(path, type):
    """Generate the dat files for the training plots.
    """

    results = np.loadtxt(path)

    # Remove extension and separate by os seperator
    new_path = os.path.splitext(path)[0]
    new_path = new_path.split(os.sep)

    # Get folder and remove, will be part of new file namee
    new_file_name = new_path[-3] + "_" + new_path[-2] + "_" + new_path[-1]
    # Join new file name into path and join path
    folder   = new_path[0] + os.sep + "pgf_dat_train"
    new_path = folder + os.sep + new_file_name + ".dat"

    if not os.path.exists(folder):
        os.makedirs(folder)

    # For plotting 1 std dev away
    if len(results.shape) < 2:
        results = np.expand_dims(results, axis=0)

    std  = np.sqrt(np.var(results, axis=0))
    mean = np.mean(results, axis=0)

    # # mean = mean + 37.8599834442139 if np.mean(mean) < 10 else mean

    epoch = np.arange(1, len(std) + 1)

    with open(new_path, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(epoch, mean, mean - std, mean + std))


def main():

    # Generate training data files when all of the training data is used
    file_list = [f for f in iglob("results/results_ftrl/**/**train_cancellation*train_size_1_0*initializer_hammerstein_rayleigh*.out", recursive=True) if os.path.isfile(f)]

    for f in file_list:
        print("Processing:", f)
        gen_dat_train(f, type="db")

    file_list = [f for f in iglob("results/results_adam/complex_ffnn_nl/**/**train_cancellation*train_size_1_0*.out", recursive=True) if os.path.isfile(f)]

    for f in file_list:
        print("Processing:", f)
        gen_dat_train(f, type="db")

    file_list = [f for f in iglob("results/results_adam/ffnn_nl/**/**train_cancellation*train_size_1_0*.out", recursive=True) if os.path.isfile(f)]

    for f in file_list:
        print("Processing:", f)
        gen_dat_train(f, type="db")

    file_list = [f for f in iglob("results/results_adam/rnn_simple_nl/**/**train_cancellation*train_size_1_0*.out", recursive=True) if os.path.isfile(f)]

    for f in file_list:
        print("Processing:", f)
        gen_dat_train(f, type="db")


if __name__ == '__main__':

    main()
