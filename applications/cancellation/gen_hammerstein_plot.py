#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Go through all directories and generate the Hammerstein Violin plots:
"""

import os
from glob import iglob
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv

from helpers.postprocess import gen_new_path


def add_plot_type(path, plot_type):
    """Function to add plot type to the file name
    """

    new_path = path.split(os.sep)
    tmp_file_name = new_path[-1].split("_")
    tmp_file_name.insert(1, plot_type)
    new_path[-1] = "_".join(tmp_file_name)
    new_path = os.sep.join(new_path)

    return new_path


def gen_hammerstein_k_plot(path, kwargs):
    """Functions which generates plots to show the distribution of the weights
    over diff
    """

    results = np.genfromtxt(path, delimiter=',', dtype=np.complex128)

    new_path = gen_new_path(path)

    sns.violinplot(data=np.abs(results), **kwargs).set_title('Magnitude Plot')
    plt.savefig(add_plot_type(new_path, "abs"), bbox_inches="tight")
    plt.close()

    sns.violinplot(data=np.angle(results), **kwargs).set_title('Phase Plot')
    plt.savefig(add_plot_type(new_path, "phase"), bbox_inches="tight")
    plt.close()

    sns.violinplot(data=np.imag(results), **kwargs).set_title('Real Plot')
    plt.savefig(add_plot_type(new_path, "real"), bbox_inches="tight")
    plt.close()

    sns.violinplot(data=np.real(results), **kwargs).set_title('Imaginary Plot')
    plt.savefig(add_plot_type(new_path, "imag"), bbox_inches="tight")
    plt.close()


def gen_hammerstein_weights_plot(path, kwargs):
    """Functions which generates plots to show the distribution of the weights
    over diff.

    We also show the statistics for a single power.
    """

    results = np.load(path)

    new_path = gen_new_path(path)

    # Iterate over powers and generate violin plots for these separately
    for index in range(0, results.shape[0]):
        power = 2 * (index) + 1

        sns.violinplot(data=np.abs(results[index]), **kwargs).set_title('Magnitude Plot Power {}, Mean {}, Std. Dev. {}'.format(power, np.mean(np.abs(results[index])), np.sqrt(np.var(np.abs(results[index])))))
        plt.savefig(add_plot_type(new_path, "abs_" + str(power)), bbox_inches="tight")
        plt.close()

        sns.violinplot(data=np.angle(results[index]), **kwargs).set_title('Phase Plot {}, Mean {}, Std. Dev. {}'.format(power, np.mean(np.angle(results[index])), np.sqrt(np.var(np.angle(results[index])))))
        plt.savefig(add_plot_type(new_path, "phase_" + str(power)), bbox_inches="tight")
        plt.close()

        sns.violinplot(data=np.imag(results[index]), **kwargs).set_title('Real Plot {}, Mean {}, Std. Dev. {}'.format(power, np.mean(np.imag(results[index])), np.sqrt(np.var(np.imag(results[index])))))
        plt.savefig(add_plot_type(new_path, "real_" + str(power)), bbox_inches="tight")
        plt.close()

        sns.violinplot(data=np.real(results[index]), **kwargs).set_title('Imaginary Plot {}, Mean {}, Std. Dev. {}'.format(power, np.mean(np.real(results[index])), np.sqrt(np.var(np.real(results[index])))))
        plt.savefig(add_plot_type(new_path, "imag_" + str(power)), bbox_inches="tight")
        plt.close()


def main():

    # Generate training curves for everything
    kwargs = {"palette" : "muted", "scale" : "count", "inner" : "box", "orient" : "v"}
    # kwargs = {"palette" : "muted", "scale" : "count", "inner" : "stick", "orient" : "v"}

    file_list = [f for f in iglob("results/**/k1*.out", recursive=True) if os.path.isfile(f)]
    for f in file_list:
        print("Processing:", f)
        gen_hammerstein_k_plot(f, kwargs)


if __name__ == '__main__':
    main()
