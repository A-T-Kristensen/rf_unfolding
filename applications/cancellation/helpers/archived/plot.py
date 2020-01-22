# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# Functions related to post-processing of data for the full-duplex application,
# generating plots and so forth.
# """

# import argparse
# import numpy as np
# import sys
# import os
# import json

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

# import pandas as pd

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
# from complex_nn.complex_misc import calc_odd_powers
# from collections import OrderedDict


# def save_plot(name, path, loss, valid_loss):
#     sns.set_style("darkgrid")

#     plt.plot(loss, '--bo')
#     plt.plot(valid_loss, '--ro')

#     plt.legend(labels = ["Training Loss", "Validation Loss"])

#     plt.title(name)
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.savefig(path + os.sep + name + '.png')
#     plt.close()


# def save_canc_plot(name, path, canc):
#     sns.set_style("darkgrid")

#     plt.plot(canc, '--bo')
#     plt.legend(labels = ["Cancellation"])

#     plt.title(name)
#     plt.xlabel('Epoch')
#     plt.ylabel('Cancellation dBm')
#     plt.savefig(path + os.sep + name + '.png')
#     plt.close()


# def gen_plot(history, n_epochs):
#     # Plot learning curve
#     plt.plot(np.arange(1, len(history.history["loss"]) + 1), -10 * np.log10(history.history["loss"]), "bo-",)
#     plt.plot(np.arange(1, len(history.history["loss"]) + 1), -10 * np.log10(history.history["val_loss"]), "ro-",)

#     plt.ylabel("Self-Interference Cancellation (dB)")
#     plt.xlabel("Training Epoch")
#     plt.legend(["Training Frame", "Test Frame"], loc="lower right")
#     plt.grid(which="major", alpha=0.25)
#     plt.xlim([0, n_epochs + 1])
#     plt.xticks(range(1, n_epochs, 2))
#     plt.savefig("figures/NNconv.pdf", bbox_inches="tight")
#     plt.show()
