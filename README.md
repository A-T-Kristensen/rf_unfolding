# Identification of Non-Linear RF Systems Using Backpropagation

This directory contains the source-code used for SI cancellation using model-based NNs and complex-valued neural networks.
The directory contains the following sub-directories:

* [applications](./applications): Contains the source code for specific applications, e.g. full-duplex cancellation.
* [complex_nn](./complex_nn): Main source code for the project, e.g., custom definitions of layers, initializers, and so forth.

The main approach for the complex-valued neural networks is as described here [here](http://deeplearning.net/software/theano_versions/dev/proposals/complex_gradient.html) and [here](https://arxiv.org/abs/1005.5170).

## Setup

To setup a conda environment just run (in this directory)

```
conda env create -f env_rf_unfolding.yml
```

This will install all the required packages, including TensorFlow v2.

### Manual Installation

Alternatively, the following set of commands will provide the necessary packages using Anaconda.
This assumes that conda-forge has been added, otherwise it can be added using the command

```
conda config --append channels conda-forge
```

To setup the environment, call (you may also start by cloning base `conda create --name tf2_cpu --clone base`)

```
conda create -n env_rf_unfolding python=3.7
conda activate env_rf_unfolding
conda install -c anaconda pandas
conda install scipy
conda install -c conda-forge matplotlib
conda install seaborn scikit-learn
conda install -c conda-forge tqdm
conda install -c anaconda tensorflow
```
