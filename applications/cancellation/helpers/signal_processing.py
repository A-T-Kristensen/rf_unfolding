#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for signal processing and plotting.
"""

import numpy as np
from scipy.signal import savgol_filter
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def signal_power(x):
    """Compute the power of a discrete signal.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Input signal.

    Returns
    -------
    output : :class:`numpy.ndarray`
        Power of discrete signal x.
    """
    return 10 * np.log10(np.mean(np.abs(x)**2))


def signal_fft(x, fftpoints=4096, axis=0, norm="ortho"):
    """Compute the discrete Fourier Transform of an input signal and shifts the
    FFT output.

    Parameters
    ----------
    x           : :class:`numpy.ndarray`
        Input signal.
    fftpoints   : :class:`numpy.ndarray`
        Length of the transformed axis of the output.
    axis        : :class:`numpy.ndarray`
        Axis over which to compute the FFT. If not given, the last axis is used.
    norm        : str
        Normalization mode.

    Returns
    -------
    output : :class:`numpy.ndarray`
        FFT of discrete signal x.
    """
    return np.fft.fftshift(np.fft.fft(x, fftpoints, axis=axis, norm=norm))


def signal_filter(x, savgol_window=45, savgol_degree=1):
    """Filter in the frequency domain by applying a Savitzky–Golay filter.

    Parameters
    ----------
    x               : :class:`numpy.ndarray`
        Input signal.
    savgol_window   : int
        The length of the filter window (i.e. the number of coefficients). window_length must be a positive odd integer.
    savgol_degree    : int
        The order of the polynomial used to fit the samples. polyorder must be less than window_length.

    Returns
    -------
    output : :class:`numpy.ndarray`
        The filtered data.
    """
    return 10 * np.log10(savgol_filter(np.power(np.abs(x), 2), savgol_window, savgol_degree))


def cancellation_power(params, y_test, y_canc, y_canc_nonlin, noise, y_var):
    """Calculate power of the cancellation signals.
    """

    noise_power           = signal_power(noise)
    y_test_power          = signal_power(y_test)
    y_test_lin_canc_power = signal_power(y_test - y_canc)

    if params.data_scaled:
        y_test_nonlin_canc_power = signal_power(((y_test - y_canc) / np.sqrt(y_var) - y_canc_nonlin) * np.sqrt(y_var))
    else:
        y_test_nonlin_canc_power = signal_power(y_test - y_canc_nonlin)

    # Return signal powers
    return noise_power, y_test_power, y_test_lin_canc_power, y_test_nonlin_canc_power


def compute_scaling(noise, measured_noise_power, y_test, y_canc, y_canc_nonlin):
    """Computes the scaling based on the noise.
    """

    noise_power  = 10 * np.log10(np.mean(np.abs(noise)**2))
    scaling_const = np.power(10, -(measured_noise_power - noise_power) / 10)

    noise_scaled         = noise / np.sqrt(scaling_const)
    y_test_scaled        = y_test / np.sqrt(scaling_const)
    y_canc_scaled        = y_canc / np.sqrt(scaling_const)
    y_canc_nonlin_scaled = y_canc_nonlin / np.sqrt(scaling_const)

    return y_test_scaled, y_canc_scaled, y_canc_nonlin_scaled, noise_scaled


def plotPSD(params, y_test, y_canc, y_canc_nonlin, noise, y_var, path=None):
    """Plots PSD and computes various signal powers
    """

    # Get self-interference channel length
    chan_len = params.hsi_len

    noise_power, y_test_power, y_test_lin_canc_power, y_test_nonlin_canc_power = cancellation_power(params, y_test, y_canc, y_canc_nonlin, noise, y_var)

    # Calculate spectra
    sampling_freq_MHz = params.sampling_freq_MHz
    scaling_const     = sampling_freq_MHz * 1e6
    fftpoints         = 4096
    freq_axis         = np.linspace(-sampling_freq_MHz / 2, sampling_freq_MHz / 2, fftpoints)
    savgol_window     = 45
    savgol_degree     = 1

    noise_fft           = signal_fft(noise / np.sqrt(scaling_const), fftpoints)
    y_test_fft          = signal_fft(y_test / np.sqrt(scaling_const), fftpoints)
    y_test_lin_canc_fft = signal_fft((y_test - y_canc) / np.sqrt(scaling_const), fftpoints)

    if params.data_scaled:
        y_test_nonlin_canc_fft = signal_fft((((y_test - y_canc) / np.sqrt(y_var) - y_canc_nonlin) * np.sqrt(y_var)) / np.sqrt(scaling_const), fftpoints)
    else:
        y_test_nonlin_canc_fft = signal_fft((y_test - y_canc_nonlin) / np.sqrt(scaling_const), fftpoints)

    # Plot spectra
    to_plot_y_test_fft             = signal_filter(y_test_fft, savgol_window, savgol_degree)
    to_plot_y_test_lin_canc_fft    = signal_filter(y_test_lin_canc_fft, savgol_window, savgol_degree)
    to_plot_y_test_nonlin_canc_fft = signal_filter(y_test_nonlin_canc_fft, savgol_window, savgol_degree)
    to_plot_noise_fft              = signal_filter(noise_fft, savgol_window, savgol_degree)

    # plt.plot(freq_axis, to_plot_y_test_fft, 'b-', freq_axis, to_plot_y_test_lin_canc_fft, 'r-', freq_axis, to_plot_y_test_nonlin_canc_fft, 'm-', freq_axis, to_plot_noise_fft, 'k-')
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Power Spectral Density (dBm/Hz)')
    # plt.title('Polynomial Non-Linear Cancellation')
    # plt.xlim([-sampling_freq_MHz / 2, sampling_freq_MHz / 2])
    # plt.ylim([-170, -90])
    # plt.xticks(range(-int(sampling_freq_MHz / 2), int(sampling_freq_MHz / 2), 2))
    # plt.grid(which='major', alpha=0.25)
    # plt.legend(['Received SI Signal ({:.1f} dBm)'.format(y_test_power), 'After Linear Digital Cancellation ({:.1f} dBm)'.format(y_test_lin_canc_power), 'After Non-Linear Digital Cancellation ({:.1f} dBm)'.format(y_test_nonlin_canc_power), 'Measured Noise Floor ({:.1f} dBm)'.format(noise_power)], loc='upper center')

    # plt.savefig(path + os.sep + 'cancellation.png', bbox_inches='tight')
    # plt.show()

    return noise_power, y_test_power, y_test_lin_canc_power, y_test_nonlin_canc_power
