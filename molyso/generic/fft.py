# -*- coding: utf-8 -*-
"""
fft.py contains Fourier transform related helper functions
mainly it abstracts the Fourier transform itself
(currently just passing the calls through to the numpy functions)
"""

import numpy as np
from scipy.fftpack import next_fast_len

fft = np.fft.fft
ifft = np.fft.ifft
fftfreq = np.fft.fftfreq


def spectrum_fourier(signal):
    """
    Calls the Fourier transform, and returns only the first half of the transform results

    :param signal: input signal
    :type signal: numpy.array
    :return: Fourier transformed data
    :rtype: numpy.array
    """
    return fft(signal)[:len(signal) // 2]


def spectrum_bins_by_length(len_signal):
    """
    Returns the bins associated with a Fourier transform of a signal of the length len_signal

    :param len_signal: length of the desired bin distribution
    :type len_signal: int
    :return: frequency bins
    :rtype: numpy.array
    """
    freqs = fftfreq(len_signal)[:len_signal // 2]
    freqs[1:] = 1.0 / freqs[1:]
    return freqs


def spectrum_bins(signal):
    """
    Returns the bins associated with a Fourier transform of a signal of the same length of signal

    :param signal: input signal
    :type signal: numpy.array
    :return: frequency bins
    :rtype: numpy.array
    """
    len_arr = len(signal)
    return spectrum_bins_by_length(len_arr)


def spectrum(signal):
    """
    Return a raw spectrum (values are complex). Use :func:`power_spectrum` to directly get real values.

    :param signal: input signal
    :type signal: numpy.array
    :return: frequencies and fourier transformed values
    :rtype: tuple(numpy.array, numpy.array)
    """
    return spectrum_bins(signal), spectrum_fourier(signal)


def power_spectrum(signal):
    """
    Return a power (absolute/real) spectrum (as opposed to the complex spectrum returned by :func:`spectrum` itself)

    :param signal: input signal
    :type signal: numpy.array
    :return: frequencies and fourier transformed values
    :rtype: tuple(numpy.array, numpy.array)
    """
    freqs, fourier = spectrum(signal)
    return freqs, np.absolute(fourier)


def hires_power_spectrum(signal, oversampling=1):
    """
    Return a high resolution power spectrum (compare :func:`power_spectrum`)
    Resolution is enhanced by feeding the FFT a n times larger, zero-padded signal,
    which will yield frequency values of higher precision.

    :param signal: input signal
    :type signal: numpy.array
    :param oversampling: oversampling factor
    :type oversampling: int
    :return: frequencies and fourier transformed values
    :rtype: tuple(numpy.array, numpy.array)
    """
    arr_len = len(signal)
    fast_size = next_fast_len(oversampling * arr_len)

    tmp_data = np.zeros(fast_size)
    tmp_data[:arr_len] = signal

    frequencies, fourier_values = power_spectrum(tmp_data)
    fourier_values[0] = 0

    fourier_values = fourier_values[frequencies < arr_len]
    frequencies = frequencies[frequencies < arr_len]

    return frequencies, fourier_values
