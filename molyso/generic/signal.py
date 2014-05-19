# -*- coding: utf-8 -*-
"""
signal processing helper routines
"""

from __future__ import division, unicode_literals, print_function
from collections import namedtuple

import numpy
import scipy.signal
import scipy.interpolate

import warnings

from .fft import fft, ifft, fftfreq, get_optimal_dft_size
from .util import relative_maxima, relative_minima
from .smoothing import smooth


def find_phase(signal1=None, signal2=None,
               fft1=None, fft2=None,
               return1=False, return2=False):
    """
    Finds the phase (time shift) between two signals.
    Either signalX or fftX should be set; you can get the FFTs returned
    in order to cache them locally...

    :param signal1: first input signal
    :param signal2: second input signal
    :param fft1: first input fft
    :param fft2: second input fft
    :param return1: whether fft1 should be returned
    :param return2: whether fft2 should be returned
    :return:
    """

    if signal1 is not None and fft1 is None:
        fft1 = fft(signal1)
    if signal2 is not None and fft2 is None:
        fft2 = fft(signal2)

    corr = ifft(fft1 * -fft2.conjugate())

    corr = numpy.absolute(corr)
    themax = numpy.argmax(corr)
    #if themax > 2 and themax < (len(corr) - 2):
    #    sur = corr[themax-1:themax+2]
    #    themax += -0.5*sur[0] + 0.5*sur[2]

    themax = -themax if themax < len(fft1) / 2 else len(fft1) - themax

    result = (themax,)
    if return1:
        result += (fft1,)
    if return2:
        result += (fft2,)
    return result


ExtremeAndProminence = namedtuple("ExtremeAndProminence", ["maxima", "minima", "signal", "order", "max_spline",
                                                           "min_spline", "xpts", "max_spline_points",
                                                           "min_spline_points", "prominence"])


def find_extrema_and_prominence(signal, order=5):
    # we are FORCING some kind of result here, although it might be meaningless

    maxima = numpy.array([numpy.argmax(signal)])
    iorder = order
    while iorder > 0:
        try:
            maxima = relative_maxima(signal, order=iorder)
        except ValueError:
            iorder -= 1
            continue
        break

    minima = numpy.array([numpy.argmin(signal)])
    iorder = order
    while iorder > 0:
        try:
            minima = relative_minima(signal, order=iorder)
        except ValueError:
            iorder -= 1
            continue
        break

    maximaintpx = numpy.zeros(len(maxima) + 2)
    maximaintpy = numpy.copy(maximaintpx)

    minimaintpx = numpy.zeros(len(minima) + 2)
    minimaintpy = numpy.copy(minimaintpx)

    maximaintpx[0] = 0
    maximaintpx[1:-1] = maxima[:]
    maximaintpx[-1] = len(signal) - 1

    maximaintpy[0] = signal[maxima][0]
    maximaintpy[1:-1] = signal[maxima][:]
    maximaintpy[-1] = signal[maxima][-1]

    minimaintpx[0] = 0
    minimaintpx[1:-1] = minima[:]
    minimaintpx[-1] = len(signal) - 1

    minimaintpy[0] = signal[minima][0]
    minimaintpy[1:-1] = signal[minima][:]
    minimaintpy[-1] = signal[minima][-1]

    k = 3
    if len(maximaintpy) <= 3:
        k = len(maximaintpy) - 1
    if k < 1:
        def max_spline(arg):
            arg = numpy.zeros_like(arg)
            arg[:] = float("Inf")
            return arg
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            max_spline = scipy.interpolate.UnivariateSpline(maximaintpx, maximaintpy, bbox=[0, len(signal)], k=k)

    k = 3
    if len(minimaintpy) <= 3:
        k = len(minimaintpy) - 1
    if k < 1:
        def min_spline(arg):
            arg = numpy.zeros_like(arg)
            arg[:] = float("-Inf")
            return arg
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            min_spline = scipy.interpolate.UnivariateSpline(minimaintpx, minimaintpy, bbox=[0, len(signal)], k=k)

    xpts = numpy.linspace(0, len(signal) - 1, len(signal))

    maxsy = max_spline(xpts)
    minsy = min_spline(xpts)

    return ExtremeAndProminence(maxima, minima, signal, order, max_spline, min_spline, xpts, maxsy, minsy,
                                maxsy - minsy)


def simple_baseline_correction(signal, window_width=100):
    """
    performs a simple baseline correction by subtracting a strongly smoothed version of the signal from itself
    :param signal: input signal
    :param window_width: smoothing window width
    :return:
    """
    if window_width is None:
        window_width = len(signal)
    elif window_width > len(signal):
        window_width = len(signal)

    smoothing_signal = numpy.hamming(window_width)
    return signal - smooth(smooth(signal, smoothing_signal), smoothing_signal)


def _spectrum(arr):
    """
    spectrum function using alternate construction of associated frequencies
    :param arr: input signal
    :return:
    """
    ft = fft(arr)
    ft = ft[:len(ft) // 2]
    # I say frequency but I mean 'wavelength'
    freq = 1.0 / (
        numpy.array([1] + range(1, len(ft))).astype(numpy.float)
        / float(len(ft)) * 0.5)
    return freq, ft


def _spec_fft(arr):
    return fft(arr)[:len(arr) // 2]


def _spec_bins_n(len_arr):
    freqs = fftfreq(len_arr)[:len_arr // 2]
    freqs[1:] = 1.0 / freqs[1:]
    return freqs


def _spec_bins(arr):
    len_arr = len(arr)
    return _spec_bins_n(len_arr)


def spectrum(arr):
    """
    returns the spectrum (tuple of frequencies and their occurrence)
    :param arr: input signal
    :return:
    """
    return _spec_bins(arr), _spec_fft(arr)


def powerspectrum(arr):
    """
    return a power (absolute/real) spectrum (as opposed to the complex spectrum returned by spectrum itself
    :param arr:
    :return:
    """
    freqs, fourier = spectrum(arr)
    return freqs, numpy.absolute(fourier)


def hires_powerspectrum(arr, oversampling=1):
    arr_len = len(arr)
    xsize = get_optimal_dft_size(oversampling * arr_len)

    tmp_data = numpy.zeros(xsize)
    tmp_data[:arr_len] = arr

    frequencies, fourier_values = powerspectrum(tmp_data)
    fourier_values[0] = 0

    fourier_values = fourier_values[frequencies < arr_len]
    frequencies = frequencies[frequencies < arr_len]

    return frequencies, fourier_values


def oversample(arr, times=2):
    """
    oversamples the signal times times, without interpolating
    :param arr:
    :param times:
    :return:
    """

    if times == 1:
        return arr

    newarr = numpy.zeros(len(arr) * times, dtype=arr.dtype)
    for num in range(0, times):
        newarr[num::times] = arr
    return newarr
