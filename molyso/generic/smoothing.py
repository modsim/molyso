# -*- coding: utf-8 -*-
"""
smoothing.py contains the main smoothing function, which works by convolving a signal with a smoothing kernel,
a signals function which acts as a cache for kernels, as well as the hamming_smooth function, which is the only
one currently used by external files, providing a simplified interface for smoothing with hamming kernels.
"""
from __future__ import division, unicode_literals, print_function

import numpy


def smooth(signal, kernel):
    """
    Generic smoothing function, smooths by convolving one signal with another.

    :param signal: input signal to be smoothed
    :param kernel: smoothing kernel to be used. will be normalized to :math:`\sum=1`
    :return:
    """

    return numpy.convolve(
        kernel / kernel.sum(),
        numpy.r_[signal[kernel.size - 1:0:-1], signal, signal[-1:-kernel.size:-1]],
        mode='valid')[kernel.size / 2 - 1:-kernel.size / 2][0:len(signal)]


def hamming_smooth(signal, window_width, no_cache=False):
    """
    Smooths a signal by convolving with a hamming window of given width. Caches by the hamming windows by default.

    :param signal: input signal to be smoothed
    :param window_width: window width for the hamming kernel
    :param no_cache: default False, disables caching, e.g. for non-standard window sizes
    :return:
    """
    return smooth(signal,
                  numpy.hamming(window_width) if no_cache
                  else signals(numpy.hamming, window_width))


_signals = {}


def signals(function, parameters):
    """
    Signal cache helper function. Either retrieves or creates and stores a signal which can be created by calling
    the given function with the given parameters.

    :param function: Window function to be called
    :param parameters: Parameters to be passed to the function
    :return: function(*parameters)
    """
    global _signals
    if function not in _signals:
        _signals[function] = {}
    if not type(parameters) == tuple:
        parameters = (parameters,)
    sf = _signals[function]
    if parameters not in sf:
        result = function(*parameters)
        result = result.astype(numpy.float64)
        result.flags.writeable = False
        sf[parameters] = result
        return result
    else:
        return sf[parameters]
