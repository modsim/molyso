# -*- coding: utf-8 -*-
"""
smoothing.py contains the main smoothing function, which works by convolving a signal with a smoothing kernel,
a signals function which acts as a cache for kernels, as well as the hamming_smooth function, which is the only
one currently used by external files, providing a simplified interface for smoothing with hamming kernels.
"""
from __future__ import division, unicode_literals, print_function

import numpy as np


def smooth(signal, kernel):
    """
    Generic smoothing function, smooths by convolving one signal with another.

    :param signal: input signal to be smoothed
    :type signal: numpy.ndarray
    :param kernel: smoothing kernel to be used. will be normalized to :math:`\sum=1`
    :type kernel: numpy.ndarray
    :return: The signal convolved with the kernel
    :rtype: numpy.ndarray

    >>> smooth(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]), np.ones(3))
    array([0.        , 0.        , 0.        , 0.        , 0.33333333,
           0.33333333, 0.33333333, 0.        , 0.        ])
    """

    return np.convolve(
        kernel / kernel.sum(),
        np.r_[signal[kernel.size - 1:0:-1], signal, signal[-1:-kernel.size:-1]],
        mode='valid')[kernel.size // 2 - 1:-kernel.size // 2][0:len(signal)]


def hamming_smooth(signal, window_width, no_cache=False):
    """
    Smooths a signal by convolving with a hamming window of given width. Caches by the hamming windows by default.

    :param signal: input signal to be smoothed
    :type signal: numpy.ndarray
    :param window_width: window width for the hamming kernel
    :type window_width: int
    :param no_cache: default `False`, disables caching, *e.g.*, for non-standard window sizes
    :type no_cache: bool
    :return: the smoothed signal
    :rtype: numpy.ndarray

    >>> hamming_smooth(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]), 3)
    array([0.        , 0.        , 0.        , 0.        , 0.06896552,
           0.86206897, 0.06896552, 0.        , 0.        ])
    """

    if len(signal) == 1:
        return signal

    if len(signal) < window_width:
        window_width = len(signal)
        no_cache = True

    return smooth(signal,
                  np.hamming(window_width) if no_cache
                  else signals(np.hamming, window_width))


_signals = {}


# TODO: In a Python3 only version, this could be replaced by decorating calls with a @functools.lru_cache
def signals(function, parameters):
    """
    Signal cache helper function. Either retrieves or creates and stores a signal which can be created by calling
    the given function with the given parameters.

    :param function: Window function to be called
    :type function: callable
    :param parameters: Parameters to be passed to the function
    :type parameters: \*any
    :return: function(\*parameters)
    :rtype: dependent on function

    >>> signals(np.ones, 3)
    array([1., 1., 1.])
    """
    global _signals
    if function not in _signals:
        _signals[function] = {}
    if not type(parameters) == tuple:
        parameters = (parameters,)
    sf = _signals[function]
    if parameters not in sf:
        result = function(*parameters)
        result = result.astype(np.float64)
        result.flags.writeable = False
        sf[parameters] = result
        return result
    else:
        return sf[parameters]
