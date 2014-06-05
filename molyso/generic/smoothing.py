# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy
import scipy.signal


def smooth(signal, kernel):
    """

    :param signal:
    :param kernel:
    :return:
    """

    return numpy.convolve(
        kernel / kernel.sum(),
        numpy.r_[signal[kernel.size - 1:0:-1], signal, signal[-1:-kernel.size:-1]],
        mode='valid')[kernel.size / 2 - 1:-kernel.size / 2][0:len(signal)]


def hamming_smooth(signal, window_width):
    return smooth(signal, signals(numpy.hamming, window_width))


_signals = {}


def signals(function, parameters):
    """

    :param function:
    :param parameters:
    :return:
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



