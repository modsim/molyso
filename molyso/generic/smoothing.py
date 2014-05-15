# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy
import scipy.signal


def regular_smooth(x, k):
    """

    :param x:
    :param k:
    :return:
    """
    kl = len(k)
    s = numpy.r_[x[kl - 1:0:-1], x, x[-1:-kl:-1]]
    #print(x.dtype,k.dtype)
    return numpy.convolve(k / k.sum(), s, mode='valid')[kl / 2 - 1:-kl / 2][0:len(x)]


def fft_smooth(x, k):
    """

    :param x:
    :param k:
    :return:
    """
    kl = len(k)
    s = numpy.r_[x[kl - 1:0:-1], x, x[-1:-kl:-1]]
    return scipy.signal.fftconvolve(k / k.sum(), s, mode='valid')[kl / 2 - 1:-kl / 2][0:len(x)]


smooth = regular_smooth

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



