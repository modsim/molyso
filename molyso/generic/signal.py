# -*- coding: utf-8 -*-
"""
signal processing helper routines
"""

from __future__ import division, unicode_literals, print_function
from collections import namedtuple
import warnings

import numpy
import scipy.signal
import scipy.interpolate

from .smoothing import hamming_smooth
from .fft import *


def find_phase(signal_1=None, signal_2=None,
               fft_1=None, fft_2=None,
               return_1=False, return_2=False):
    """
    Finds the phase (time shift) between two signals.
    Either signalX or fftX should be set; you can get the FFTs returned
    in order to cache them locally...

    :param signal_1: first input signal
    :param signal_2: second input signal
    :param fft_1: first input fft
    :param fft_2: second input fft
    :param return_1: whether fft1 should be returned
    :param return_2: whether fft2 should be returned
    :return:
    """

    if signal_1 is not None and fft_1 is None:
        fft_1 = fft(signal_1)
    if signal_2 is not None and fft_2 is None:
        fft_2 = fft(signal_2)

    corr = ifft(fft_1 * -fft_2.conjugate())

    corr = numpy.absolute(corr)
    themax = numpy.argmax(corr)
    # if themax > 2 and themax < (len(corr) - 2):
    #    sur = corr[themax-1:themax+2]
    #    themax += -0.5*sur[0] + 0.5*sur[2]

    themax = -themax if themax < len(fft_1) / 2 else len(fft_1) - themax

    result = (themax,)
    if return_1:
        result += (fft_1,)
    if return_2:
        result += (fft_2,)
    return result


ExtremeAndProminence = namedtuple('ExtremeAndProminence', ['maxima', 'minima', 'signal', 'order', 'max_spline',
                                                           'min_spline', 'xpts', 'max_spline_points',
                                                           'min_spline_points', 'prominence'])


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

    if len(maxima) == 0:
        maxima = numpy.array([numpy.argmax(signal)])


    minima = numpy.array([numpy.argmin(signal)])
    iorder = order
    while iorder > 0:
        try:
            minima = relative_minima(signal, order=iorder)
        except ValueError:
            iorder -= 1
            continue
        break
    if len(minima) == 0:
        minima = numpy.array([numpy.argmin(signal)])


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
            arg[:] = float('Inf')
            return arg
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            max_spline = scipy.interpolate.UnivariateSpline(maximaintpx, maximaintpy, bbox=[0, len(signal)], k=k)

    k = 3
    if len(minimaintpy) <= 3:
        k = len(minimaintpy) - 1
    if k < 1:
        def min_spline(arg):
            arg = numpy.zeros_like(arg)
            arg[:] = float('-Inf')
            return arg
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            min_spline = scipy.interpolate.UnivariateSpline(minimaintpx, minimaintpy, bbox=[0, len(signal)], k=k)

    xpts = numpy.linspace(0, len(signal) - 1, len(signal))

    maxsy = max_spline(xpts)
    minsy = min_spline(xpts)

    return ExtremeAndProminence(maxima, minima, signal, order, max_spline, min_spline, xpts, maxsy, minsy,
                                maxsy - minsy)


def simple_baseline_correction(signal, window_width=None):
    """
    performs a simple baseline correction by subtracting a strongly smoothed version of the signal from itself

    :param signal: input signal
    :param window_width: smoothing window width
    :return:
    """
    if window_width is None or window_width > len(signal):
        window_width = len(signal)

    return signal - hamming_smooth(signal, window_width, no_cache=True)


def vertical_mean(image):
    return numpy.average(image, axis=1)


def horizontal_mean(image):
    return numpy.average(image, axis=0)


def relative_maxima(signal, order=1):
    value, = scipy.signal.argrelmax(signal, order=order)
    return value


def relative_minima(signal, order=1):
    value, = scipy.signal.argrelmin(signal, order=order)
    return value


def normalize(data):
    """
    normalizes the values in arr to 0 - 1

    :param data: input array
    :return: normalized array
    """
    maximum = data.max()
    minimum = data.min()
    return (data - minimum) / (maximum - minimum)


def rescale_and_fit_to_type(image, new_dtype):
    img_min = image.min()
    img_max = image.max()
    scaled_img = ((image.astype(numpy.float32) - img_min) / (img_max - img_min))
    to_type = numpy.dtype(new_dtype)
    if to_type.kind == 'f':
        return scaled_img.astype(to_type)
    elif to_type.kind == 'u':
        return (scaled_img * 2 ** (to_type.itemsize * 8)).astype(to_type)
    elif to_type.kind == 'i':
        return (scaled_img * 2 ** (to_type.itemsize * 8 - 1)).astype(to_type)
    elif to_type.kind == 'b':
        return scaled_img > 0.5
    else:
        raise TypeError("Unsupported new_dtype value used for rescale_and_fit_to_type used!")


def threshold_outliers(data, times_std=2.0):
    """
    removes outliers

    :param data:
    :param times_std:
    :return:
    """

    data = data.copy()
    median = numpy.median(data)
    std = numpy.std(data)
    data[(data - median) > (times_std * std)] = median + (times_std * std)
    data[((data - median) < 0) & (abs(data - median) > times_std * std)] = median - (times_std * std)
    return data


def outliers(data, times_std=2.0):
    return numpy.absolute(data - numpy.median(data)) > (times_std * numpy.std(data))


def remove_outliers(data, times_std=2.0):
    try:
        return data[~outliers(data, times_std)]
    except TypeError:
        return data


def find_insides(signal):
    # had a nicer numpy using solution ... which failed in some cases ...
    # plain, but works.
    pairs = []
    last_true = None
    for n, i in enumerate(signal):
        if i and not last_true:
            last_true = n
        elif not i and last_true:
            pairs.append([last_true, n])
            last_true = None

    if last_true:
        pairs.append([last_true, len(signal)-1])

    return numpy.array(pairs)


def one_every_n(length, n=1, shift=0):
    signal = numpy.zeros(int(length))
    signal[numpy.around(numpy.arange(shift % n, length - 1, n)).astype(numpy.int32)] = 1
    return signal


def each_image_slice(image, steps, direction='vertical'):
    if direction == 'vertical':
        step = image.shape[1] // steps
        for n in range(steps):
            yield n, step, image[:, step * n:step * (n + 1)]
    elif direction == 'horizontal':
        step = image.shape[0] // steps
        for n in range(steps):
            yield n, step, image[step * n:step * (n + 1), :]
    else:
        raise ValueError("Unknown direction passed.")