# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy
import scipy.signal


class NotReallyATree(list):
    def __init__(self, iterable):
        super(NotReallyATree, self).__init__(self)
        for i in iterable:
            self.append(i)
        self.na = numpy.array(iterable)

    def query(self, q):  # w_numpy
        distances = numpy.sqrt(numpy.sum(numpy.power(self.na - q, 2.0), 1))
        pos = numpy.argmin(distances, 0)
        return distances[pos], pos


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
    tmp = signal != numpy.roll(signal, shift=1)
    tmp[0], tmp[-1] = signal[0], signal[-1]
    tmp, = numpy.where(tmp)
    if tmp[-1] == signal.size - 1:
        tmp[-1] += 1
    tmp = tmp.reshape((tmp.size//2, 2))
    tmp[:, 1] -= 1
    return tmp


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