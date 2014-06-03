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


def vertical_mean(im):
    return numpy.average(im, axis=1)


def horizontal_mean(im):
    return numpy.average(im, axis=0)


def relative_maxima(data, order=1):
    value, = scipy.signal.argrelmax(data, order=order)
    return value


def relative_minima(data, order=1):
    value, = scipy.signal.argrelmin(data, order=order)
    return value


def normalize(arr):
    """
    normalizes the values in arr to 0 - 1
    :param arr: input array
    :return: normalized array
    """
    maxi = arr.max()
    mini = arr.min()
    return (arr - mini) / (maxi - mini)


def rescale_and_fit_to_type(img, new_dtype):
    img_min = img.min()
    img_max = img.max()
    scaled_img = ((img.astype(numpy.float32) - img_min) / (img_max - img_min))
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


def threshold_outliers(arr, times_std=2.0):
    """
    removes outliers
    :param arr:
    :param times_std:
    :return:
    """

    arr = arr.copy()
    median = numpy.median(arr)
    std = numpy.std(arr)
    arr[(arr - median) > (times_std * std)] = median + (times_std * std)
    arr[((arr - median) < 0) & (abs(arr - median) > times_std * std)] = median - (times_std * std)
    return arr


def outliers(arr, times_std=2.0):
    return numpy.absolute(arr - numpy.median(arr)) > (times_std * numpy.std(arr))


def remove_outliers(arr, times_std=2.0):
    try:
        return arr[~outliers(arr, times_std)]
    except TypeError:
        return arr


def numerical_differentiation(arr):
    rshift = numpy.zeros_like(arr)
    rshift[1:] = arr[:-1]
    return arr - rshift


def corrected_numerical_differentiation(arr):
    newarr = numerical_differentiation(arr)
    newarr[0] = newarr[1]
    return newarr


def find_insides(arr):
    positions = []

    inside = False
    beg, end = 0, 0

    for n in range(len(arr)):
        if inside:
            if not arr[n]:
                end = n - 1
                positions += [[beg, end]]
                inside = False
        else:
            if arr[n]:
                inside = True
                beg = n
    if inside:
        positions += [[beg, len(arr) - 1]]

    return positions


def find_insides_w_int(arr):
    positions = []

    inside = False
    beg, end = 0, 0
    summing = 0.0

    for n in range(len(arr)):
        if inside:
            if not arr[n]:
                end = n - 1
                positions += [[beg, end, end - beg, summing, summing / (1 + end - beg)]]
                summing = 0.0
                inside = False
            else:
                summing += arr[n]
        else:
            if arr[n]:
                inside = True
                beg = n
                summing += arr[n]
    if inside:
        end = len(arr) - 1
        positions += [[beg, end, end - beg, summing, summing / (1 + end - beg)]]

    return numpy.array(positions)


def make_false_until_false(arr, range_of_interest):
    for n in range_of_interest:
        if arr[n]:
            arr[n] = False
        else:
            return arr


def make_beginning_false(arr):
    return make_false_until_false(arr, range(len(arr)))


def make_ending_false(arr):
    return make_false_until_false(arr, range(len(arr) - 1, -1, -1))


def one_every_n(l, n=1, shift=0):
    signal = numpy.zeros(int(l))
    signal[numpy.around(numpy.arange(shift % n, l - 1, n)).astype(numpy.int32)] = 1
    return signal


def each_image_slice_vertical(img, steps):
    step = img.shape[1] // steps
    for n in range(steps):
        yield n, step, img[:, step * n:step * (n + 1)]


def each_image_slice_horizontal(img, steps):
    step = img.shape[0] // steps
    for n in range(steps):
        yield n, step, img[step * n:step * (n + 1), :]
