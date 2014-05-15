# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from .highperformance import relative_maxima, relative_minima

relative_maxima, relative_minima = relative_maxima, relative_minima

import numpy


class NotReallyATree(list):
    def __init__(self, iterable):
        super(NotReallyATree, self).__init__(self)
        self += iterable
        self.na = numpy.array(iterable)

    def query(self, q):  # w_numpy
        distances = numpy.sqrt(numpy.sum(numpy.square(self.na - q), 1))
        pos = numpy.argmin(distances, 0)
        return distances[pos], pos


def vertical_mean(im):
    return numpy.average(im, axis=1)


def horizontal_mean(im):
    return numpy.average(im, axis=0)


def normalize(arr):
    """
    normalizes the values in arr to 0 - 1
    :param arr: input array
    :return: normalized array
    """
    maxi = arr.max()
    mini = arr.min()
    return (arr - mini) / (maxi - mini)


def threshold_outliers(arr, times_std=2.0):
    """
    removes outliers
    :param arr:
    :param times_std:
    :return:
    """
    median = numpy.median(arr)
    std = numpy.std(arr)
    arr[(arr - median) > (times_std * std)] = median + (times_std * std)
    arr[((arr - median) < 0) & (abs(arr - median) > times_std * std)] = median - (times_std * std)
    return arr


def outliers(arr, times_std=2.0):
    return numpy.abs(arr - numpy.median(arr)) > (times_std * numpy.std(arr))


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

    return numpy.array(positions)


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
