# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy

from ..generic.otsu import threshold_otsu
from ..generic.signal import hamming_smooth,  simple_baseline_correction, find_extrema_and_prominence, \
    vertical_mean, threshold_outliers

from .. import DebugPlot, tunable


class Cell(object):
    __slots__ = ['local_top', 'local_bottom', 'channel']

    def __init__(self, top, bottom, channel):
        self.local_top = float(top)
        self.local_bottom = float(bottom)

        self.channel = channel

    @property
    def top(self):
        return self.channel.top + self.local_top

    @property
    def bottom(self):
        return self.channel.top + self.local_bottom

    @property
    def length(self):
        return abs(self.top - self.bottom)

    @property
    def centroid_1d(self):
        return (self.top + self.bottom) / 2.0

    @property
    def centroid(self):
        return [self.channel.centroid[0], self.centroid_1d]

    def __lt__(self, other_cell):
        return self.local_top < other_cell.local_top


class Cells(object):
    """
        docstring
    """

    __slots__ = ['cells_list', 'channel', 'nearest_tree']

    cell_type = Cell

    def __init__(self, channel, bootstrap=True):

        self.cells_list = []

        self.channel = channel

        self.nearest_tree = None

        if not bootstrap:
            return

        for b, e in find_cells_in_channel(self.channel.channel_image):
            if (tunable('cells.minimal_length.in_mu', 1.0) / self.channel.image.calibration_px_to_mu) < e - b:
                self.cells_list.append(self.__class__.cell_type(b, e, self.channel))

    def __len__(self):
        return len(self.cells_list)

    def __iter__(self):
        return iter(self.cells_list)

    def clean(self):
        pass

    @property
    def centroids(self):
        return [cell.centroid for cell in self.cells_list]


def find_cells_in_channel(image):
    # processing is as always mainly performed on the intensity profile
    profile = vertical_mean(image)

    # empty channel detection
    thresholded_profile = threshold_outliers(profile, tunable('cells.empty_channel.skipping.outlier_times_sigma', 2.0))

    # if active, a non-empty channel must have a certain dynamic range min/max
    if ((thresholded_profile.max() - thresholded_profile.min()) / thresholded_profile.max()) < \
            tunable('cells.empty_channel.skipping.intensity_range_quotient', 0.5) and \
            tunable('cells.empty_channel.skipping', False):  # is off by default!
        return []

    # for cell detection, another intensity profile based on an Otsu binarization is used as well
    binary_image = image > threshold_otsu(image) * tunable('cells.otsu_bias', 1.0)
    profile_of_binary_image = vertical_mean(binary_image.astype(float))

    # the profile is first baseline corrected and smoothed ...
    profile = simple_baseline_correction(profile)
    profile = hamming_smooth(profile, tunable('cells.smoothing.length', 10))

    # ... then local extrema are searched
    extrema = find_extrema_and_prominence(profile, order=tunable('cells.extrema.order', 15))

    # based on the following lambda,
    # it will be decided whether a pair of extrema marks a cell or not
    # #1# size must be larger than zero
    # #2# the cell must have a certain 'blackness' (based on the Otsu binarization)
    # #3# the cell must have a certain prominence (difference from background brightness)
    is_a_cell = lambda last_pos, pos: \
        pos - last_pos > 2 and \
        numpy.mean(profile_of_binary_image[last_pos:pos]) < tunable('cells.filtering.maximum_brightness', 0.5) and \
        extrema.prominence[last_pos:pos].mean() > tunable('cells.filtering.minimum_prominence', 10.0) and \
        True

    # possible positions are constructed, and a cell list is generated by checking them with the is_a_cell function
    positions = [pos for pos in extrema.maxima if extrema.prominence[pos] > 0] + [profile.size]
    cells = [[last_pos + 1, pos - 1] for last_pos, pos in zip([0] + positions, positions) if is_a_cell(last_pos, pos)]

    with DebugPlot('cell_detection', 'channel', 'graph') as p:
        p.title("Cell detection")
        p.imshow(numpy.transpose(image), aspect='auto', extent=(0, image.shape[0], 10 * image.shape[1], 0))
        p.imshow(numpy.transpose(binary_image), aspect='auto', extent=(0, image.shape[0], 0, -10 * image.shape[1]))
        p.plot(profile)

        p.plot(thresholded_profile)

        cell_lines = [pos for pos in cells for pos in pos]

        p.vlines(cell_lines,
                 [image.shape[1] * -10] * len(cell_lines),
                 [image.shape[1] * 10] * len(cell_lines),
                 colors='yellow')

    return cells
