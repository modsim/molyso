# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy
import scipy
import scipy.signal

from ..generic.otsu import threshold_otsu
from ..generic.smoothing import hamming_smooth
from ..generic.util import vertical_mean, threshold_outliers, find_insides
from ..generic.signal import simple_baseline_correction, find_extrema_and_prominence

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
    def centroid1dloc(self):
        return (self.top + self.bottom) / 2.0

    @property
    def centroid(self):
        return [self.channel.centroid[0], self.centroid1dloc]


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
            if (tunable("cells.minimal_length.in_mu", 1.0) / self.channel.image.calibration_px_to_mu) < e - b:
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


def find_cells_in_channel(im):
    profile = vertical_mean(im)

    # empty channel detection
    thresholded_profile = threshold_outliers(profile, tunable("cells.empty_channel.skipping.outlier_times_sigma", 2.0))

    if ((thresholded_profile.max() - thresholded_profile.min()) / thresholded_profile.max()) < \
            tunable("cells.empty_channel.skipping.intensity_range_quotient", 0.5) and \
            tunable("cells.empty_channel.skipping", False):
        return []

    binary_image = im > threshold_otsu(im)
    profile_of_binary_image = vertical_mean(binary_image.astype(float))

    profile = simple_baseline_correction(profile)
    profile = hamming_smooth(profile, tunable("cells.smoothing.length", 10))

    extrema = find_extrema_and_prominence(profile, order=tunable("cells.extrema.order", 15))

    is_a_cell = lambda last_pos, pos: \
        pos - last_pos > 2 and \
        numpy.mean(profile_of_binary_image[last_pos:pos]) < tunable("cells.filtering.maximum_brightness", 0.5) and \
        extrema.prominence[last_pos:pos].mean() > tunable("cells.filtering.minimum_prominence", 10.0) and \
        True

    positions = [pos for pos in extrema.maxima if extrema.prominence[pos] > 0] + [profile.size]
    cells = [[last_pos + 1, pos - 1] for last_pos, pos in zip([0] + positions, positions) if is_a_cell(last_pos, pos)]

    print(cells)

    with DebugPlot() as p:
        p.title("Cell detection")
        p.imshow(numpy.transpose(im), aspect="auto", extent=(0, im.shape[0], 10 * im.shape[1], 0))
        p.imshow(numpy.transpose(binary_image), aspect="auto", extent=(0, im.shape[0], 0, -10 * im.shape[1]))
        p.plot(profile)

        p.plot(thresholded_profile)

        cell_lines = [pos for pos in cells for pos in pos]

        p.vlines(cell_lines,
                 [im.shape[1] * -10] * len(cell_lines),
                 [im.shape[1] * 10] * len(cell_lines),
                 colors="yellow")

    return cells
