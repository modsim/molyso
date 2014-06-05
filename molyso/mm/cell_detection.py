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

    def get_neighbor(self, shift=0):
        cl = self.channel.cells.cells_list
        try:
            return cl[cl.index(self) + shift]
        except IndexError:
            return None

    @property
    def inc_neighbor(self):
        return self.get_neighbor(1)

    @property
    def dec_neighbor(self):
        return self.get_neighbor(-1)

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

    # ma, mi = profile.max(), profile.min()

    # TODO make that more meaningful! cells dark, background light
    # empty channel detection
    # noinspection PyArgumentEqualDefault
    thresholded_profile = threshold_outliers(profile, tunable("cells.empty_channel.skipping.outlier_times_sigma", 2.0))

    delta_thresholded_profile = thresholded_profile - thresholded_profile.mean()
    if delta_thresholded_profile[delta_thresholded_profile > 0].sum() < \
            delta_thresholded_profile[delta_thresholded_profile < 0].sum():
        pass

    if ((thresholded_profile.max() - thresholded_profile.min()) / thresholded_profile.max()) < \
            tunable("cells.empty_channel.skipping.intensity_range_quotient", 0.5) and \
            tunable("cells.empty_channel.skipping", False):
        return []

    thresh = threshold_otsu(im)
    bwimg = im > thresh
    bwprof = vertical_mean(bwimg.astype(float))

    profile = simple_baseline_correction(profile, window_width=None)

    profile = hamming_smooth(profile, tunable("cells.smoothing.length", 10))

    extrema = find_extrema_and_prominence(profile, order=tunable("cells.extrema.order", 15))

    newsignal = numpy.zeros_like(profile)
    newsignal[extrema.maxima] = extrema.prominence[extrema.maxima]
    newsignal = newsignal > 0

    # noinspection PyUnresolvedReferences
    newsignal[0] = 1
    # noinspection PyUnresolvedReferences
    newsignal[-1] = 1

    points, = numpy.where(newsignal)

    bnewsignal = newsignal.astype(float)

    last_point = points[0]

    for next_point in points[1:]:
        bnewsignal[last_point:next_point] = numpy.mean(bwprof[last_point:next_point])
        last_point = next_point

    multiplicator = (bnewsignal < tunable("cells.multiplicator.threshold", 0.75))

    cells = multiplicator - newsignal

    cell_i = find_insides(cells)
    # cell_i = [[a, b] for a, b in cell_i if not (((a - b) == 0) or (a == 0) or (b == (len(cells) - 1)))]
    cell_i = [[a, b] for a, b in cell_i if not (((a - b) == 0))]
    cell_i = [[b, e] for b, e in cell_i if bwprof[b:e].mean() < 0.5]

    cell_i = [[b, e] for b, e in cell_i if extrema.prominence[b:e].mean() >
              tunable("cells.filtering.minimum_prominence", 10.0)]

    with DebugPlot() as p:
        p.title("Cell detection")
        p.imshow(numpy.transpose(im), aspect="auto", extent=(0, im.shape[0], 10 * im.shape[1], 0))
        p.imshow(numpy.transpose(bwimg), aspect="auto", extent=(0, im.shape[0], 0, -10 * im.shape[1]))
        p.plot(profile)

        p.plot(thresholded_profile)

        cell_lines = [x for x in cell_i for x in x]

        p.vlines(cell_lines, [im.shape[1] * -10] * len(cell_lines), [im.shape[1] * 10] * len(cell_lines),
                 colors="yellow")

    return cell_i
