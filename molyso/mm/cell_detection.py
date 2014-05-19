# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy
import scipy
import scipy.signal

from ..generic.otsu import threshold_otsu
from ..generic.smoothing import smooth, signals
from ..generic.util import vertical_mean, threshold_outliers, find_insides, corrected_numerical_differentiation
from ..generic.signal import simple_baseline_correction, find_extrema_and_prominence

from .. import DebugPlot, tunable


class Cell(object):
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
        return (self.top + self.bottom) / 2

    @property
    def centroid(self):
        return [self.channel.centroid[0], self.centroid1dloc]

    def get_neighbor(self, shift=0):
        c = self.channel.cells
        pos = c.index(self)
        try:
            return c[pos + shift]
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


class Cells(list):
    """
        docstring
    """

    cell_type = Cell

    def __init__(self, image, channel, imagedata, bootstrap=True):
        super(Cells, self).__init__(self)

        self.image = image
        self.channel = channel

        self.imagedata = imagedata

        self.nearest_tree = None

        if not bootstrap:
            return

        for b, e in find_cells_in_channel(self.imagedata):
            self.append(self.__class__.cell_type(b, e, self.channel))

    def clean(self):
        del self.imagedata

    @property
    def centroids(self):
        return [cell.centroid for cell in self]


def find_cells_in_channel(im):
    profile = vertical_mean(im)
    oldprofile = profile.copy()

    # ma, mi = profile.max(), profile.min()

    # empty channel detection
    # noinspection PyArgumentEqualDefault
    xprofile = threshold_outliers(profile, 2.0)
    if ((xprofile.max() - xprofile.min()) / xprofile.max()) < 0.5:
        return []
        pass

    profile = simple_baseline_correction(profile, window_width=None)

    profile = smooth(profile, signals(scipy.signal.flattop, 15))

    extrema_order = 15
    extrema = find_extrema_and_prominence(profile, order=extrema_order)

    newsignal = numpy.zeros_like(profile)
    newsignal[extrema.maxima] = extrema.prominence[extrema.maxima]
    newsignal = newsignal > 0

    # noinspection PyUnresolvedReferences
    newsignal[0] = 1
    # noinspection PyUnresolvedReferences
    newsignal[-1] = 1

    points, = numpy.where(newsignal)

    tp = numpy.zeros(len(points) * 2)
    tp[:len(points)] = points
    tp[len(points):-1] = points[1:]
    tp = tp.reshape((len(points), 2), order='F')[:-1]

    #thresh, bwimg = cv2.threshold(im, 0, 1, cv2.THRESH_OTSU)

    thresh = threshold_otsu(im)
    bwimg = im > thresh
    bwprof = vertical_mean(bwimg.astype(float))

    bnewsignal = newsignal.astype(float)

    for b, e in tp:
        bnewsignal[b:e] = numpy.mean(bwprof[b:e])

    multiplicator = (bnewsignal < 0.75)

    cells = multiplicator - newsignal

    cell_i = find_insides(cells)
    #cell_i = [[a, b] for a, b in cell_i if not (((a - b) == 0) or (a == 0) or (b == (len(cells) - 1)))]
    cell_i = [[a, b] for a, b in cell_i if not (((a - b) == 0))]

    cell_i = [[b, e] for b, e in cell_i if extrema.prominence[b:e].mean() > 10.0]

    with DebugPlot("celldetection_exp") as p:
        p.title("Cell detection exp")
        p.imshow(numpy.transpose(im))
        p.plot(oldprofile)

        p.plot(smooth(oldprofile, signals(numpy.hamming, 15)), color="purple")

        p.plot(extrema.xpts, extrema.prominence)

    with DebugPlot("celldetection_exp") as p:
        p.title("Cell detection exp")
        p.imshow(numpy.transpose(im))

        diff = corrected_numerical_differentiation(oldprofile)
        diff = abs(diff)

        diff = smooth(diff, signals(numpy.hamming, 5))
        diff /= diff.mean()
        # noinspection PyArgumentEqualDefault
        dextrema = find_extrema_and_prominence(diff, order=5)

        p.plot(diff + 100)
        p.scatter(dextrema.maxima, dextrema.signal[dextrema.maxima] + 100, color="green")
        p.scatter(dextrema.minima, dextrema.signal[dextrema.minima] + 100, color="red")
        p.plot(dextrema.prominence + 50)
        p.scatter([150], [150])  # the fu point

    with DebugPlot("celldetection") as p:
        p.title("Cell detection")
        p.imshow(numpy.transpose(im))
        p.plot(profile)
        p.plot(bwprof)

        p.plot(extrema.xpts, extrema.max_spline_points)
        p.plot(extrema.xpts, extrema.min_spline_points)

        p.scatter(extrema.maxima, profile[extrema.maxima])
        p.scatter(extrema.minima, profile[extrema.minima])

        celllines = [x for x in cell_i for x in x]

        p.vlines(celllines, [0] * len(celllines), [im.shape[1]] * len(celllines))

        for b, e in cell_i:
            pro = extrema.prominence[b:e].mean()
            p.text((b + e) / 2, -10, "%.1f" % (pro,))

    return cell_i
