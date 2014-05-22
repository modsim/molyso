# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy

from .. import DebugPlot, tunable
from ..generic.signal import _spec_fft, _spec_bins_n, hires_powerspectrum, find_phase, find_extrema_and_prominence
from ..generic.smoothing import smooth, signals
from ..generic.util import horizontal_mean, vertical_mean, NotReallyATree, find_insides, numerical_differentiation, \
    one_every_n, normalize, threshold_outliers
from .cell_detection import Cells


class Channel(object):
    cells_type = Cells

    def __init__(self, image, left, right, top, bottom):
        self.image = image
        self.left = float(left)
        self.right = float(right)
        self.real_top = float(top)
        self.real_bottom = float(bottom)
        self.cells = None

        try:
            self.channelimage = self.crop_out_of_image(self.image.image)
        except AttributeError:
            pass

    @property
    def top(self):
        return self.real_top + self.image.shift[0]

    @property
    def bottom(self):
        return self.real_bottom + self.image.shift[0]

    @property
    def centroid(self):
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def crop_out_of_image(self, img):
        return img[self.real_top:self.real_bottom, self.left:self.right]

    def get_coordinates(self):
        return [[self.left, self.bottom], [self.right, self.bottom], [self.right, self.top], [self.left, self.top],
                [self.left, self.bottom]]

    def detect_cells(self):
        self.cells = self.__class__.cells_type(self.image, self, self.channelimage)

    def clean(self):
        self.cells.clean()
        if not self.image.keep_channel_image:
            del self.channelimage
            self.channelimage = None


treeprovider = NotReallyATree
#treeprovider = KDTree
#treeprovider = ckdtree.cKDTree


class Channels(list):
    """
        docstring
    """

    channel_type = Channel

    def __init__(self, image, bootstrap=True):
        super(Channels, self).__init__(self)
        self.image = image
        self.nearest_tree = None

        if not bootstrap:
            return

        positions, (upper, lower) = find_channels(self.image.image)

        for begin, end in positions:
            #noinspection PyMethodFirstArgAssignment
            self += [self.__class__.channel_type(self.image, begin, end, upper, lower)]

        with DebugPlot('channeldetection', 'result', 'onoriginal') as p:
            p.title("Detected channels (on original image)")
            p.imshow(self.image.original_image)
            for chan in self:
                coords = [self.image.cp(*pp) for pp in chan.get_coordinates()]
                p.poly_drawing_helper(coords, lw=1, edgecolor='r', fill=False)

        with DebugPlot('channeldetection', 'result', 'rotated') as p:
            p.title("Detected channels")
            p.imshow(self.image.image)
            for chan in self:
                coords = chan.get_coordinates()
                p.poly_drawing_helper(coords, lw=1, edgecolor='r', fill=False)

    def clean(self):
        for channel in self:
            channel.clean()

    @property
    def centroids(self):
        return [chan.centroid for chan in self]

    def find_nearest(self, pos):
        if self.nearest_tree is None:
            self.nearest_tree = treeprovider(self.centroids)
        return self.nearest_tree.query(pos)

    def align_with_and_return_indices(self, other_channels):
        if len(other_channels) == 0 or len(self) == 0:
            return []
        return [[ind, other_channels.find_nearest(cen)[1]] for ind, cen in enumerate(self.centroids)]

    def align_with(self, other_channels):
        if len(other_channels) == 0 or len(self) == 0:
            return []
        return [[channel, other_channels[other_channels.find_nearest(channel.centroid)[1]]] for channel in self]


def find_channels_in_profile_fft_assisted(profile):
    """

    @param profile:
    @return:
    """
    nothing_found = ([], 0, 0, 0, 0, -1, )

    differentiated_profile = numerical_differentiation(profile)
    differentiated_profile[0] = differentiated_profile[1]

    upper_profile = (differentiated_profile * (differentiated_profile > 0))
    lower_profile = -(differentiated_profile * (differentiated_profile < 0))

    upper_profile = smooth(upper_profile, signals(numpy.hamming, 15))  # 5
    lower_profile = smooth(lower_profile, signals(numpy.hamming, 15))  # 5

    with DebugPlot('channeldetection', 'details', 'differentials', 'smoothed') as p:
        p.title("Channeldetection/Differentials/Smoothed")
        p.plot(upper_profile)
        p.plot(lower_profile)

    # fft oversample
    n = 2

    # get the power spectra of the two signals
    frequencies_upper, fourier_value_upper = hires_powerspectrum(upper_profile, oversampling=n)
    frequencies_lower, fourier_value_lower = hires_powerspectrum(lower_profile, oversampling=n)

    mainfrequency_upper = frequencies_upper[numpy.argmax(fourier_value_upper)]
    mainfrequency_lower = frequencies_lower[numpy.argmax(fourier_value_lower)]

    with DebugPlot('channeldetection', 'details', 'powerspectra', 'upper') as p:
        p.title("Powerspectrum (upper)")
        p.semilogx(frequencies_upper, fourier_value_upper)
        p.title("mainfreq=%f" % mainfrequency_upper)
    with DebugPlot('channeldetection', 'details', 'powerspectra', 'lower') as p:
        p.title("Powerspectrum (lower)")
        p.semilogx(frequencies_lower, fourier_value_lower)
        p.title("mainfreq=%f" % mainfrequency_lower)

    mainfrequency = (mainfrequency_upper + mainfrequency_lower) / 2

    if mainfrequency == 0.0:
        return nothing_found

    profile_len = len(profile)

    absolute_differentiated_profile = numpy.absolute(differentiated_profile)

    width, = find_phase(upper_profile, lower_profile)

    mainfrequency += (width / ((profile_len / mainfrequency)))

    preliminary_signal = one_every_n(profile_len, mainfrequency) + one_every_n(profile_len, mainfrequency, shift=width)

    tmpsignal = numpy.zeros_like(absolute_differentiated_profile)

    tmpex = find_extrema_and_prominence(absolute_differentiated_profile, order=max(1, abs(width // 2)))
    tmpsignal[tmpex.maxima] = 1
    phase, = find_phase(tmpsignal, preliminary_signal)

    new_signal = \
        one_every_n(profile_len, mainfrequency, shift=phase + 0.5 * width) + \
        one_every_n(profile_len, mainfrequency, shift=phase + 0.5 * width + width)

    # dependence on 'new_signal' removed, works apparently without
    help_signal = normalize(smooth(absolute_differentiated_profile, signals(numpy.hamming, 50)))  # * new_signal

    # under certain conditions, the help signal may contain a totally extreme
    # maximum (testimage), I guess it's wiser to remove ity
    threshold_outliers(help_signal)

    ma = numpy.max(help_signal)
    mi = numpy.min(help_signal)

    threshold_factor = 0.5
    threshold = (ma - mi) * threshold_factor + mi

    help_signal = help_signal > threshold

    remaining_phase_shift, = find_phase(new_signal, absolute_differentiated_profile)
    ####

    new_signal = \
        one_every_n(profile_len, mainfrequency, shift=remaining_phase_shift + phase + 0.5 * width) + \
        one_every_n(profile_len, mainfrequency, shift=remaining_phase_shift + phase + 0.5 * width + width)

    left = help_signal
    right = help_signal[::-1]

    try:
        left = int(numpy.where(left)[0][0])
    except IndexError:
        left = 0

    try:
        right = int(len(right) - numpy.where(right)[0][0])
    except IndexError:
        right = int(len(right))

    new_signal[:left] = 0
    new_signal[right:] = 0

    positions, = numpy.where(new_signal)

    if len(positions) % 2 == 1:
        # either there's an additional line on the left or on the right
        # lets try to find it ...
        #print positions
        if abs(abs(positions[-1] - positions[-2]) - width) < 1:
            positions = positions[1:]
        elif abs(abs(positions[0] - positions[1]) - width) < 1:
            positions = positions[:-1]
        else:
            return nothing_found

    positions = positions.reshape((len(positions) // 2, 2))
    times = int((right - left) / mainfrequency)

    return positions, left, right, width, times, mainfrequency,


def _brute_force_fft_up_down_detect(img):
    f = _spec_bins_n(img.shape[1])

    def hori_main_freq(imgs):
        ft = numpy.absolute(_spec_fft(horizontal_mean(imgs)))
        ft[0] = 0
        ft[-1] = 0
        # to counteract the rebound effect similar to the larger detection routine above
        ft = smooth(ft, signals(numpy.hamming, 5))
        return f[numpy.argmax(ft)]

    overall_f = hori_main_freq(img)

    height = img.shape[0]

    collector = numpy.ones(height)

    d = 10.0
    break_condition = 5.0

    def recurse(top, bottom, orientation=0):
        if top >= bottom:
            return
        new_f = hori_main_freq(img[top:bottom, :])
        if abs(overall_f - new_f) < d:
            if (bottom - top) < break_condition:
                return
            mid = (top + bottom) // 2
            if orientation == -1:
                recurse(top, mid, orientation)
            elif orientation == 1:
                recurse(mid, bottom, orientation)

        else:
            collector[top:bottom] = 0

    recurse(0, height // 2, -1)
    recurse(height // 2, height, 1)
    result = sorted([(e - b, (b, e)) for b, e in find_insides(collector)], reverse=True)
    #if len(result) == 0:
    #    print(collector)
    #    print(height)

    _, (upper, lower) = result[0]
    return upper, lower


def find_channels(img):
    """
    channel finder
    :param img:
    :return:
    """

    upper, lower = _brute_force_fft_up_down_detect(img)

    profile = horizontal_mean(img)

    with DebugPlot('channeldetection', 'details', 'overview', 'horizontal') as p:
        p.title("Basic horizontal overview")
        p.plot(profile)

    with DebugPlot("channeldetection", "details", "overview", "vertical") as p:
        p.title("Basic vertical overview")
        p.plot(vertical_mean(img))

    profile = horizontal_mean(img[upper:lower, :])

    positions, left, right, width, times, mainfreq = find_channels_in_profile_fft_assisted(profile)

    return positions, (upper, lower)
