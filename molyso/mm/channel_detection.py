# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy

from .. import DebugPlot, tunable
from ..generic.signal import _spec_fft, _spec_bins_n, hires_powerspectrum, find_phase, find_extrema_and_prominence
from ..generic.smoothing import hamming_smooth
from ..generic.util import horizontal_mean, vertical_mean, NotReallyATree, find_insides, \
    one_every_n, normalize, threshold_outliers
from .cell_detection import Cells


class Channel(object):
    cells_type = Cells

    __slots__ = ['image', 'left', 'right', 'real_top', 'real_bottom', 'putative_orientation', 'cells', 'channel_image']

    def __init__(self, image, left, right, top, bottom):
        self.image = image
        self.left = float(left)
        self.right = float(right)
        self.real_top = float(top)
        self.real_bottom = float(bottom)
        self.putative_orientation = 0
        self.cells = None

        if self.image.image is not None:
            self.channel_image = self.crop_out_of_image(self.image.image)

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
        return img[self.real_top:self.real_bottom, self.left:self.right].copy()

    def get_coordinates(self):
        return [[self.left, self.bottom], [self.right, self.bottom], [self.right, self.top], [self.left, self.top],
                [self.left, self.bottom]]

    def detect_cells(self):
        self.cells = self.__class__.cells_type(self)

    def clean(self):
        self.cells.clean()
        if not self.image.keep_channel_image:
            self.channel_image = None


treeprovider = NotReallyATree
# treeprovider = KDTree
# treeprovider = ckdtree.cKDTree


class Channels(object):
    """
        docstring
    """

    __slots__ = ['channels_list', 'image', 'nearest_tree']

    channel_type = Channel

    def __init__(self, image, bootstrap=True):
        self.channels_list = []
        self.image = image
        self.nearest_tree = None

        if not bootstrap:
            return

        positions, (upper, lower) = find_channels(self.image.image)

        for begin, end in positions:
            #noinspection PyMethodFirstArgAssignment
            self.channels_list.append(self.__class__.channel_type(self.image, begin, end, upper, lower))

    def __len__(self):
        return len(self.channels_list)

    def __iter__(self):
        return iter(self.channels_list)

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

    profile_diff = numpy.diff(profile)

    upper_profile = (profile_diff * (profile_diff > 0))
    lower_profile = -(profile_diff * (profile_diff < 0))

    upper_profile[upper_profile < upper_profile.max() * 0.5] *= 0.1
    lower_profile[lower_profile < lower_profile.max() * 0.5] *= 0.1

    upper_profile = hamming_smooth(upper_profile, 5)  # 5
    lower_profile = hamming_smooth(lower_profile, 5)  # 5

    with DebugPlot('channeldetection', 'details', 'differentials', 'smoothed') as p:
        p.title("Channeldetection/Differentials/Smoothed")
        p.plot(upper_profile)
        p.plot(lower_profile)

    # oversample the fft n-times
    n = 4

    # get the power spectra of the two signals
    frequencies_upper, fourier_value_upper = hires_powerspectrum(upper_profile, oversampling=n)
    frequencies_lower, fourier_value_lower = hires_powerspectrum(lower_profile, oversampling=n)

    fourier_value_upper = hamming_smooth(fourier_value_upper, 3)
    fourier_value_lower = hamming_smooth(fourier_value_lower, 3)

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

    main_frequency = (mainfrequency_upper + mainfrequency_lower) / 2

    if main_frequency == 0.0:
        return nothing_found

    profile_diff_len = profile_diff.size
    absolute_differentiated_profile = numpy.absolute(profile_diff)

    width, = find_phase(upper_profile, lower_profile)

    if width > main_frequency:
        width = int(width % main_frequency)
    elif width < 0:
        width = int(width + main_frequency)

    main_frequency += (width / ((profile_diff_len / main_frequency)))

    preliminary_signal = one_every_n(profile_diff_len, main_frequency) + \
                         one_every_n(profile_diff_len, main_frequency, shift=width)

    tempoary_signal = numpy.zeros_like(absolute_differentiated_profile)

    tempoary_extrema = find_extrema_and_prominence(absolute_differentiated_profile, order=max(1, abs(width // 2)))
    tempoary_signal[tempoary_extrema.maxima] = 1
    phase, = find_phase(tempoary_signal, preliminary_signal)

    new_signal = \
        one_every_n(profile_diff_len, main_frequency, shift=phase + 0.5 * width) + \
        one_every_n(profile_diff_len, main_frequency, shift=phase + 0.5 * width + width)

    # dependence on 'new_signal' removed, works apparently without
    help_signal = normalize(hamming_smooth(absolute_differentiated_profile, 50))  # * new_signal

    # under certain conditions, the help signal may contain a totally extreme
    # maximum (testimage), I guess it's wiser to remove it
    threshold_outliers(help_signal)

    ma = numpy.max(help_signal)
    mi = numpy.min(help_signal)

    threshold_factor = 0.2
    threshold = (ma - mi) * threshold_factor + mi

    help_signal = help_signal > threshold

    remaining_phase_shift, = find_phase(new_signal, absolute_differentiated_profile)

    new_signal = \
        one_every_n(profile_diff_len, main_frequency, shift=remaining_phase_shift + phase + 0.5 * width) + \
        one_every_n(profile_diff_len, main_frequency, shift=remaining_phase_shift + phase + 1.5 * width)


    try:
        left = numpy.where(help_signal)[0][0]
    except IndexError:
        left = 0

    try:
        right = help_signal.size - numpy.where(help_signal[::-1])[0][0]
    except IndexError:
        right = help_signal.size

    left, right = int(left), int(right)

    new_signal[:left] = 0
    new_signal[right:] = 0

    positions, = numpy.where(new_signal)

    if len(positions) % 2 == 1:
        # either there's an additional line on the left or on the right
        # lets try to find it ...
        if abs(abs(positions[-1] - positions[-2]) - width) < 1:
            positions = positions[1:]
        elif abs(abs(positions[0] - positions[1]) - width) < 1:
            positions = positions[:-1]
        else:
            return nothing_found

    positions = positions.reshape((len(positions) // 2, 2))
    times = int((right - left) / main_frequency)

    # we worked with a simple difference, which made our signal shorter by one
    # add 1 to the appropriate variables to match positions with the original image

    positions += 1
    left, right = left + 1, right + 1

    return positions, left, right, width, times, main_frequency,


def _brute_force_fft_up_down_detect(img):
    f = _spec_bins_n(img.shape[1])

    def hori_main_freq(img_frag, clean_around=None, clean_width=0.0):

        ft = numpy.absolute(_spec_fft(horizontal_mean(img_frag)))

        ft /= 0.5 * ft[0]
        ft[0] = 0

        ft = hamming_smooth(ft, 3)

        if clean_around:
            ft[numpy.absolute(f - clean_around) > clean_width] = 0.0

        return ft.max(), f[numpy.argmax(ft)]

    power_overall_f, overall_f = hori_main_freq(img)

    d = 1.0
    power_min_quotient = 0.1
    break_condition = 2.0

    current_clean_width = overall_f / 2.0

    def matches(img_frag):
        power_local_f, local_f = hori_main_freq(img_frag, clean_around=overall_f, clean_width=current_clean_width)
        return (abs(overall_f - local_f) < d) and ((power_local_f / power_overall_f) > power_min_quotient)

    height = img.shape[0]

    collector = numpy.zeros(height)

    FIRST_CALL, FROM_TOP, FROM_BOTTOM = 0, -1, 1

    def recurse(top, bottom, orientation=FIRST_CALL):
        if (bottom - top) < break_condition:
            return

        mid = (top + bottom) // 2

        upper = matches(img[top:mid, :])
        lower = matches(img[mid:bottom, :])

        collector[top:mid] = upper
        collector[mid:bottom] = lower

        if orientation is FIRST_CALL:
            if upper:
                recurse(top, mid, FROM_TOP)
            if lower:
                recurse(mid, bottom, 1)
        elif orientation is FROM_TOP:
            if upper and lower:
                recurse(top, mid, FROM_TOP)
            elif not upper and lower:
                recurse(mid, bottom, FROM_TOP)
        elif orientation is FROM_BOTTOM:
            if lower and upper:
                recurse(mid, bottom, FROM_BOTTOM)
            elif not lower and upper:
                recurse(top, mid, FROM_BOTTOM)

    recurse(0, height)

    return sorted(find_insides(collector), key=lambda pair: pair[1] - pair[0], reverse=True)[0]


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
