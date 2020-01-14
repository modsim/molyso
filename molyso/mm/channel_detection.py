# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import sys
import warnings

import numpy as np

from ..debugging import DebugPlot
from ..generic.signal import find_phase, find_extrema_and_prominence, spectrum_fourier, spectrum_bins_by_length,\
    hires_power_spectrum, vertical_mean, horizontal_mean, normalize, threshold_outliers, find_insides, one_every_n,\
    hamming_smooth, each_image_slice
from .cell_detection import Cells
from ..generic.etc import NotReallyATree
from ..generic.tunable import tunable


class Channel(object):
    """

    :param image:
    :param left:
    :param right:
    :param top:
    :param bottom:
    """
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
        """


        :return:
        """
        return self.real_top + self.image.shift[0]

    @property
    def bottom(self):
        """


        :return:
        """
        return self.real_bottom + self.image.shift[0]

    @property
    def centroid(self):
        """


        :return:
        """
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def crop_out_of_image(self, image):
        """

        :param image:
        :return:
        """
        return image[int(self.real_top):int(self.real_bottom), int(self.left):int(self.right)].copy()

    def get_coordinates(self):
        """


        :return:
        """
        return [[self.left, self.bottom], [self.right, self.bottom], [self.right, self.top], [self.left, self.top],
                [self.left, self.bottom]]

    def detect_cells(self):
        """
        Performs Cell detection (by instantiating a Cells object).

        """
        self.cells = self.__class__.cells_type(self)

    def clean(self):
        """
        Peforms clean up routines.

        """
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

        find_channels_function = find_channels
        if hasattr(self.image, 'find_channels_function'):
            find_channels_function = self.image.find_channels_function

        positions, (upper, lower) = find_channels_function(self.image.image)

        for begin, end in positions:
            self.channels_list.append(self.__class__.channel_type(self.image, begin, end, upper, lower))

    def __len__(self):
        return len(self.channels_list)

    def __iter__(self):
        return iter(self.channels_list)

    def clean(self):
        """
        Performs clean up routines.

        """
        for channel in self:
            channel.clean()

    @property
    def centroids(self):
        """


        :return:
        """
        return [chan.centroid for chan in self]

    def find_nearest(self, pos):
        """

        :param pos:
        :return:
        """
        if self.nearest_tree is None:
            self.nearest_tree = treeprovider(self.centroids)
        return self.nearest_tree.query(pos)

    def align_with_and_return_indices(self, other_channels):
        """

        :param other_channels:
        :return:
        """
        if len(other_channels) == 0 or len(self) == 0:
            return []
        return [[ind, other_channels.find_nearest(cen)[1]] for ind, cen in enumerate(self.centroids)]


def horizontal_channel_detection(image):
    """

    @param image:
    @return:
    """

    # nothing found is a helper variable, which will be returned in case.
    nothing_found = ([], 0, 0, 0, 0, -1, )

    profile = horizontal_mean(image)
    profile_diff = np.diff(profile)

    upper_profile = +(profile_diff * (profile_diff > 0))
    lower_profile = -(profile_diff * (profile_diff < 0))

    upper_profile[
        upper_profile < upper_profile.max() *
        tunable(
            'channels.horizontal.noise_suppression_range.upper', 0.5,
            description="For channel detection, upper profile, noise reduction, reduction range."
        )
    ] *= tunable('channels.horizontal.noise_suppression_factor.upper', 0.1,
                 description="For channel detection, upper profile, noise reduction, reduction factor.")

    lower_profile[
        lower_profile < lower_profile.max() *
        tunable('channels.horizontal.noise_suppression_range.lower', 0.5,
                description="For channel detection, lower profile, noise reduction, reduction range.")
    ] *= tunable('channels.horizontal.noise_suppression_factor.lower', 0.1,
                 description="For channel detection, lower profile, noise reduction, reduction factor.")

    with DebugPlot('channel_detection', 'details', 'differences') as p:
        p.title("Channel detection/Differences")
        p.plot(upper_profile)
        p.plot(lower_profile)

    # oversample the fft n-times
    n = tunable('channels.horizontal.fft_oversampling', 8,
                description="For channel detection, FFT oversampling factor.")

    def calc_bins_freqs_main(the_profile):
        """

        :param the_profile:
        :return:
        """
        frequencies, fourier_value = hires_power_spectrum(the_profile, oversampling=n)
        fourier_value = hamming_smooth(fourier_value, tunable('channels.horizontal.fourier_smoothing', 3,
                                           description="For channel detection, smoothing width for the spectrum."))
        return frequencies, fourier_value, frequencies[np.argmax(fourier_value)]

    # get the power spectra of the two signals
    frequencies_upper, fourier_value_upper, mainfrequency_upper = calc_bins_freqs_main(upper_profile)
    frequencies_lower, fourier_value_lower, mainfrequency_lower = calc_bins_freqs_main(lower_profile)

    upper_profile = hamming_smooth(upper_profile,
                                   tunable('channels.horizontal.profile_smoothing_width.upper', 5,
                                           description="For channel detection, upper profile, smoothing window width."))
    lower_profile = hamming_smooth(lower_profile,
                                   tunable('channels.horizontal.profile_smoothing_width.lower', 5,
                                           description="For channel detection, lower profile, smoothing window width."))

    with DebugPlot('channel_detection', 'details', 'powerspectra', 'upper') as p:
        p.title("Powerspectrum (upper)")
        p.semilogx(frequencies_upper, fourier_value_upper)
        p.title("main_frequency=%f" % mainfrequency_upper)

    with DebugPlot('channel_detection', 'details', 'powerspectra', 'lower') as p:
        p.title("Powerspectrum (lower)")
        p.semilogx(frequencies_lower, fourier_value_lower)
        p.title("main_frequency=%f" % mainfrequency_lower)

    main_frequency = (mainfrequency_upper + mainfrequency_lower) / 2

    if main_frequency == 0.0:
        return nothing_found

    maximum_channel_count = profile.size / main_frequency

    allowed_maximum_channel_count = tunable(
        'channels.horizontal.channel_count.max', 50,
        description="For channel detection, maximum allowed channels to be detected.")

    allowed_minimum_channel_count = tunable(
        'channels.horizontal.channel_count.min', 3,
        description="For channel detection, minimum allowed channels to be detected.")

    if maximum_channel_count > allowed_maximum_channel_count or maximum_channel_count < allowed_minimum_channel_count:
        return nothing_found

    profile_diff_len = profile_diff.size
    absolute_differentiated_profile = np.absolute(profile_diff)

    width, = find_phase(upper_profile, lower_profile)

    if width > main_frequency:
        width = int(width % main_frequency)
    elif width < 0:
        width = int(width + main_frequency * np.ceil(abs(width / main_frequency)))

    main_frequency += (width / (profile_diff_len / main_frequency))

    preliminary_signal = \
        one_every_n(profile_diff_len, main_frequency) + one_every_n(profile_diff_len, main_frequency, shift=width)

    temporary_signal = np.zeros_like(absolute_differentiated_profile)

    temporary_extrema = find_extrema_and_prominence(absolute_differentiated_profile, order=max(1, abs(width // 2)))
    temporary_signal[temporary_extrema.maxima] = 1
    phase, = find_phase(temporary_signal, preliminary_signal)

    new_signal = \
        one_every_n(profile_diff_len, main_frequency, shift=phase + 0.5 * width) + \
        one_every_n(profile_diff_len, main_frequency, shift=phase + 1.5 * width)

    # dependence on 'new_signal' removed, works apparently without
    help_signal = normalize(hamming_smooth(absolute_differentiated_profile, 50))  # * new_signal

    # under certain conditions, the help signal may contain a totally extreme
    # maximum (test image), I guess it's wiser to remove it
    help_signal = threshold_outliers(help_signal)

    threshold_factor = tunable('channels.horizontal.threshold_factor', 0.2,
                               description="For channel detection, threshold factor for l/r border determination.")
    min_val, max_val = help_signal.min(), help_signal.max()
    help_signal = help_signal > (max_val - min_val) * threshold_factor + min_val

    remaining_phase_shift, = find_phase(new_signal, absolute_differentiated_profile)

    new_signal = \
        one_every_n(profile_diff_len, main_frequency, shift=remaining_phase_shift + phase + 0.5 * width) + \
        one_every_n(profile_diff_len, main_frequency, shift=remaining_phase_shift + phase + 1.5 * width)

    try:
        left = np.where(help_signal)[0][0]
    except IndexError:
        left = 0

    try:
        # noinspection PyUnresolvedReferences
        right = help_signal.size - np.where(help_signal[::-1])[0][0]
    except IndexError:
        # noinspection PyUnresolvedReferences
        right = help_signal.size

    left, right = int(left), int(right)

    new_signal[:left] = 0
    new_signal[right:] = 0

    positions, = np.where(new_signal)

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

# TODO fix the proper one, or merge them, or document this here, or use just the new one


def alternate_vertical_channel_region_detection(image):

    """

    :param image:
    :return:
    """
    f = spectrum_bins_by_length(image.shape[1])
    ft_h_s = tunable('channels.vertical.alternate.fft_smoothing_width', 3,
                     description="For channel detection (alternate, vertical), spectrum smoothing width.")

    def horizontal_mean_frequency(img_frag, clean_around=None, clean_width=0.0):
        """

        :param img_frag:
        :param clean_around:
        :param clean_width:
        :return:
        """
        ft = np.absolute(spectrum_fourier(horizontal_mean(img_frag)))

        ft /= 0.5 * ft[0]
        ft[0] = 0

        ft = hamming_smooth(ft, ft_h_s)

        if clean_around:
            ft[np.absolute(f - clean_around) > clean_width] = 0.0

        return ft.max(), f[np.argmax(ft)]

    split_factor = tunable('channels.vertical.alternate.split_factor', 60,
                           description="For channel detection (alternate, vertical), split factor.")

    collector = np.zeros(image.shape[0])

    for n, the_step, image_slice in each_image_slice(image, split_factor, direction='horizontal'):
        power_local_f, local_f = horizontal_mean_frequency(image_slice)

        collector[n*the_step:(n+1)*the_step] = local_f

    np.set_printoptions(threshold=sys.maxsize)
    # print(collector)
    int_collector = collector.astype(np.int32)

    if (int_collector == 0).all():
        warnings.warn(
            "Apparently no channel region was detectable. If the images are flipped, try filename?rotate=<90|270>",
            RuntimeWarning
        )
        return 0, len(int_collector)

    bins = np.bincount(int_collector)
    winner = np.argmax(bins[1:]) + 1

    delta = tunable('channels.vertical.alternate.delta', 5,
                    description="For channel detection (alternate, vertical), acceptable delta.")

    collector = (np.absolute(int_collector - winner) < delta) | (np.absolute(int_collector - 2*winner) < delta)

    return sorted(find_insides(collector), key=lambda pair: pair[1] - pair[0], reverse=True)[0]


FIRST_CALL, FROM_TOP, FROM_BOTTOM = 0, -1, 1


def vertical_channel_region_detection(image):
    """

    :param image:
    :return:
    """
    f = spectrum_bins_by_length(image.shape[1])

    ft_h_s = tunable('channels.vertical.recursive.fft_smoothing_width', 3,
                     description="For channel detection (recursive, vertical), spectrum smoothing width.")

    def horizontal_mean_frequency(img_frag, clean_around=None, clean_width=0.0):

        """

        :param img_frag:
        :param clean_around:
        :param clean_width:
        :return:
        """
        ft = np.absolute(spectrum_fourier(horizontal_mean(img_frag)))

        ft /= 0.5 * ft[0]

        ft[0] = 0

        ft = hamming_smooth(ft, ft_h_s)

        if clean_around:
            ft[np.absolute(f - clean_around) > clean_width] = 0.0

        return ft.max(), f[np.argmax(ft)]

    power_overall_f, overall_f = horizontal_mean_frequency(image)

    d = tunable('channels.vertical.recursive.maximum_delta', 2.0,
                description="For channel detection (recursive, vertical), maximum delta.")
    power_min_quotient = tunable('channels.vertical.recursive.power_min_quotient', 0.005,
                                 description="For channel detection (recursive, vertical), minimum power quotient")
    break_condition = tunable('channels.vertical.recursive.break_condition', 2.0,
                              description="For channel detection (recursive, vertical), recursive break condition.")

    current_clean_width = overall_f / 2.0

    def matches(img_frag):
        """

        :param img_frag:
        :return:
        """
        power_local_f, local_f = horizontal_mean_frequency(
            img_frag, clean_around=overall_f, clean_width=current_clean_width)
        return (abs(overall_f - local_f) < d) and ((power_local_f / power_overall_f) > power_min_quotient)

    height = image.shape[0]

    collector = np.zeros(height)

    def recursive_check(top, bottom, orientation=FIRST_CALL):
        """

        :param top:
        :param bottom:
        :param orientation:
        :return:
        """
        if (bottom - top) < break_condition:
            return

        mid = (top + bottom) // 2

        upper = matches(image[top:mid, :])
        lower = matches(image[mid:bottom, :])

        collector[top:mid] = upper
        collector[mid:bottom] = lower

        if orientation is FIRST_CALL:
            if upper:
                recursive_check(top, mid, FROM_TOP)
            if lower:
                recursive_check(mid, bottom, 1)
        elif orientation is FROM_TOP:
            if upper and lower:
                recursive_check(top, mid, FROM_TOP)
            elif not upper and lower:
                recursive_check(mid, bottom, FROM_TOP)
        elif orientation is FROM_BOTTOM:
            if lower and upper:
                recursive_check(mid, bottom, FROM_BOTTOM)
            elif not lower and upper:
                recursive_check(top, mid, FROM_BOTTOM)

    recursive_check(0, height)

    return sorted(find_insides(collector), key=lambda pair: pair[1] - pair[0], reverse=True)[0]


def find_channels(image):
    """
    channel finder
    :param image:
    :return:
    """

    method_to_use = tunable(
        'channels.vertical.method',
        'alternate',
        description="For channel detection, vertical method to use (either alternate or recursive).")

    if method_to_use == 'alternate':
        upper, lower = alternate_vertical_channel_region_detection(image)
    elif method_to_use == 'recursive':
        upper, lower = vertical_channel_region_detection(image)
    else:
        raise RuntimeError("Tunable set to unsupported vertical channel detection method.")

    profile = horizontal_mean(image)

    with DebugPlot('channel_detection', 'details', 'overview', 'horizontal') as p:
        p.title("Basic horizontal overview")
        p.plot(profile)

    with DebugPlot('channel_detection', 'details', 'overview', 'vertical') as p:
        p.title("Basic vertical overview")
        p.plot(vertical_mean(image))

    positions, left, right, width, times, mainfreq = horizontal_channel_detection(image[upper:lower, :])

    return positions, (upper, lower)
