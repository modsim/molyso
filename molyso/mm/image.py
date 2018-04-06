# -*- coding: utf-8 -*-
"""
    image.py

    General info
"""
from __future__ import division, unicode_literals, print_function

import math
import warnings

from ..generic.signal import fit_to_type
from ..generic.rotation import find_rotation, apply_rotate_and_cleanup
from ..generic.registration import translation_2x1d
from .channel_detection import Channels
from ..debugging import DebugPlot
from ..generic.tunable import tunable


class BaseImage(object):
    """
        An image object stores the original image, rotation and cropping information as well as the modified image.
        It points to channel objects as well.
    """

    def __init__(self):
        self.image = None
        self.original_image = None

        # metadata

        self.multipoint = 0
        self.timepoint = 0.0
        self.timepoint_num = 0

        self.calibration_px_to_mu = 1.0  # uncalibrated

        # metadata

        self.metadata = {'x': 0.0,
                         'y': 0.0,
                         'z': 0.0,
                         'time': 0.0,
                         'timepoint': 0,
                         'multipoint': 0,
                         'calibration_px_to_mu': 1.0,
                         'tag': '',
                         'tag_number': 0}

        self.angle = float('NaN')
        self.crop_height = 0
        self.crop_width = 0

        self.shift = [0, 0]

    def setup_image(self, image):
        """

        :param image:
        """
        self.image = image
        self.original_image = image

    def pixel_to_mu(self, pix):
        """
        converts a distance in pixels to micrometers using the calibration data of the image

        :param pix: pixel distance to convert
        :return: distance in micrometers as floating point number
        """
        return float(self.calibration_px_to_mu * pix)

    def mu_to_pixel(self, mu):
        """
        converts a distance in micrometers to pixels using the calibration data of the image

        :param mu: micrometer distance to convert
        :return: distance in pixels as floating point number
        """
        return float(mu / self.calibration_px_to_mu)

    def cp(self, x, y):
        """
        calculate point. transforms one coordinate in rotated, cropped space to one in the original image

        :param x: x position
        :param y: y position
        :return: a tuple of floats
        """
        hw = (self.original_image.shape[1] - self.crop_width) / 2
        hh = (self.original_image.shape[0] - self.crop_height) / 2
        r = math.sqrt((x - hw) ** 2 + (y - hh) ** 2)
        phi = math.atan2(y - hh, x - hw)
        x = r * math.cos(phi + self.angle / (180.0 / math.pi)) + hw
        y = r * math.sin(phi + self.angle / (180.0 / math.pi)) + hh
        return [x + self.crop_width, y + self.crop_height]


class AutoRotationProvider(object):
    """
    This mixin class adds automatic rotation
    (by :py:func:`molyso.generic.rotation.find_rotation`) functionality to the Image class.
    """

    def __init__(self):
        super(AutoRotationProvider, self).__init__()
        self.angle = float('NaN')
        self.crop_height = 0.0
        self.crop_width = 0.0

    def autorotate(self):
        """
        performs automatic rotation detection, rotation and cropping of the image
        :returns: None
        """

        if self.angle != self.angle:
            self.angle = find_rotation(
                self.image,
                steps=tunable(
                    'orientation-detection.strips', 10,
                    description="Number of strips for orientation correction."
                )
            )

        # noinspection PyAttributeOutsideInit
        self.image, self.angle, self.crop_height, self.crop_width = \
            apply_rotate_and_cleanup(self.image, self.angle)

        if self.image.size == 0:
            warnings.warn(
                "Autorotation failed. This likely means that the image is unsuitable for use with molyso.",
                RuntimeWarning
            )

            # noinspection PyAttributeOutsideInit
            self.image = self.original_image.copy()
            self.angle = 0.0
            self.crop_height = self.crop_width = 0


# noinspection PyUnresolvedReferences
class AutoRegistrationProvider(object):
    """
    This mixin class adds automatic registration
    (by :py:func:`molyso.generic.registration.translation_2x1d`) functionality to the Image class.
    """

    def __init__(self):
        super(AutoRegistrationProvider, self).__init__()
        self._fft_pair_cached = False
        self.shift = [0.0, 0.0]

    @property
    def fft_pair(self):
        """
        Retrieves the cached or calculates and caches the FT pair necessary for fast registration.

        :return:
        """
        if not getattr(self, '_fft_pair_cached', False):
            _, self._fft_pair_cached = translation_2x1d(self.original_image, self.original_image, return_a=True)
        return self._fft_pair_cached

    def autoregistration(self, reference):
        """
        Performs automatic registration of the image.

        :param reference:
        """
        shift, self._fft_pair_cached = translation_2x1d(None, self.original_image, ffts_a=reference.fft_pair,
                                                        return_b=True)

        air = self.angle * (math.pi / 180)  # angle in rads
        asi, aco = math.sin(air), math.cos(air)
        y, x = shift

        xn = x * aco - y * asi
        yn = x * asi + y * aco

        self.shift = [yn, xn]


cell_color = tunable('colors.cell', '#005b82', description="For debug output, cell color.")
channel_color = tunable('colors.channel', '#e7af12', description="For debug output, channel color.")


class Image(AutoRegistrationProvider, AutoRotationProvider, BaseImage):
    """
        An image object stores the original image, rotation and cropping information as well as the modified image.
        It points to channel objects as well.
    """

    channels_type = Channels

    def __init__(self):
        """
        creates an image object
        :return: image object
        """

        super(Image, self).__init__()

        self.flattened = False
        self.keep_channel_image = False

        self.pack_channel_image = False

        self.channels = None

        self.channel_orientation_cache = 0

        # empty data structures for flattening/unflattening
        self.channels_left = None
        self.channels_right = None
        self.channels_real_top = None
        self.channels_real_bottom = None
        self.channels_putative_orientations = None

        self.channels_cells_local_top = None
        self.channels_cells_local_bottom = None

        self.channel_images = None

    def setup_image(self, image):
        """

        :param image:
        """
        super(Image, self).setup_image(image)
        with DebugPlot('image', 'input') as p:
            p.title("Input image")
            p.imshow(self.image)

    def autorotate(self):
        """
        performs automatic rotation detection, rotation and cropping of the image
        :returns: None
        """

        super(Image, self).autorotate()

        with DebugPlot('image', 'rotated') as p:
            p.title("Rotated image")
            p.imshow(self.image)

    def find_channels(self):
        """
        calls channel detection routines (by instantiating the correct Channels object)
        :returns: None
        """

        self.channels = self.__class__.channels_type(self)

        with DebugPlot('channel_detection', 'result', 'on_original') as p:
            p.title("Detected channels (on original image)")
            p.imshow(self.original_image)
            for chan in self.channels:
                coords = [self.cp(*pp) for pp in chan.get_coordinates()]
                p.poly_drawing_helper(coords, lw=1, edgecolor=channel_color, fill=False, closed=True)

        with DebugPlot('channel_detection', 'result', 'rotated') as p:
            p.title("Detected channels")
            p.imshow(self.image)
            for chan in self.channels:
                coords = chan.get_coordinates()
                p.poly_drawing_helper(coords, lw=1, edgecolor=channel_color, fill=False, closed=True)

    def find_cells_in_channels(self):
        """
        performs cell detection by calling each channels cell detection routine.
        will visualize the outcome, if debugging is enabled
        :return:
        """

        # noinspection PyTypeChecker
        for channel in self.channels:
            channel.detect_cells()

        with DebugPlot('cell_detection', 'result', 'rotated') as p:
            self.debug_print_cells(p)

    def debug_print_cells(self, p):
        """

        :param p:
        """
        p.title("Detected cells")
        p.imshow(self.image)
        # noinspection PyTypeChecker
        for channel in self.channels:
            coordinates = channel.get_coordinates()
            p.poly_drawing_helper(coordinates, lw=1, edgecolor=channel_color, fill=False)

            for cell in channel.cells:
                coordinates = [
                    [channel.left, cell.bottom], [channel.right, cell.bottom],
                    [channel.right, cell.top], [channel.left, cell.top],
                    [channel.left, cell.bottom]
                ]

                p.poly_drawing_helper(coordinates, lw=0.5, edgecolor=cell_color, fill=False)

    def guess_channel_orientation(self):
        """


        :return:
        """
        if getattr(self, 'channel_orientation_cache', None) is None:
            orientations = [float(channel.putative_orientation) for channel in self.channels
                            if channel.putative_orientation != 0]
            self.channel_orientation_cache = 1 if (sum(orientations) / len(orientations)) > 0 else -1

        return self.channel_orientation_cache

    def flatten(self):
        """

        Flattens the image by reducing the object graph to information-identical array representations.
        This is done to ease the burden on the serializer and get smaller, conciser caches.

        It can as well be helpful, if serialized single frame data should be transferred over the wire.

        Warning, dependent on inner structure of dependent classes.

        :return:
        """

        channels = self.channels
        self.channels = None

        self.flattened = True

        self.channels_left = [c.left for c in channels]
        self.channels_right = [c.right for c in channels]
        self.channels_real_top = [c.real_top for c in channels]
        self.channels_real_bottom = [c.real_bottom for c in channels]
        self.channels_putative_orientations = [c.putative_orientation for c in channels]

        self.channels_cells_local_top = [[cc.local_top for cc in c.cells] for c in channels]
        self.channels_cells_local_bottom = [[cc.local_bottom for cc in c.cells] for c in channels]

        if self.keep_channel_image:
            def _pack_image(image):
                if self.pack_channel_image is False:
                    return image
                else:
                    if image is None:
                        return image
                    else:
                        return fit_to_type(image, self.pack_channel_image)

            self.channel_images = [_pack_image(c.channel_image) for c in channels]

    def unflatten(self):
        """
        Reconstructs the associated analysis results from the flattened state.

        :return:
        """

        self.channels = self.channels_type(self, bootstrap=False)

        # noinspection PyTypeChecker
        for channel_num, _ in enumerate(self.channels_left):
            channel = self.channels.__class__.channel_type(self, self.channels_left[channel_num],
                                                           self.channels_right[channel_num],
                                                           self.channels_real_top[channel_num],
                                                           self.channels_real_bottom[channel_num])

            channel.putative_orientation = self.channels_putative_orientations[channel_num]

            if self.channel_images is not None:
                channel.channel_image = self.channel_images[channel_num]

            self.channels.channels_list.append(channel)
            cells = channel.__class__.cells_type(channel, bootstrap=False)

            channel.cells = cells

            for cell_num, __ in enumerate(self.channels_cells_local_top[channel_num]):
                cells.cells_list.append(cells.cell_type(
                    self.channels_cells_local_top[channel_num][cell_num],
                    self.channels_cells_local_bottom[channel_num][cell_num],
                    channel
                ))

        self.flattened = False

        # empty data structures for flattening/unflattening
        self.channels_left = None
        self.channels_right = None
        self.channels_real_top = None
        self.channels_real_bottom = None
        self.channels_putative_orientations = None

        self.channels_cells_local_top = None
        self.channels_cells_local_bottom = None

    def clean(self):
        """
        Performs clean up, among other by removing the image data from the object.

        :return:
        """

        self.channels.clean()
        self.image = None
        self.original_image = None

        # quick hack, basically it should call a clean function in the parent class
        if hasattr(self, '_fft_pair_cached'):
            self._fft_pair_cached = False


