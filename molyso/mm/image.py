# -*- coding: utf-8 -*-
"""
    image.py

    General info
"""
from __future__ import division, unicode_literals, print_function

import math
from ..generic.rotation import find_rotation, apply_rotate_and_cleanup
from ..generic.registration import translation_2x1d
from .channel_detection import Channels
from .. import DebugPlot


class BaseImage(object):
    """
        An image object stores the original image, rotation and cropping information as well as the modified image.
        It points to channel objects as well.
    """

    def __init__(self):
        """
        creates an image object
        :return: baseimage object
        """

        self.image = None
        self.original_image = None

        # metadata

        self.multipoint = 0
        self.timepoint = 0.0
        self.timepoint_num = 0

        self.calibration_px_to_mu = 1.0  # "uncalibrated"

        # metadata

        self.metadata = {"x": 0.0,
                         "y": 0.0,
                         "z": 0.0,
                         "time": 0.0,
                         "timepoint": 0,
                         "multipoint": 0,
                         "calibration_px_to_mu": 1.0,
                         "tag": "",
                         "tag_number": 0}

        self.angle = float('NaN')
        self.crop_height = 0
        self.crop_width = 0

        self.shift = [0, 0]

    def setup_image(self, img):
        self.image = img
        self.original_image = img

    def pixel_to_mu(self, pix):
        return float(self.calibration_px_to_mu * pix)

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
            self.angle = find_rotation(self.image)

        # noinspection PyAttributeOutsideInit
        self.image, self.angle, self.crop_height, self.crop_width = \
            apply_rotate_and_cleanup(self.image, self.angle)


# noinspection PyUnresolvedReferences
class AutoRegistrationProvider(object):
    def __init__(self):
        super(AutoRegistrationProvider, self).__init__()
        self._fft_pair_cached = False
        self.shift = [0.0, 0.0]

    @property
    def fft_pair(self):
        if not getattr(self, '_fft_pair_cached', False):
            _, self._fft_pair_cached = translation_2x1d(self.original_image, self.original_image, return_a=True)
        return self._fft_pair_cached

    def autoregister(self, reference):
        shift, self._fft_pair_cached = translation_2x1d(None, self.original_image, ffts_a=reference.fft_pair,
                                                        return_b=True)

        air = self.angle * (math.pi / 180)  # angle in rads
        asi, aco = math.sin(air), math.cos(air)
        y, x = shift

        xn = x * aco - y * asi
        yn = x * asi + y * aco

        self.shift = [yn, xn]


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
        self.channels = None

        # empty data structures for flattening/unflattening
        self.channels_left = None
        self.channels_right = None
        self.channels_real_top = None
        self.channels_real_bottom = None

        self.channels_cells_local_top = None
        self.channels_cells_local_bottom = None

    def setup_image(self, img):
        super(Image, self).setup_image(img)
        with DebugPlot("image", "input") as p:
            p.title("Input image")
            p.imshow(self.image)

    def autorotate(self):
        """
        performs automatic rotation detection, rotation and cropping of the image
        :returns: None
        """

        super(Image, self).autorotate()

        with DebugPlot("image", "rotated") as p:
            p.title("Rotated image")
            p.imshow(self.image)

    def find_channels(self):
        """
        calls channel detection routines (by instanciating
        :returns: None
        """

        self.channels = self.__class__.channels_type(self)

    def find_cells_in_channels(self):
        """
        performs cell detection by calling each channels cell detection routine.
        will visualize the outcome, if debugging is enabled
        :return:
        """

        # noinspection PyTypeChecker
        for channel in self.channels:
            channel.detect_cells()

        with DebugPlot("celldetection", "result", "rotated") as p:
            p.title("Detected cells")
            p.imshow(self.image)
            # noinspection PyTypeChecker
            for channel in self.channels:
                coords = channel.get_coordinates()
                p.poly_drawing_helper(coords, lw=1, edgecolor='r', fill=False)

                for cell in channel.cells:
                    coords = [[channel.left, cell.bottom], [channel.right, cell.bottom],
                              [channel.right, cell.top], [channel.left, cell.top],
                              [channel.left, cell.bottom]]

                    p.poly_drawing_helper(coords, lw=0.5, edgecolor='b', fill=False)

    def flatten(self):
        """
        flattens the internal data structures into a more compact way
        (or to be more precise: in a way which is more efficiently serialized)

        Warning, dependant on inner structure of dependant classes.

        :return:
        """

        channels = self.channels
        self.channels = None

        self.flattened = True

        self.channels_left = [c.left for c in channels]
        self.channels_right = [c.right for c in channels]
        self.channels_real_top = [c.real_top for c in channels]
        self.channels_real_bottom = [c.real_bottom for c in channels]

        self.channels_cells_local_top = [[cc.local_top for cc in c.cells] for c in channels]
        self.channels_cells_local_bottom = [[cc.local_bottom for cc in c.cells] for c in channels]

    def unflatten(self):
        """
        unflattens the internal data structures from an object with previously flattened structures
        :return:
        """

        self.channels = self.channels_type(self, bootstrap=False)

        # noinspection PyTypeChecker
        for n, _ in enumerate(self.channels_left):
            chan = self.channels.__class__.channel_type(self, self.channels_left[n], self.channels_right[n],
                                                        self.channels_real_top[n], self.channels_real_bottom[n])

            self.channels.append(chan)
            cells = chan.__class__.cells_type(self, chan, [], bootstrap=False)

            chan.cells = cells

            for cn, __ in enumerate(self.channels_cells_local_top[n]):
                ltop, lbottom = self.channels_cells_local_top[n][cn], self.channels_cells_local_bottom[n][cn]

                cell = cells.cell_type(ltop, lbottom, chan)

                cells.append(cell)

        self.flattened = False

        del self.channels_left
        del self.channels_right
        del self.channels_real_top
        del self.channels_real_bottom

        del self.channels_cells_local_top
        del self.channels_cells_local_bottom

    def clean(self):
        """
        removes references to image data to allow for cleanup
        :return:
        """

        self.channels.clean()
        del self.image
        del self.original_image


