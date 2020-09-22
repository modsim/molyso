# -*- coding: utf-8 -*-
"""
Module for handling datasets with fluorescence information.
Due to *molyso*'s object-oriented design, the added functionality can be achieved
merely by subclassing the particular non-fluorescence-handling base classes.
"""
from __future__ import division, unicode_literals, print_function

import numpy as np
from .image import Image
from .cell_detection import Cell, Cells
from .channel_detection import Channel, Channels
from ..generic.rotation import apply_rotate_and_cleanup
from ..generic.signal import fit_to_type


class FluorescentCell(Cell):
    """

    :param args:
    :param kwargs:
    """
    __slots__ = ['fluorescences_mean', 'fluorescences_std', 'fluorescences_min', 'fluorescences_max',
                 'fluorescences_median']

    def __init__(self, *args, **kwargs):
        super(FluorescentCell, self).__init__(*args, **kwargs)

        fluorescences_count = len(self.channel.image.image_fluorescences)

        self.fluorescences_mean = [float('nan')] * fluorescences_count
        self.fluorescences_std = [float('nan')] * fluorescences_count
        self.fluorescences_min = [float('nan')] * fluorescences_count
        self.fluorescences_max = [float('nan')] * fluorescences_count
        self.fluorescences_median = [float('nan')] * fluorescences_count

        try:
            # reconstitution from flattened state will fail
            # due to missing image_fluorescence.
            # keep calm and carry on!

            for f in range(fluorescences_count):
                if self.channel.image.image_fluorescences[f] is None:
                    continue

                fluorescence_cell_image = self.get_fluorescence_cell_image(f)

                self.fluorescences_mean[f] = float(fluorescence_cell_image.mean())
                self.fluorescences_std[f] = float(fluorescence_cell_image.std())
                self.fluorescences_min[f] = float(fluorescence_cell_image.min())
                self.fluorescences_max[f] = float(fluorescence_cell_image.max())
                self.fluorescences_median[f] = float(np.median(fluorescence_cell_image))

        except (AttributeError, TypeError):
            pass

    @property
    def fluorescences(self):
        """


        :return:
        """
        return [
            self.fluorescences_mean[f] - self.channel.image.background_fluorescences[f]
            for f in range(len(self.channel.image.image_fluorescences))
        ]

    @property
    def fluorescences_raw(self):
        """


        :return:
        """
        return self.fluorescences_mean

    def get_fluorescence_cell_image(self, f=0):
        """

        :param f:
        :return:
        """
        return self.crop_out_of_channel_image(self.channel.fluorescences_channel_image[f])


class FluorescentCells(Cells):
    """
    A subclass to handle fluorescences.
    """
    cell_type = FluorescentCell


class FluorescentChannel(Channel):
    """
    A subclass to handle fluorescences.

    :param image:
    :param left:
    :param right:
    :param top:
    :param bottom:
    """
    __slots__ = 'fluorescences_channel_image'
    cells_type = FluorescentCells

    def __init__(self, image, left, right, top, bottom):
        super(FluorescentChannel, self).__init__(image, left, right, top, bottom)

        fluorescences_count = len(image.image_fluorescences)

        self.fluorescences_channel_image = [None] * fluorescences_count

        for f in range(fluorescences_count):
            if image.image_fluorescences[f] is None:
                continue

            self.fluorescences_channel_image[f] = self.crop_out_of_image(image.image_fluorescences[f])


class FluorescentChannels(Channels):
    """
    A subclass to handle fluorescences.

    """
    channel_type = FluorescentChannel


class FluorescentImage(Image):
    """
    A subclass to handle fluorescences.

    """
    channels_type = FluorescentChannels

    def __init__(self):
        super(FluorescentImage, self).__init__()

        self.keep_fluorescences_image = False
        self.pack_fluorescences_image = False

        self.image_fluorescences = []
        self.original_image_fluorescences = []
        self.background_fluorescences = []

        self.channels_cells_fluorescences_mean = None
        self.channels_cells_fluorescences_std = None
        self.channels_cells_fluorescences_min = None
        self.channels_cells_fluorescences_max = None
        self.channels_cells_fluorescences_median = None

        self.channel_fluorescences_images = None

    def setup_add_fluorescence(self, fimg):
        """

        :param fimg:
        """
        self.image_fluorescences.append(fimg)
        self.original_image_fluorescences.append(fimg)

        self.background_fluorescences.append(0.0)

    def autorotate(self):
        """
        Rotates the image, as well as the fluorescence channels.

        """
        super(FluorescentImage, self).autorotate()
        self.image_fluorescences = [
            apply_rotate_and_cleanup(fluorescence_image, self.angle)[0]
            for fluorescence_image in self.image_fluorescences]

    def clean(self):
        """
        Performs clean up routines.

        """
        super(FluorescentImage, self).clean()
        fluorescences_count = len(self.image_fluorescences)
        self.image_fluorescences = [None] * fluorescences_count
        self.original_image_fluorescences = [None] * fluorescences_count

    def find_channels(self):
        """
        Find channels in the image.

        """
        super(FluorescentImage, self).find_channels()

        fluorescences_count = len(self.image_fluorescences)

        if len(self.channels) == 0:
            # do something more meaningful ?!
            self.background_fluorescences = [0.0] * fluorescences_count
        else:

            for i in range(fluorescences_count):

                fluorescence_image = self.image_fluorescences[i]

                background_fluorescence_means = np.zeros((len(self.channels) - 1, 2), dtype=np.float64)

                channel_iterator = iter(self.channels)

                previous_channel = next(channel_iterator)
                for n, next_channel in enumerate(channel_iterator):
                    background_fragment = fluorescence_image[
                        int(next_channel.real_top):int(next_channel.real_bottom),
                        int(previous_channel.right):int(next_channel.left)
                    ]
                    background_fluorescence_means[n, 0] = background_fragment.mean()
                    background_fluorescence_means[n, 1] = background_fragment.size
                    previous_channel = next_channel

                background_fluorescence_means[:, 0] *= background_fluorescence_means[:, 1]

                self.background_fluorescences[i] = \
                    np.sum(background_fluorescence_means[:, 0]) / np.sum(background_fluorescence_means[:, 1])

    def flatten(self):
        """
        Flattens the image by reducing the object graph to information-identical array representations.
        This is done to ease the burden on the serializer and get smaller, conciser caches.

        It can as well be helpful, if serialized single frame data should be transferred over the wire.

        Warning, dependent on inner structure of dependent classes.

        :return:
        """
        channels = self.channels

        fluorescences_count = len(self.image_fluorescences)

        self.channels_cells_fluorescences_mean = [
            [[cc.fluorescences_mean[f] for cc in c.cells] for c in channels]
            for f in range(fluorescences_count)
        ]
        self.channels_cells_fluorescences_std = [
            [[cc.fluorescences_std[f] for cc in c.cells] for c in channels]
            for f in range(fluorescences_count)
        ]
        self.channels_cells_fluorescences_min = [
            [[cc.fluorescences_min[f] for cc in c.cells] for c in channels]
            for f in range(fluorescences_count)
        ]
        self.channels_cells_fluorescences_max = [
            [[cc.fluorescences_max[f] for cc in c.cells] for c in channels]
            for f in range(fluorescences_count)
        ]
        self.channels_cells_fluorescences_median = [
            [[cc.fluorescences_median[f] for cc in c.cells] for c in channels]
            for f in range(fluorescences_count)
        ]

        if self.keep_fluorescences_image:
            def _pack_image(image):
                if self.pack_fluorescences_image is False:
                    return image
                else:
                    if image is None:
                        return image
                    else:
                        return fit_to_type(image, self.pack_fluorescences_image)

            self.channel_fluorescences_images = [
                [_pack_image(ci) for ci in c.fluorescences_channel_image] for c in channels
            ]

        super(FluorescentImage, self).flatten()

    def unflatten(self):
        """
        Reconstructs the associated analysis results from the flattened state.

        """
        super(FluorescentImage, self).unflatten()
        fluorescences_count = len(self.image_fluorescences)
        for n, channel in enumerate(self.channels):
            if self.channel_fluorescences_images is not None:
                channel.fluorescences_channel_image = self.channel_fluorescences_images[n]

            for cn, cell in enumerate(channel.cells):
                cell.fluorescences_mean = [
                    self.channels_cells_fluorescences_mean[f][n][cn] for f in range(fluorescences_count)
                ]
                cell.fluorescences_std = [
                    self.channels_cells_fluorescences_std[f][n][cn] for f in range(fluorescences_count)
                ]
                cell.fluorescences_min = [
                    self.channels_cells_fluorescences_min[f][n][cn] for f in range(fluorescences_count)
                ]
                cell.fluorescences_max = [
                    self.channels_cells_fluorescences_max[f][n][cn] for f in range(fluorescences_count)
                ]
                cell.fluorescences_median = [
                    self.channels_cells_fluorescences_median[f][n][cn] for f in range(fluorescences_count)
                ]
        self.channels_cells_fluorescences_mean = None
        self.channels_cells_fluorescences_std = None
        self.channel_fluorescences_images = None
