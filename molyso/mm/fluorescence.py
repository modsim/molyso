# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, unicode_literals, print_function

import numpy
from .image import Image
from .cell_detection import Cell, Cells
from .channel_detection import Channel, Channels
from ..generic.rotation import apply_rotate_and_cleanup


class FluorescentCell(Cell):
    fluorescence_mean = float('nan')

    def __init__(self, *args, **kwargs):
        super(FluorescentCell, self).__init__(*args, **kwargs)

        try:  # reconstitution from flattened state will fail
            # due to missing image_fluorescence.
            # keep calm and carry on!

            cell_img = self.channel.image.image_fluorescence[
                       (self.local_top + self.channel.real_top):(self.local_bottom + self.channel.real_top),
                       self.channel.left:self.channel.right]

            self.fluorescence_mean = float(cell_img.mean())

        except AttributeError:
            self.fluorescence_mean = float('nan')

    @property
    def fluorescence(self):
        return self.fluorescence_mean - self.channel.image.background_fluorescence


    @property
    def fluorescence_raw(self):
        return self.fluorescence_mean


class FluorescentCells(Cells):
    cell_type = FluorescentCell


class FluorescentChannel(Channel):
    cells_type = FluorescentCells


class FluorescentChannels(Channels):
    channel_type = FluorescentChannel


class FluorescentImage(Image):
    channels_type = FluorescentChannels

    def __init__(self):
        super(FluorescentImage, self).__init__()
        self.image_fluorescence = None
        self.original_image_fluorescence = None
        self.background_fluorescence = float('NaN')


    def setup_fluorescence(self, fimg):
        self.image_fluorescence = fimg
        self.original_image_fluorescence = fimg

        self.background_fluorescence = 0.0

    def autorotate(self):
        super(FluorescentImage, self).autorotate()

        self.image_fluorescence, _, _, _ = apply_rotate_and_cleanup(self.image_fluorescence, self.angle)

    def clean(self):
        super(FluorescentImage, self).clean()
        del self.image_fluorescence
        del self.original_image_fluorescence

    def find_channels(self):
        super(FluorescentImage, self).find_channels()

        if len(self.channels) == 0:
            self.background_fluorescence = 0.0  # do something more meaningful ?!
        else:
            background_fluorescence_means = numpy.zeros((len(self.channels) - 1, 2), dtype=numpy.float64)

            previous_channel = self.channels[0]
            for n, next_channel in enumerate(self.channels[1:]):
                background_fragment = self.image_fluorescence[next_channel.real_top:next_channel.real_bottom,
                                      previous_channel.right:next_channel.left]
                background_fluorescence_means[n, 0] = background_fragment.mean()
                background_fluorescence_means[n, 1] = background_fragment.size
                previous_channel = next_channel

            background_fluorescence_means[:, 0] *= background_fluorescence_means[:, 1]

            self.background_fluorescence = numpy.sum(background_fluorescence_means[:, 0]) / \
                                           numpy.sum(background_fluorescence_means[:, 1])

    def flatten(self):

        channels = self.channels

        super(FluorescentImage, self).flatten()

        self.channels_cells_fluorescence_mean = [[cc.fluorescence_mean for cc in c.cells] for c in channels]


    def unflatten(self):
        super(FluorescentImage, self).unflatten()
        for n, _ in enumerate(self.channels):
            for cn, __ in enumerate(self.channels[n].cells):
                self.channels[n].cells[cn].fluorescence_mean = self.channels_cells_fluorescence_mean[n][cn]
        del self.channels_cells_fluorescence_mean
