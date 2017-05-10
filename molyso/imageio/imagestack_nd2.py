# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from nd2file import ND2MultiDim

from .imagestack import MultiImageStack


def catch_key_error(what, otherwise):
    """
    runs callable 'what' and catches KeyErrors, returning 'otherwise' if one occurred
    :param what: callable
    :param otherwise: alternate result in case of IndexError
    :return: result of 'what' or 'otherwise' in case of IndexError
    """
    try:
        return what()
    except KeyError:
        return otherwise


class ND2Stack(MultiImageStack):
    def __init__(self, parameters):
        self.generate_parameters_from_defaults({}, parameters)

        self.nd2 = ND2MultiDim(self.parameters['filename'])

    def _get_image(self, *args, **kwargs):
        image = self.nd2.image(multipoint=kwargs['pos'], timepoint=kwargs['t'])

        channel = kwargs['channel'] if 'channel' in kwargs else None

        if channel is None:
            return image
        else:
            if channel < 0:
                channel = self._get_channel_by_type(channel)

            return image[:, :, channel]

    def _get_meta(self, *args, **kwargs):
        what = args[0]

        t = kwargs['t'] if 't' in kwargs else 0
        pos = kwargs['pos'] if 'pos' in kwargs else 0

        return {
            'calibration': lambda: float(self.nd2.calibration),
            'channels': lambda: int(self.nd2.channels),
            'fluorescenceChannels': lambda: self.nd2.heuristic_fluorescences,
            'position': lambda: catch_key_error(
                lambda: (
                    float(self.nd2.multipoints[pos]['x']),
                    float(self.nd2.multipoints[pos]['y']),
                    float(self.nd2.multipoints[pos]['z']),),
                (float('nan'), float('nan'), float('nan'))),
            'time': lambda: float(self.nd2.get_time(self.nd2.calc_num(multipoint=pos, timepoint=t))),
            'timepoints': lambda: int(self.nd2.timepointcount),
            'multipoints': lambda: int(self.nd2.multipointcount)
        }[what]()

    def _get_channel_by_type(self, type_of_interest):
        return {
            MultiImageStack.Phase_Contrast: self.nd2.heuristic_pcm,
            MultiImageStack.DIC: self.nd2.heuristic_pcm,
            MultiImageStack.Fluorescence: self.nd2.heuristic_fluorescence,
        }[type_of_interest]

    def get_channel_by_type(self, type_of_interest):
        return self._get_channel_by_type(type_of_interest)


MultiImageStack.ExtensionRegistry['.nd2'] = ND2Stack
