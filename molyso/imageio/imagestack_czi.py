# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from .imagestack import MultiImageStack

import warnings

from czifile import CziFile

def _is_dimension_oi(d):
    return d in {'T', 'C', 'S'}


def _dim_shape_to_dict(dim, shape):
    if not isinstance(dim, str):
        dim = dim.decode()
    return dict(zip(list(dim), shape))


def _get_subblock_position(subblock):
    return _dim_shape_to_dict(subblock.axes, subblock.start)


def _get_subblock_identifier(subblock):
    return _normalize(_get_subblock_position(subblock))


def _get_image_from_subblock(subblock):
    return subblock.data_segment().data().reshape([s for s in subblock.stored_shape if s != 1])


def _normalize(d):
    return tuple([(k, v,) for k, v in sorted(d.items()) if _is_dimension_oi(k)])


class CziStack(MultiImageStack):
    """

    :param parameters:
    :raise RuntimeError:
    """
    SimpleMapping = {
        MultiImageStack.Phase_Contrast: 0,
        MultiImageStack.DIC: 0,
        MultiImageStack.Fluorescence: 1,
    }

    def __init__(self, parameters):
        self.generate_parameters_from_defaults({
            'subsample_t': 1,
            'subsample_xy': 1
        }, parameters)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.czi = CziFile(self.parameters['filename'])

        self.frames = {
            _get_subblock_identifier(subblock): subblock for subblock in self.czi.filtered_subblock_directory
            }

        self.size = _dim_shape_to_dict(self.czi.axes, self.czi.shape)

        self.metadata = self.czi.metadata(raw=False)

        scaling = self.metadata['ImageDocument']['Metadata']['Scaling']['Items']['Distance']

        # /ImageDocument
        calibration_x = float(
            next(iter([scale for scale in scaling if scale['Id'] == 'X']))['Value']
        ) * 1E6

        calibration_y = float(
            next(iter([scale for scale in scaling if scale['Id'] == 'Y']))['Value']
        ) * 1E6

        assert calibration_x == calibration_y

        self.calibration = calibration_x

        timestamps = None

        for entry in self.czi.attachment_directory:
            if entry.name == 'TimeStamps':
                timestamps = entry.data_segment().data()
                break
        assert timestamps is not None

        self.timestamps = timestamps

        positions = []

        for scene in sorted(self.metadata['ImageDocument'
                            ]['Metadata']['Information']['Image']['Dimensions']['S']['Scenes']['Scene'],
                key=lambda scene: int(scene['Index'])):
            x, y = scene['CenterPosition'].split(',')
            positions.append((float(x), float(y)))

        self.positions = positions

    # noinspection PyProtectedMember
    def notify_fork(self):
        # noinspection PyProtectedMember
        """
        Notify class of fork.

        """
        self.czi._fh.close()
        # noinspection PyProtectedMember
        self.czi._fh.open()

    def _get_image(self, **kwargs):
        subsampling_temporal = int(self.parameters['subsample_t'])
        subsampling_spatial = int(self.parameters['subsample_xy'])

        channel = 0
        if 'channel' in kwargs:
            channel = kwargs['channel']
        if channel in self.__class__.SimpleMapping:
            channel = self.__class__.SimpleMapping[channel]

        t = kwargs['t'] * subsampling_temporal

        pos = kwargs['pos']
        return _get_image_from_subblock(
            self.frames[_normalize(dict(C=channel, S=pos, T=t))])[::subsampling_spatial, ::subsampling_spatial]

    def _get_meta(self, *args, **kwargs):
        what = args[0]

        t = kwargs['t'] if 't' in kwargs else 0
        pos = kwargs['pos'] if 'pos' in kwargs else 0

        subsampling_temporal = int(self.parameters['subsample_t'])
        subsampling_spatial = int(self.parameters['subsample_xy'])

        t *= subsampling_temporal

        return {
            'calibration': lambda: self.calibration * subsampling_spatial,
            'channels': lambda: self.size['C'],
            'fluorescenceChannels': lambda: list(range(1, self.size['C'])),
            'position': lambda: (
                self.positions[pos][0],
                self.positions[pos][1],
                0.0,
            ),
            'time': lambda: self.timestamps[t],
            'timepoints': lambda: self.size['T'] // subsampling_temporal,
            'multipoints': lambda: self.size['S']
        }[what]()


MultiImageStack.ExtensionRegistry['.czi'] = CziStack
