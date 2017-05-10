# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from .imagestack import MultiImageStack

from xml.etree import cElementTree as ElementTree

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # code tend to throw warnings because of missing C extensions
    from tifffile import TiffFile


class OMETiffStack(MultiImageStack):
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
            'treat_z_as_mp': False,
            'subsample_t': 1,
            'subsample_xy': 1
        }, parameters)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.tiff = TiffFile(self.parameters['filename'], fastij=False, is_ome=True)  # shouldn't be ij, but safe is safe

        self.fp = self.tiff.pages[0]
        if not self.fp.is_ome:
            raise RuntimeError('Not an OMETiffStack')
        self.xml = None
        self.ns = ''
        self.xml_str = self.fp.tags['image_description'].value

        self.images = self._parse_ome_xml(self.xml_str)

    # noinspection PyProtectedMember
    def notify_fork(self):
        # noinspection PyProtectedMember
        """
        Notify class of fork. Important, as tifffile will otherwise return garbled data.

        """
        self.tiff._fh.close()
        # noinspection PyProtectedMember
        self.tiff._fh.open()

    @staticmethod
    def pixel_attrib_sanity_check(pa):
        """

        :param pa:
        :raise RuntimeError:
        """
        if 'BigEndian' in pa and pa['BigEndian'] == 'true':
            raise RuntimeError("Unsupported Pixel format")
        if 'Interleaved' in pa and pa['Interleaved'] == 'true':
            raise RuntimeError("Unsupported Pixel format")

    def _parse_ome_xml(self, xml):
        try:  # bioformats seem to copy some (wrongly encoded) strings verbatim
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError:
            root = ElementTree.fromstring(xml.decode('iso-8859-1').encode('utf-8'))

        self.xml = root

        self.ns = ns = root.tag.split('}')[0][1:]

        # sometimes string properties creep up, but then we don't care as we don't plan on using them ...
        def float_or_int(s):
            """

            :param s:
            :return:
            """
            try:
                if '.' in s:
                    return float(s)
                else:
                    return int(s)
            except ValueError:
                return s

        keep_pa = {'SizeZ', 'SizeY', 'SizeX', 'SignificantBits', 'PhysicalSizeX', 'PhysicalSizeY', 'SizeC', 'SizeT'}

        images = {}

        if bool(self.parameters['treat_z_as_mp']):  # handling for mal-encoded files
            image_nodes = [n for n in root.getchildren() if n.tag == ElementTree.QName(ns, 'Image')]
            # there will be only one image node
            imn = image_nodes[0]

            pixels = [n for n in imn.getchildren() if n.tag == ElementTree.QName(ns, 'Pixels')][0]

            pa = pixels.attrib

            self.pixel_attrib_sanity_check(pa)

            pai = list({k: v for k, v in pa.items() if k in keep_pa}.items())

            tiff_data = {
                (n.attrib['FirstC'], n.attrib['FirstT'], n.attrib['FirstZ']): n.attrib
                for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'TiffData')
            }
            planes = [dict(
                list(n.attrib.items()) +
                list(tiff_data[(n.attrib['TheC'], n.attrib['TheT'], n.attrib['TheZ'])].items()) + pai
            ) for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'Plane')]

            planes = [{k: float_or_int(v) for k, v in p.items()} for p in planes]
            multipoints = range(planes[0]['SizeZ'])
            images = {mp: [p for p in planes if p['TheZ'] == mp] for mp in multipoints}
            # more fixing

            def _correct_attributes(inner_p, inner_planes):
                inner_p['PositionX'] = inner_planes[0]['PositionX']
                inner_p['PositionY'] = inner_planes[0]['PositionY']
                inner_p['PositionZ'] = inner_planes[0]['PositionZ']
                inner_p['TheZ'] = 0
                return inner_p

            images = {mp: [_correct_attributes(p, planes) for p in planes] for mp, planes in images.items()}

        else:
            image_nodes = [n for n in root.getchildren() if n.tag == ElementTree.QName(ns, 'Image')]
            for n, imn in enumerate(image_nodes):
                pixels = [n for n in imn.getchildren() if n.tag == ElementTree.QName(ns, 'Pixels')][0]

                pa = pixels.attrib

                self.pixel_attrib_sanity_check(pa)

                pai = list({k: v for k, v in pa.items() if k in keep_pa}.items())

                tiff_data = {
                    (n.attrib['FirstC'], n.attrib['FirstT'], n.attrib['FirstZ']): n.attrib
                    for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'TiffData')
                }
                planes = [dict(
                    list(n.attrib.items()) +
                    list(tiff_data[(n.attrib['TheC'], n.attrib['TheT'], n.attrib['TheZ'])].items()) + pai
                ) for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'Plane')]

                planes = [{k: float_or_int(v) for k, v in p.items()} for p in planes]

                images[n] = planes

        return images

    def _get_image(self, **kwargs):
        subsampling_temporal = int(self.parameters['subsample_t'])
        subsampling_spatial = int(self.parameters['subsample_xy'])

        channel = 0
        if 'channel' in kwargs:
            channel = kwargs['channel']
        if channel in self.__class__.SimpleMapping:
            channel = self.__class__.SimpleMapping[channel]

        t = kwargs['t'] * subsampling_temporal

        tps = self.images[kwargs['pos']]
        tp = [tp for tp in tps if tp['TheT'] == t and tp['TheC'] == channel][0]

        return self.tiff.pages[tp['IFD']].asarray()[::subsampling_spatial, ::subsampling_spatial]

    def _get_meta(self, *args, **kwargs):
        what = args[0]

        t = kwargs['t'] if 't' in kwargs else 0
        pos = kwargs['pos'] if 'pos' in kwargs else 0

        subsampling_temporal = int(self.parameters['subsample_t'])
        subsampling_spatial = int(self.parameters['subsample_xy'])

        t *= subsampling_temporal

        image = [tp for tp in self.images[pos] if tp['TheT'] == t][0]

        return {
            'calibration': lambda: image['PhysicalSizeX'] * subsampling_spatial,
            'channels': lambda: image['SizeC'],
            'fluorescenceChannels': lambda: list(range(1, image['SizeC'])),
            'position': lambda: (
                image['PositionX'] if 'PositionX' in image else float('nan'),
                image['PositionY'] if 'PositionY' in image else float('nan'),
                image['PositionZ'] if 'PositionZ' in image else float('nan'),
            ),
            'time': lambda: image['DeltaT'],
            'timepoints': lambda: image['SizeT'] // subsampling_temporal,
            'multipoints': lambda: len(self.images)
        }[what]()


MultiImageStack.ExtensionRegistry['.ome.tiff'] = OMETiffStack
MultiImageStack.ExtensionRegistry['.ome.tif'] = OMETiffStack


class PlainTiffStack(MultiImageStack):
    """

    :param parameters:
    """
    SimpleMapping = {
        MultiImageStack.Phase_Contrast: 0,
        MultiImageStack.DIC: 0,
        MultiImageStack.Fluorescence: 1,
    }

    def __init__(self, parameters):
        self.generate_parameters_from_defaults({
            'interval': 1,
            'calibration': 1,
            'subsample_t': 1,
            'subsample_xy': 1
        }, parameters)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.tiff = TiffFile(self.parameters['filename'], fastij=False)  # fastij breaks the current _get_image !

        self.fp = self.tiff.pages[0]

        self.series = self.tiff.series[0]
        self._series_array = None

        if self.series.axes not in {'ZYX', 'TYX', 'ZCYX', 'TCYX'}:
            warnings.warn("Unsupported TIFF structure, processing will most likely fail.")

    # fix to a very nasty bug:
    # if multiprocessing on linux uses fork, the file descriptor
    # of the TiffFile object becomes shared between the child processes
    # the result? all move it around concurrently,
    # totally gobbling the input data to garbage!
    # thus, the class here closes and reopens the file descriptor
    # in the child process (which saves parsing time compared to
    # completely reinstantiating the while ImageStack)

    # noinspection PyProtectedMember
    def notify_fork(self):
        # noinspection PyProtectedMember
        """
        Notify class of fork. Important, as tifffile will otherwise return garbled data.

        """
        self.tiff._fh.close()
        # noinspection PyProtectedMember
        self.tiff._fh.open()

    @property
    def series_array(self):
        if self._series_array is None:
                self._series_array = self.series.asarray()
        return self._series_array

    def _get_image(self, **kwargs):
        subsampling_temporal = int(self.parameters['subsample_t'])
        subsampling_spatial = int(self.parameters['subsample_xy'])

        channel = 0
        if 'channel' in kwargs:
            channel = kwargs['channel']
        if channel in self.__class__.SimpleMapping:
            channel = self.__class__.SimpleMapping[channel]

        if len(self.series.axes) == 4:
            return self.series_array[
                   kwargs['t'] * subsampling_temporal,
                   channel,
                   ::subsampling_spatial,
                   ::subsampling_spatial
                   ]
        else:
            return self.tiff.pages[kwargs['t'] * subsampling_temporal].asarray()[
                   ::subsampling_spatial, ::subsampling_spatial
                    ]

    def _get_meta(self, *args, **kwargs):
        what = args[0]
        t = 0
        if 't' in kwargs:
            t = kwargs['t']

        subsampling_temporal = int(self.parameters['subsample_t'])
        subsampling_spatial = int(self.parameters['subsample_xy'])

        # pos = 0
        # if 'pos' in kwargs:
        #     pos = kwargs['pos']

        return {
            'calibration': lambda: float(self.parameters['calibration']) * subsampling_spatial,
            'channels': lambda: self.series_array.shape[1] if len(self.series.axes) == 4 else 1,
            'fluorescenceChannels': lambda: list(range(1, self.series_array.shape[1])) if len(self.series.axes) == 4 else [],
            'position': lambda: (0.0, 0.0, 0.0,),
            'time': lambda: float(self.parameters['interval']) * t * subsampling_temporal,
            'timepoints': lambda: (self.series_array.shape[0] if len(self.series.axes) == 4 else len(self.tiff.pages)) // subsampling_temporal,
            'multipoints': lambda: 1
        }[what]()


MultiImageStack.ExtensionRegistry['.tiff'] = PlainTiffStack
MultiImageStack.ExtensionRegistry['.tif'] = PlainTiffStack
