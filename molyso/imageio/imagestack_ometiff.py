# -*- coding: utf-8 -*-
'''
documentation
'''
from __future__ import division, unicode_literals, print_function

from .imagestack import MultiImageStack

from xml.etree import cElementTree as ElementTree

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # code tend to throw warnings because of missing C extensions
    from .tifffile import TiffFile


class OMETiffStack(MultiImageStack):
    SimpleMapping = {
        MultiImageStack.Phase_Contrast: 0,
        MultiImageStack.DIC: 0,
        MultiImageStack.Fluorescence: 1,
    }

    def __init__(self, filename='', treat_z_as_mp=False):
        self.treat_z_as_mp = treat_z_as_mp
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.tiff = TiffFile(filename)
        self.fp = self.tiff.pages[0]
        if not self.fp.is_ome:
            raise Exception('OMETiffStack')
        self.xml = None
        self.ns = ''
        self.xml_str = self.fp.tags['image_description'].value

        self.images = self._parse_ome_xml(self.xml_str)

    def pixel_attrib_sanity_check(self, pa):
        if pa['BigEndian'] == 'true':
            raise Exception("Unsupported Pixel format")
        if pa['Interleaved'] == 'true':
            raise Exception("Unsupported Pixel format")

    def _parse_ome_xml(self, xml):
        try:  # bioformats seem to copy some (wrongly encoded) strings verbatim
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError:
            root = ElementTree.fromstring(xml.decode('iso-8859-1').encode('utf-8'))

        self.xml = root

        self.ns = ns = root.tag.split('}')[0][1:]

        float_or_int = lambda s: float(s) if '.' in s else int(s)

        keep_pa = {'SizeZ', 'SizeY', 'SizeX', 'SignificantBits', 'PhysicalSizeX', 'PhysicalSizeY', 'SizeC', 'SizeT'}

        images = {}

        if self.treat_z_as_mp:  # handling for malencoded files
            image_nodes = [n for n in root.getchildren() if n.tag == ElementTree.QName(ns, 'Image')]
            # there will be only one image node
            imn = image_nodes[0]

            pixels = [n for n in imn.getchildren() if n.tag == ElementTree.QName(ns, 'Pixels')][0]

            pa = pixels.attrib

            self.pixel_attrib_sanity_check(pa)

            pai = list({k: v for k, v in pa.items() if k in keep_pa}.items())

            tiffdata = {(n.attrib['FirstC'], n.attrib['FirstT'], n.attrib['FirstZ']): n.attrib for n in
                        pixels.getchildren() if n.tag == ElementTree.QName(ns, 'TiffData')}
            planes = [dict(
                list(n.attrib.items()) +
                list(tiffdata[(n.attrib['TheC'], n.attrib['TheT'], n.attrib['TheZ'])].items()) + pai
            ) for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'Plane')]

            planes = [{k: float_or_int(v) for k, v in p.items()} for p in planes]
            multipoints = range(planes[0]['SizeZ'])
            images = {mp: [p for p in planes if p['TheZ'] == mp] for mp in multipoints}
            # more fixing

            def _correct_attributes(p, planes):
                p['PositionX'] = planes[0]['PositionX']
                p['PositionY'] = planes[0]['PositionY']
                p['PositionZ'] = planes[0]['PositionZ']
                p['TheZ'] = 0
                return p

            images = {mp: [_correct_attributes(p, planes) for p in planes] for mp, planes in images.items()}

        else:
            image_nodes = [n for n in root.getchildren() if n.tag == ElementTree.QName(ns, 'Image')]
            for n, imn in enumerate(image_nodes):
                pixels = [n for n in imn.getchildren() if n.tag == ElementTree.QName(ns, 'Pixels')][0]

                pa = pixels.attrib

                self.pixel_attrib_sanity_check(pa)

                pai = list({k: v for k, v in pa.items() if k in keep_pa}.items())

                tiffdata = {(n.attrib['FirstC'], n.attrib['FirstT'], n.attrib['FirstZ']): n.attrib for n in
                            pixels.getchildren() if n.tag == ElementTree.QName(ns, 'TiffData')}
                planes = [dict(
                    list(n.attrib.items()) +
                    list(tiffdata[(n.attrib['TheC'], n.attrib['TheT'], n.attrib['TheZ'])].items()) + pai
                ) for n in pixels.getchildren() if n.tag == ElementTree.QName(ns, 'Plane')]

                planes = [{k: float_or_int(v) for k, v in p.items()} for p in planes]

                images[n] = planes

        return images


    def _get_image(self, **kwargs):
        channel = 0
        if 'channel' in kwargs:
            channel = kwargs['channel']
        if channel in self.__class__.SimpleMapping:
            channel = self.__class__.SimpleMapping[channel]

        tps = self.images[kwargs['pos']]
        tp = [tp for tp in tps if tp['TheT'] == kwargs['t'] and tp['TheC'] == channel][0]

        return self.tiff.pages[tp['IFD']].asarray()

    def _get_meta(self, *args, **kwargs):
        what = args[0]
        t = 0
        if 't' in kwargs:
            t = kwargs['t']
        pos = 0
        if 'pos' in kwargs:
            pos = kwargs['pos']

        image = [tp for tp in self.images[pos] if tp['TheT'] == t][0]

        return {
            'calibration': lambda: image['PhysicalSizeX'],
            'channels': lambda: image['SizeC'],
            'position': lambda: (image['PositionX'], image['PositionY'], image['PositionZ'],),
            'time': lambda: image['DeltaT'],
            'timepoints': lambda: image['SizeT'],
            'multipoints': lambda: len(self.images)
        }[what]()


MultiImageStack.ExtensionRegistry['.ome.tiff'] = OMETiffStack
MultiImageStack.ExtensionRegistry['.ome.tif'] = OMETiffStack