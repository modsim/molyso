# -*- coding: utf-8 -*-
"""

"""

import os
import sys
import argparse
import pandas
import xml.etree.ElementTree as ET
from collections import namedtuple
from itertools import chain
from copy import deepcopy

def create_argparser():
    argparser = argparse.ArgumentParser(description="molyso2vizardous molyso-tabular data format to Vizardous metaXML/phyloXML converter")

    def _error(message=''):
        argparser.print_help()
        sys.stderr.write("%serror: %s%s" % (os.linesep, message, os.linesep,))
        sys.exit(1)

    argparser.error = _error

    argparser.add_argument('input', metavar='input', type=str, help="input file")
    argparser.add_argument('-o', '--output', dest='output', type=str, default=None)
    argparser.add_argument('-d', '--minimum-depth', dest='minimum_depth', default=0, type=int)
    argparser.add_argument('-q', '--quiet', dest='quiet', default=False, action='store_true')

    return argparser

def unit(value):
    return {'unit': value}

def root_phyloXML():
    return ET.Element(
        'phyloxml', {
            'xmlns': 'http://www.phyloxml.org',
            'xmlns:metaxml': 'http://13cflux.net/static/schemas/metaXML/2',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://www.phyloxml.org http://www.phyloxml.org/1.10/phyloxml.xsd'
        }
    )


def root_metaXML():
    return ET.Element(
        'metaInformation', {
            'xmlns': 'http://13cflux.net/static/schemas/metaXML/2',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://13cflux.net/static/schemas/metaXML/2 metaXML-2.7.0.xsd'
        }
    )


def empty_metaXML(project_name, duration_in_seconds):
    metaXML = root_metaXML()
    ET.SubElement(metaXML, 'projectName').text = project_name
    ET.SubElement(metaXML, 'experimentDuration', unit('min')).text = str(duration_in_seconds / 60.0)
    return metaXML


def empty_phyloXML(project_name):
    phyloXML = root_phyloXML()
    ET.SubElement(phyloXML, 'metaxml:projectName').text = project_name
    return phyloXML


def empty_trees(project_name, duration):
    return empty_phyloXML(project_name), empty_metaXML(project_name, duration)


def molyso2vizardous(data, phyloXML, metaXML):
    def make_cell(cell):
        cell_element = ET.Element('cell', {'id': str(cell.uid_thiscell)})

        ET.SubElement(cell_element, 'length', unit('um')).text = str(cell.length)

        ET.SubElement(cell_element, 'area', unit('um^2')).text = str(cell.length * cell.channel_width)

        mapping = {0: 'yfp', 1: 'crimson'}

        fluorescence_count = cell.fluorescence_count
        # fluorescence_count = 2

        if fluorescence_count > 0:
            fluorescences_node = ET.SubElement(cell_element, 'fluorescences')
            for n in range(fluorescence_count):
                fluorescence = ET.SubElement(fluorescences_node, 'fluorescence', {'channel': mapping[n]})
                ET.SubElement(fluorescence, 'mean', unit('au')).text = str(0.0)
                ET.SubElement(fluorescence, 'stddev', unit('au')).text = str(0.0)

        return cell_element

    def make_frame(frame_number, frame_time):
        frame = ET.SubElement(metaXML, 'frame', {'id': str(frame_number - 1)})
        ET.SubElement(frame, 'elapsedTime', unit('min')).text = str(frame_time)
        return frame

    def make_clade(cell):
        clade = ET.Element('clade')
        ET.SubElement(clade, 'name').text = str(cell.uid_thiscell)
        ET.SubElement(clade, 'branch_length').text = '1.0'
        return clade

    data = data.sort('timepoint')

    timepoints = {
        the_frame_number: make_frame(the_frame_number, the_frame.timepoint / 60.0)
        for the_frame_number, the_frame in data.groupby('timepoint_num').mean().iterrows()
    }

    current_positions = {}

    named_row_tuple = namedtuple('named_row_tuple', ['index'] + list(data))

    for row_tuple in data.itertuples():
        row = named_row_tuple(*row_tuple)

        clade = make_clade(row)

        timepoints[row.timepoint_num].append(make_cell(row))

        if row.uid_cell in current_positions:
            current_positions[row.uid_cell][-1].append(clade)
            current_positions[row.uid_cell].append(clade)
        else:
            if row.uid_parent == 0:
                ET.SubElement(phyloXML, 'phylogeny', {'rooted': str('false')}).append(clade)
                current_positions[row.uid_cell] = [clade]
            else:
                current_positions[row.uid_parent][-1].append(clade)
                current_positions[row.uid_cell] = [clade]

    return phyloXML, metaXML


def depth(element, num=0):
    return max(chain([num], (depth(child, num+1) for child in element)))


def filter_trees(phyloXML, metaXML, keep=0):

    for n, phylogeny in enumerate(phyloXML.findall('phylogeny')):
        if n != keep:
            phyloXML.remove(phylogeny)

    names_kept = set(p.text for p in phyloXML.findall('.//name'))

    for frame in metaXML.findall('.//frame'):
        for cell in frame:
            if cell.tag == 'cell':
                if cell.attrib.get('id') not in names_kept:
                    frame.remove(cell)


def main():
    argparser = create_argparser()
    args = argparser.parse_args()

    data = pandas.read_table(args.input)

    duration = data.timepoint.max()
    project_name = 'Mother Machine Experiment'


    for (multipoint, channel_in_multipoint), subset in data.groupby(by=['multipoint', 'channel_in_multipoint']):


        phyloXML, metaXML = empty_trees(project_name, duration)

        phyloXML, metaXML = molyso2vizardous(subset, phyloXML, metaXML)

        if args.output is None:
            args.output, _ = os.path.splitext(args.input)

        PHYLO_SUFFIX = 'phylo.xml'
        META_SUFFIX = 'meta.xml'


        def write_outputs(result_files):
            for file_suffix, tree in result_files.items():
                with open('%s.%s' % (args.output, file_suffix), mode='wb+') as fp:
                    ET.ElementTree(tree).write(fp)

        channel_identifier = 'mp.%d.channel.%d' % (multipoint, channel_in_multipoint,)

        jobs = phyloXML.findall('phylogeny')

        for n, phylogeny in enumerate(jobs):
            depth_of_phylogeny = depth(phylogeny)

            if depth_of_phylogeny < args.minimum_depth:
                continue

            copy_phyloXML = deepcopy(phyloXML)
            copy_metaXML = deepcopy(metaXML)

            filter_trees(copy_phyloXML, copy_metaXML, n)

            infix = '%s.%d.of.%d.depth.%d.' % (channel_identifier, n+1, len(jobs), depth_of_phylogeny)

            write_outputs({infix + PHYLO_SUFFIX: copy_phyloXML, infix + META_SUFFIX: copy_metaXML})

# write_outputs({PHYLO_SUFFIX: phyloXML, META_SUFFIX: metaXML})


if __name__ == '__main__':
    main()