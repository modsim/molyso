# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import argparse
import numpy
import pandas

from matplotlib.pylab import *

try:
    # noinspection PyUnresolvedReferences
    import seaborn
except ImportError:
    pass


class AttributeAsKeyDict(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            return super(AttributeAsKeyDict, self).__getattribute__(item)

    def __setattr__(self, key, value):
        self[key] = value


def run_the_processor(parser_hook=None, process_hook=None, plot_hook=None, desc=''):
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('-mww', '--mean-window-width', dest='mean_window_width', default=100, type=int)
    parser.add_argument('-fo', '--figure-output', dest='figure_output', default='', type=str)

    args = parser.parse_args(sys.argv[1:])

    if parser_hook:
        parser_hook(parser)

    environment = AttributeAsKeyDict()
    output = AttributeAsKeyDict()

    if args.mean_window_width > 0:
        environment.process = lambda d: \
            numpy.array(pandas.rolling_mean(d, args.mean_window_width))[args.mean_window_width:]
    else:
        environment.process = lambda d: numpy.array(d)

    data = pandas.read_table(args.infile)
    data.sort(columns=['timepoint'], inplace=True)

    if process_hook:
        process_hook(data, environment, output)

    max_len = max(len(_) for _ in output.values())

    for k in output.keys():
        if len(output[k]) < max_len:
            output[k] = numpy.r_[output[k], [float('NaN')] * (max_len - len(output[k]))]

    if len(output) > 0:
        o = pandas.DataFrame(data=output, columns=[k for k in sorted(output.keys()) if k.startswith('time')] +
                                                  [k for k in sorted(output.keys()) if not k.startswith('time')])
        o.to_csv(args.outfile, sep='\t', index=False, na_rep='NaN')

    if plot_hook:
        if args.figure_output != '':
            figure()
            rcParams['pdf.fonttype'] = 42
            rcParams['text.usetex'] = False
            rcParams['mathtext.fontset'] = 'custom'
            rcParams['font.sans-serif'] = ['Arial']
            rcParams['font.cursive'] = ['Arial']
            plot_hook(environment)
            savefig(args.figure_output)
