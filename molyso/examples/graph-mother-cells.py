#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from helpers import *


def parser(parser):
    pass


def process(data, env, output):
    dgk = data.groupby(by=('channel_in_multipoint'))
    env.times = {}
    env.lens = {}
    for chan, dataset in dgk:
        env.times[chan] = []
        env.lens[chan] = []
        for tpn, sdataset in dataset.groupby(by=('timepoint_num')):
            o = next(iter(sdataset.channel_orientation))
            d = sdataset.sort('cellyposition', ascending=(o != 1))

            env.times[chan].append(next(iter(d.timepoint)))
            env.lens[chan].append(next(iter(d.length)))


def ploting(env):
    xlabel('time [s]')
    ylabel('cell length')
    xlim(50000, 100000)
    keys = list(env.times.keys())
    for k in keys[8:][:1]:
        plot(env.times[k], env.lens[k])


if __name__ == '__main__':
    run_the_processor(parser, process, ploting, desc="Fluorescence")