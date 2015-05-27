#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from helpers import *


# noinspection PyUnusedLocal
def parser(parser):
    pass


def process(data, env, output):
    data = data[data.fluorescence == data.fluorescence]  # NaN check
    output.fluorescence = env.fluor = env.process(data.fluorescence)
    output.fluorescence_background = env.fluor_bg = env.process(data.fluorescence_background)
    output.times = env.times = env.process(data.timepoint)


def ploting(env):
    plot(env.times, env.fluor, label='fluorescence')
    plot(env.times, env.fluor_bg - env.fluor_bg.min(), label='fluorescence background')
    legend()


if __name__ == '__main__':
    run_the_processor(parser, process, ploting, desc="Fluorescence")