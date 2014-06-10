# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from helpers import *


def parser(parser):
    pass


def process(data, env, output):
    data = data[data.fluorescence == data.fluorescence]  # NaN check
    env.fluor = env.process(data.fluorescence)
    env.fluor_bg = env.process(data.fluorescence_background)
    env.times = env.process(data.timepoint)
    division_data = data[data.about_to_divide == 1]
    env.ages = numpy.log(2) / env.process(division_data.division_age)
    env.age_times = env.process(division_data.timepoint)


def ploting(env):
    xlabel('time [s]')
    ylabel('growth rate [min $^{-1}$]')

    plot(env.age_times, env.ages, label='µ')

    ax2 = twinx()
    ax2.set_ylabel('fluorescence [a.u.]')
    ax2.plot(0, 0, label='µ')
    ax2.plot(env.times, env.fluor, label='fluorescence')
    ax2.plot(env.times, env.fluor_bg - env.fluor_bg.min(), label='fluorescence background')
    legend()


if __name__ == '__main__':
    run_the_processor(parser, process, ploting, desc="Fluorescence")