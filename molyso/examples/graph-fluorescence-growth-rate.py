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
    gtp = data.groupby(by='timepoint_num')

    env.fluor = env.process(gtp.mean().fluorescence)
    env.fluor_bg = env.process(gtp.mean().fluorescence_background)
    env.times = env.process(gtp.mean().timepoint)

    all_cells = data
    division_data = data[data.about_to_divide == 1]

    ad = all_cells.groupby(by='timepoint_num')

    gdd = division_data.groupby(by='timepoint_num')

    env.ages = numpy.log(2) / env.process(gdd.mean().division_age)
    env.age_times = env.process(gdd.mean().timepoint)

    env.counts = env.process(gdd.count().division_age)

    env.ages_original = env.ages.copy()

    env.ages[env.counts < 5] = float('NaN')

    env.cell_counts = env.process(ad.count().division_age)
    env.cell_times = env.process(ad.mean().timepoint)

    output.growth_mu = env.ages
    output.growth_times = env.age_times
    output.growth_times_hours = env.age_times / 60.0 / 60.0

    output.fluorescence_f = env.fluor
    output.fluorescence_times = env.times
    output.fluorescence_times_hours = env.times / 60.0 / 60.0


def ploting(env):
    xlabel('time [s]')
    ylabel('growth rate [min $^{-1}$]')

    plot(env.age_times, env.ages, label='µ')

    ax2 = twinx()
    ax2.set_ylabel('fluorescence [a.u.]')
    ax2.plot(0, 0, label='µ')
    ax2.plot(env.times, env.fluor, label='fluorescence')
    # ax2.plot(env.times, env.fluor_bg - env.fluor_bg.min(), label='fluorescence background')

    # plot(env.age_times, env.counts)
    # ax2.plot(env.age_times, env.counts)
    # print(repr(env.cell_times), repr(env.cell_counts))
    # ax2.plot(env.cell_times, env.cell_counts)
    legend()


if __name__ == '__main__':
    run_the_processor(parser, process, ploting, desc="Fluorescence")