# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from helpers import *


def parser(parser):
    pass


def process(data, env, output):
    division_data = data[data.about_to_divide == 1]
    output.ages = env.ages = numpy.log(2) / env.process(division_data.division_age)
    output.times = env.times = env.process(division_data.timepoint)


def ploting(env):
    xlabel('time [s]')
    ylabel('growth rate [min $^{-1}$]')
    plot(env.times, env.ages, label='µ')
    legend()


if __name__ == '__main__':
    run_the_processor(parser, process, ploting, desc="Growth rate utility")