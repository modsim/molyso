# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function

from .debugging.debug import Debug
from .debugging.debugplot import DebugPlot
from .generic.tunable import tunable, TunableManager

__version__ = '0.1'

__citation__ = \
    """Image Analysis of Mother Machine Microfluidic Experiments:
    A Software for Unsupervised High-Throughput Processing.
    Sachs et al. 2015."""