# -*- coding: utf-8 -*-
"""
See README/and documentation for end user information. See :doc:`license` for license information.
(Or string :py:const:`molyso.__license__`).

A short starting point to understanding *molyso*'s internal structure:

The :py:mod:`molyso.mm` module contains the Mother Machine specific code,
as well as the main function (see highlevel.py).
Analysis code is basically split into two levels: an OOP representation of the data analyzed, as well as some
core functions (functional) which perform individual processing step.

*E.g.*, a :py:class:`molyso.mm.channel_detection.Channel` class calls a
:py:func:`molyso.mm.cell_detection.find_cells_in_channel` function,
which returns mere numbers, of which the class constructs :py:class:`molyso.mm.cell_detection.Cell` objects.

:py:mod:`molyso.generic` contains a mix of library functions necessary to perform these tasks,
coarsely these can be separated into signal processing etc. functionality, and general utility functions.

molyso.debugging contains the DebugPlot class, a thin abstraction layer over matplotlib which allows for conditional,
context manager based plot generation.

molyso.imageio contains the image reading functionality. The MultiImageStack is a base class and factory for opening,
multi dimensional images. molyso contains reading code to open plain TIFF and OME-TIFF files, using tifffile.py.
MultiImageStack acts as a implementation registry as well, if other formats should be supported, a reader subclass
has to be generated, and registered at MultiImageStack. It then can automatically be used by molyso to open the format.

As mentioned earlier, the main function is within highlevel.py. It can divert program flow to two additional modes,
which overtake if called: ground truth or interactive mode.

Processing itself is handled within highlevel.py.

Doctests can be called by calling the molyso.test module.
"""
from __future__ import division, unicode_literals, print_function

from .generic.tunable import tunable, TunableManager

__license__ = """
Copyright (c) 2013-2021 Christian C. Sachs, Forschungszentrum Jülich GmbH.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = "Christian C. Sachs"

__version__ = '1.0.6'

__citation__ = \
    """
    Sachs CC, Grünberger A, Helfrich S, Probst C, Wiechert W, Kohlheyer D, Nöh K (2016)
    Image-Based Single Cell Profiling:
    High-Throughput Processing of Mother Machine Experiments.
    PLoS ONE 11(9): e0163453. doi: 10.1371/journal.pone.0163453
"""
