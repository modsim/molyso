# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

__citation__ = "molyso: A software for mother machine analysis, working title. Sachs et al."

import argparse
import sys
import os

from ..generic.etc import parse_range

from ..imageio.imagestack import MultiImageStack
from ..imageio.imagestack_ometiff import OMETiffStack

OMETiffStack = OMETiffStack


def banner():
    return """
      \   /\  /\  /                             -------------------------
       | | |O| | |    molyso                    Developed  2013 - 2014 by
       | | | | |O|                              Christian   C.  Sachs  at
       |O| |O| |O|    MOther    machine         Microscale Bioengineering
       \_/ \_/ \_/    anaLYsis SOftware         Research  Center  Juelich
     --------------------------------------------------------------------
     If you use this software in a publication, please cite our paper:

     %(citation)s
     --------------------------------------------------------------------
    """ % {"citation": __citation__}


ims = None


def main():
    global ims

    argparser = argparse.ArgumentParser(description="molyso: MOther machine anaLYsis SOftware")

    def _error(message=""):
        print(banner())
        argparser.print_help()
        sys.stderr.write("%serror: %s%s" % (os.linesep, message, os.linesep,))
        sys.exit(1)

    argparser.error = _error

    argparser.add_argument("input", metavar="input", type=str, help="input file")
    argparser.add_argument("-tp", "--timepoints", dest="timepoints", default=[0, float("inf")], type=parse_range)
    argparser.add_argument("-mp", "--multipoints", dest="multipoints", default=[0, float("inf")], type=parse_range)
    argparser.add_argument("-nb", "--no-banner", dest="nb", default=False, action="store_true")
    argparser.add_argument("-cpu", "--cpus", dest="mp", default=-1, type=int)
    argparser.add_argument("-debug", "--debug", dest="debug", default=False, action="store_true")
    argparser.add_argument("-nc", "--no-cache", dest="nocache", default=False, action="store_true")

    args = argparser.parse_args()

    if not args.nb:
        print(banner())

    ims = MultiImageStack.open(args.input)

    positions_to_process = args.multipoints

    if positions_to_process[-1] == float("Inf"):
        f = positions_to_process[-2]
        del positions_to_process[-2:-1]
        positions_to_process += range(f, ims.get_meta("multipoints"))

    positions_to_process = [p for p in positions_to_process if 0 <= p <= ims.get_meta("multipoints")]

    timepoints_to_process = args.timepoints
    if timepoints_to_process[-1] == float("Inf"):
        f = timepoints_to_process[-2]
        del timepoints_to_process[-2:-1]
        timepoints_to_process += range(f, ims.get_meta("timepoints"))

    timepoints_to_process = [t for t in timepoints_to_process if 0 <= t <= ims.get_meta("timepoints")]

    print("Processing:")
    print("Positions : [%s]" % (", ".join(map(str, positions_to_process))))
    print("Timepoints: [%s]" % (", ".join(map(str, timepoints_to_process))))

