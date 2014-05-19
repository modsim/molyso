# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

__citation__ = "molyso: A software for mother machine analysis, working title. Sachs et al."

import argparse
import sys
import os
import multiprocessing

from ..generic.etc import parse_range, correct_windows_signal_handlers, debug_init

from ..imageio.imagestack import MultiImageStack
from ..imageio.imagestack_ometiff import OMETiffStack
from .image import Image

from .tracking import track_complete_channel_timeline, analyze_tracking

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


def create_argparser():
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

    return argparser


def setup_image(i, local_ims, t, pos):
    img = local_ims.get_image(t=t, pos=pos, channel=local_ims.__class__.Phase_Contrast, float=True)

    i.setup_image(img)

    if local_ims.get_meta("channels") > 1:
        fimg = local_ims.get_image(t=t, pos=pos, channel=local_ims.__class__.Fluorescence, float=True)

        i.setup_fluorescence(fimg)

    i.multipoint = int(pos)
    i.timepoint_num = int(t)

    i.timepoint = local_ims.get_meta("time", t=t, pos=pos)

    i.calibration_px_to_mu = local_ims.get_meta("calibration", t=t, pos=pos)

    i.metadata["x"], i.metadata["y"], i.metadata["z"] = local_ims.get_meta("position", t=t, pos=pos)

    i.metadata["time"] = i.timepoint
    i.metadata["timepoint"] = i.timepoint_num
    i.metadata["multipoint"] = i.multipoint

    i.metadata["calibration_px_to_mu"] = i.calibration_px_to_mu

    i.metadata["tag"] = ""
    i.metadata["tag_number"] = 0


# globals

ims = None


def processing_frame(t, pos):
    image = Image()

    setup_image(image, ims, t, pos)

    image.keep_channel_image = True

    image.autorotate()
    image.find_channels()
    image.find_cells_in_channels()

    image.clean()

    return image


def processing_setup(filename):
    global ims
    if ims is None:
        ims = MultiImageStack.open(filename)

    correct_windows_signal_handlers()


def main():
    global ims

    argparser = create_argparser()

    args = argparser.parse_args()

    if not args.nb:
        print(banner())

    try:
        import matplotlib

        matplotlib.use("TkAgg")
    except ImportError:
        if args.debug:
            print("matplotlib could not be imported. Debugging was disabled.")
            args.debug = False

    if args.debug:
        debug_init()
        if args.mp != 0:
            print("Debugging enabled, concurrent processing disabled!")
            args.mp = 0

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

    results = {pos: {} for pos in positions_to_process}

    total = len(timepoints_to_process) * len(positions_to_process)
    processed = 0

    if args.mp == 0:
        processing_setup(args.input)

        for t in timepoints_to_process:
            for pos in positions_to_process:
                results[pos][t] = processing_frame(t, pos)
                processed += 1
                print("Processed %d/%d" % (processed, total))
    else:
        if args.mp < 0:
            args.mp = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(args.mp, processing_setup, [args.input])

        workerstates = []

        for t in timepoints_to_process:
            for pos in positions_to_process:
                workerstates.append((t, pos, pool.apply_async(processing_frame, (t, pos))))

        while len(workerstates) > 0:
            for i, (t, pos, state) in reversed(list(enumerate(workerstates))):
                if state.ready():
                    results[pos][t] = state.get()
                    del workerstates[i]
                    processed += 1
                    print("Processed %d/%d" % (processed, total))

    tracked_results = {}

    for pos, times in results.items():
        tracked_results[pos] = track_complete_channel_timeline(times)

    analyze_tracking(tracked_results)


