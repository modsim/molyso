# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

from .. import __citation__

import argparse
import sys
import os
import numpy
import itertools
import codecs
import json
import multiprocessing
import datetime
import traceback
import logging

from .. import Debug, TunableManager

from ..generic.etc import parse_range, correct_windows_signal_handlers, debug_init, QuickTableDumper, \
    silent_progress_bar, fancy_progress_bar, prettify_numpy_array, bits_to_numpy_type

from ..generic.etc import Sqlite3Cache as Cache

from ..imageio.imagestack import MultiImageStack
from ..imageio.imagestack_ometiff import OMETiffStack
from .image import Image
from .fluorescence import FluorescentImage

from .tracking import TrackedPosition, analyze_tracking, plot_timeline, tracker_to_cell_list

from .highlevel_interactive_viewer import interactive_main
from .highlevel_interactive_ground_truth import interactive_ground_truth_main

OMETiffStack = OMETiffStack

class Hooks:
    main = []

    def __init__(self):
        pass


def banner():
    return r"""
     \   /\  /\  /                             -------------------------
      | | |O| | |    molyso                    Developed  2013 - 2015 by
      | | | | |O|                              Christian   C.  Sachs  at
      |O| |O| |O|    MOther    machine         ModSim / Microscale Group
      \_/ \_/ \_/    anaLYsis SOftware         Research  Center  Juelich
    --------------------------------------------------------------------
    If you use this software in a publication, cite our paper:

    %(citation)s

    --------------------------------------------------------------------
    """ % {'citation': __citation__}


def create_argparser():
    argparser = argparse.ArgumentParser(description="molyso: MOther machine anaLYsis SOftware")

    def _error(message=''):
        print(banner())
        argparser.print_help()
        sys.stderr.write("%serror: %s%s" % (os.linesep, message, os.linesep,))
        sys.exit(1)

    argparser.error = _error

    argparser.add_argument('input', metavar='input', type=str, help="input file")
    argparser.add_argument('-m', '--module', dest='modules', type=str, default=None, action='append')
    argparser.add_argument('-p', '--process', dest='process', default=False, action='store_true')
    argparser.add_argument('-gt', '--ground-truth', dest='ground_truth', type=str, default=None)
    argparser.add_argument('-ct', '--cache-token', dest='cache_token', type=str, default=None)
    argparser.add_argument('-tp', '--timepoints', dest='timepoints', default='0-', type=str)
    argparser.add_argument('-mp', '--multipoints', dest='multipoints', default='0-', type=str)
    argparser.add_argument('-o', '--table-output', dest='table_output', type=str, default=None)
    argparser.add_argument('-ot', '--output-tracking', dest='tracking_output', type=str, default=None)
    argparser.add_argument('-nb', '--no-banner', dest='nb', default=False, action='store_true')
    argparser.add_argument('-cpu', '--cpus', dest='mp', default=-1, type=int)
    argparser.add_argument('-debug', '--debug', dest='debug', default=False, action='store_true')
    argparser.add_argument('-nci', '--no-channel-images', dest='keepchan', default=True, action='store_false')
    argparser.add_argument('-cfi', '--channel-fluorescence-images', dest='keepfluorchan',
                           default=False, action='store_true')
    argparser.add_argument('-ccb', '--channel-image-channel-bits', dest='channel_bits',
                           default=numpy.uint8, type=bits_to_numpy_type)
    argparser.add_argument('-cfb', '--channel-image-fluorescence-bits', dest='channel_fluorescence_bits',
                           default=numpy.float32, type=bits_to_numpy_type)
    argparser.add_argument('-q', '--quiet', dest='quiet', default=False, action='store_true')
    argparser.add_argument('-nc', '--no-cache', dest='ignorecache', default='nothing',
                           const='everything', type=str, nargs='?')
    argparser.add_argument('-nt', '--no-tracking', dest='no_tracking', default=False, action='store_true')
    argparser.add_argument('-t', '--tunables', dest='tunables', type=str, default=None)
    argparser.add_argument('-pt', '--print-tunables', dest='print_tunables', default=False, action='store_true')
    argparser.add_argument('-rt', '--read-tunables', dest='read_tunables', type=str, default=None)
    argparser.add_argument('-wt', '--write-tunables', dest='write_tunables', type=str, default=None)

    return argparser


def setup_image(i, local_ims, t, pos):
    image = local_ims.get_image(t=t, pos=pos, channel=local_ims.__class__.Phase_Contrast, float=True)

    i.setup_image(image)

    if getattr(i, 'setup_add_fluorescence', False) and local_ims.get_meta('channels') > 1:
        for channel in local_ims.get_meta('fluorescenceChannels'):
            fimg = local_ims.get_image(t=t, pos=pos, channel=channel, float=True)
            i.setup_add_fluorescence(fimg)

    i.multipoint = int(pos)
    i.timepoint_num = int(t)

    i.timepoint = local_ims.get_meta('time', t=t, pos=pos)

    i.calibration_px_to_mu = local_ims.get_meta('calibration', t=t, pos=pos)

    i.metadata['x'], i.metadata['y'], i.metadata['z'] = local_ims.get_meta('position', t=t, pos=pos)

    i.metadata['time'] = i.timepoint
    i.metadata['timepoint'] = i.timepoint_num
    i.metadata['multipoint'] = i.multipoint

    i.metadata['calibration_px_to_mu'] = i.calibration_px_to_mu

    i.metadata['tag'] = ''
    i.metadata['tag_number'] = 0


# globals

ims = None

first_frame_cache = {}
first_to_look_at = 0


def check_or_get_first_frame(pos):
    global first_frame_cache

    if pos in first_frame_cache:
        return first_frame_cache[pos]
    else:
        if ims.get_meta('channels') > 1:
            image = FluorescentImage()
        else:
            image = Image()

        setup_image(image, ims, first_to_look_at, pos)

        image.autorotate()
        image.autoregister(image)

        first_frame_cache[pos] = image

        return image


def processing_frame(args, t, pos):
    first = check_or_get_first_frame(pos)

    if ims.get_meta('channels') > 1:
        image = FluorescentImage()
    else:
        image = Image()

    setup_image(image, ims, t, pos)

    Debug.set_context(t=t, pos=pos)

    image.keep_channel_image = args.keepchan
    image.pack_channel_image = args.channel_bits

    if type(image) == FluorescentImage:
        image.keep_fluorescences_image = args.keepfluorchan
        image.pack_fluorescences_image = args.channel_fluorescence_bits

    image.autorotate()
    image.autoregister(first)

    image.find_channels()
    image.find_cells_in_channels()

    image.clean()

    image.flatten()

    return image


def setup_modules(modules):
    import importlib
    for module in modules:
        try:
            importlib.import_module("molyso_%s" % module)
        except ImportError:
            try:
                importlib.import_module(module)
            except ImportError:
                print("WARNING: Could not load either module molyso_%s or %s!" % (module, module,))


def processing_setup(args):
    global ims
    global first_to_look_at

    first_to_look_at = args.timepoints[0]

    if args.modules:
        setup_modules(args.modules)

    if ims is None:
        ims = MultiImageStack.open(args.input)

    correct_windows_signal_handlers()

    try:
        import cv2
    except ImportError:
        pass


def main():
    global ims

    argparser = create_argparser()

    args = argparser.parse_args()

    if args.ground_truth:
        args.process = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)s %(levelname)s %(message)s")
    #log = logging.getLogger('molyso')

    def print_info(*inner_args):
        if not args.quiet:
            print(*inner_args)
            #log.info(str(*inner_args))

    def print_warning(*inner_args):
        print(*inner_args, file=sys.stderr)
        #log.warn(str(*inner_args))

    if args.quiet:  # silence the progress bar filter
        progress_bar = silent_progress_bar
    else:
        progress_bar = fancy_progress_bar

    if not args.nb:
        print_info(banner())

    print_info("molyso started at %s" % (str(datetime.datetime.now())))

    if args.modules:
        setup_modules(args.modules)

    # load tunables
    if args.read_tunables:
        with open(args.read_tunables, 'r') as tunable_file:
            tunables = json.load(tunable_file)
            print_info("Loaded tunable file \"%(filename)s\" with data: %(data)s" %
                       {'filename': args.read_tunables, 'data': repr(tunables)})
            TunableManager.load_tunables(tunables)

    if args.tunables:
        tunables = json.loads(args.tunables)
        print_info("Loaded command line tunables: %(data)s" % {'data': repr(tunables)})
        TunableManager.load_tunables(tunables)

    if args.print_tunables:
        TunableManager.set_printing(True)

    if sys.maxsize <= 2 ** 32:
        print_warning("Warning, running on a 32 bit Python interpreter! This is most likely not what you want,"
                      "and it will significantly reduce functionality!")

    for hook in Hooks.main:
        hook(args)

    if not args.process:
        return interactive_main(args)

    try:
        if not args.ground_truth:
            import matplotlib

            matplotlib.use('PDF')
    except ImportError:
        if args.debug:
            print_warning("matplotlib could not be imported. Debugging was disabled.")
            args.debug = False

    if args.debug:
        debug_init()
        if args.mp != 0:
            print_warning("Debugging enabled, concurrent processing disabled!")
            args.mp = 0

    if args.ground_truth:
        args.debug = False

    Cache.printer = print_info
    cache = Cache(args.input, ignore_cache=args.ignorecache, cache_token=args.cache_token)

    if 'tracking' not in cache:

        if 'imageanalysis' in cache:
            results = cache['imageanalysis']
        else:
            ims = MultiImageStack.open(args.input)

            args.multipoints = parse_range(args.multipoints, maximum=ims.get_meta('multipoints'))
            args.timepoints = parse_range(args.timepoints, maximum=ims.get_meta('timepoints'))

            positions_to_process = args.multipoints
            timepoints_to_process = args.timepoints

            print_info("Beginning Processing:")
            print_info(prettify_numpy_array(positions_to_process,  "Positions : "))
            print_info(prettify_numpy_array(timepoints_to_process, "Timepoints: "))

            results = {pos: {} for pos in positions_to_process}

            total = len(timepoints_to_process) * len(positions_to_process)

            if args.mp < 0:
                args.mp = multiprocessing.cpu_count()

            print_info("Performing image analysis ...")

            to_process = list(itertools.product(timepoints_to_process, positions_to_process))

            if args.mp == 0:
                processing_setup(args)

                for t, pos in progress_bar(to_process):
                    results[pos][t] = processing_frame(args, t, pos)
            else:
                print_info("... parallel with %(process_count)d processes" % {'process_count': args.mp})

                pool = multiprocessing.Pool(args.mp, processing_setup, [args])

                workerstates = []

                for t, pos in to_process:
                    workerstates.append((t, pos, pool.apply_async(processing_frame, (args, t, pos))))

                pool.close()

                progressbar_states = progress_bar(range(total))

                while len(workerstates) > 0:
                    for i, (t, pos, state) in reversed(list(enumerate(workerstates))):
                        if state.ready():
                            try:
                                results[pos][t] = state.get()
                            except Exception as e:
                                print("ERROR: Exception occured at pos: %(pos)d, time %(time)d: %(e)s\n%(traceback)s" %
                                      {'pos': pos, 'time': t, 'e': str(e), 'traceback': traceback.format_exc()})

                            del workerstates[i]
                            next(progressbar_states)

                try:
                    # to output the progress bar, the iterator must be pushed beyond its end
                    next(progressbar_states)
                except StopIteration:
                    pass

            cache['imageanalysis'] = results

    ####################################################################################################################

    if not args.no_tracking:

        if 'tracking' in cache:
            # noinspection PyUnboundLocalVariable
            results = None # free up some ram?
            tracked_results = cache['tracking']
        else:

            tracked_results = {}

            print_info()

            print_info("Set-up for tracking ...")

            pi = progress_bar(range(sum([len(l) - 1 if len(l) > 0 else 0 for l in results.values()]) - 1))

            for pos, times in results.items():
                tracked_position = TrackedPosition()
                tracked_position.set_times(times)
                tracked_position.align_channels(progress_indicator=pi)
                tracked_position.remove_empty_channels()
                tracked_position.guess_channel_orientation()
                tracked_results[pos] = tracked_position

            print_info()

            print_info("Performing tracking ...")

            pi = progress_bar(range(sum([tp.get_tracking_work_size() for tp in tracked_results.values()]) - 1))

            for pos, times in results.items():
                tracked_position = tracked_results[pos]
                tracked_position.perform_tracking(progress_indicator=pi)
                tracked_position.remove_empty_channels_post_tracking()

            cache['tracking'] = tracked_results

        # ( Diversion into ground truth processing, if applicable )

        if args.ground_truth:
            interactive_ground_truth_main(args, tracked_results)
            return

        # ( Output of textual results: )################################################################################

        def each_pos_k_tracking_tracker_channels_in_results(inner_tracked_results):
            for pos, tracking in inner_tracked_results.items():
                for inner_k in sorted(tracking.tracker_mapping.keys()):
                    tracker = tracking.tracker_mapping[inner_k]
                    channels = tracking.channel_accumulator[inner_k]
                    yield pos, inner_k, tracking, tracker, channels

        if args.table_output is None:
            recipient = sys.stdout
        else:
            recipient = codecs.open(args.table_output, 'wb+', 'utf-8')

        print_info()
        print_info("Outputting tabular data ...")

        flat_results = list(each_pos_k_tracking_tracker_channels_in_results(tracked_results))

        try:
            table_dumper = QuickTableDumper(recipient=recipient)

            iterable = progress_bar(flat_results) if recipient is not sys.stdout else silent_progress_bar(flat_results)

            for pos, k, tracking, tracker, channels in iterable:
                analyze_tracking(tracker_to_cell_list(tracker), lambda x: table_dumper.add(x))

        finally:
            if recipient is not sys.stdout:
                recipient.close()

        # ( Output of graphical tracking results: )#####################################################################

        if args.tracking_output is not None:

            try:
                import matplotlib
                import matplotlib.pylab
            except ImportError:
                print_warning("Tracking output enabled but matplotlib not found! Cannot proceed.")
                print_warning("Please install matplotlib ...")
                raise

            print_info("Outputting graphical tracking data ...")

            figdir = os.path.abspath(args.tracking_output)

            if not os.path.isdir(figdir):
                os.mkdir(figdir)

            for pos, k, tracking, tracker, channels in progress_bar(flat_results):
                plot_timeline(matplotlib.pylab, channels, tracker_to_cell_list(tracker),
                              figure_presetup=
                              lambda p: p.title("Channel #%02d (average cells = %.2f)" % (k, tracker.average_cells)),
                              figure_finished=
                              lambda p: p.savefig("%(dir)s/tracking_pt_%(mp)02d_chan_%(k)02d.pdf" %
                                                  {'dir': figdir, 'mp': pos, 'k': k}),
                              show_images=True, show_overlay=True)

    # ( Post-Tracking: Just write some tunables, if desired )###########################################################

    if args.write_tunables:
        print_info()
        if os.path.isfile(args.write_tunables):
            print_warning("Tunable output will not overwrite existing files!")
            print_warning("NOT outputing tunables.")
        else:
            fname = os.path.abspath(args.write_tunables)
            print_info("Writing tunables to \"%(fname)s\"" % {'fname': fname})
            with codecs.open(fname, 'wb+', 'utf-8') as fp:
                json.dump(TunableManager.get_defaults(), fp, indent=4, sort_keys=True)

    print_info("molyso finished at %s" % (str(datetime.datetime.now())))

