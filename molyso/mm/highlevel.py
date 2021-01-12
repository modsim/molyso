# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

from .. import __citation__

import argparse
import sys
import os
import itertools
import codecs
import json
import multiprocessing
import traceback
import logging
import platform

import numpy as np

from pilyso_io.imagestack import ImageStack, Dimensions, FloatFilter
from pilyso_io.imagestack.readers import *

from pilyso_io.imagestack.util import parse_range, prettify_numpy_array

from ..debugging import DebugPlot
from ..debugging.debugplot import inject_poly_drawing_helper
from ..debugging.callserialization import CallSerialization
from ..generic.tunable import TunableManager, tunable

from ..generic.etc import correct_windows_signal_handlers, debug_init, QuickTableDumper, \
    silent_progress_bar, fancy_progress_bar, bits_to_numpy_type

from ..generic.etc import Sqlite3Cache as Cache

from .image import Image
from .fluorescence import FluorescentImage

from .tracking import TrackedPosition, analyze_tracking, plot_timeline, tracker_to_cell_list

from .highlevel_interactive_viewer import interactive_main
from .highlevel_interactive_ground_truth import interactive_ground_truth_main
from .highlevel_interactive_advanced_ground_truth import interactive_advanced_ground_truth_main


def setup_matplotlib(throw=True, interactive=True, log=None):
    try:
        import matplotlib
    except ImportError:
        if throw:
            raise RuntimeError("matplotlib could not be imported. The requested functionality requires matplotlib.")
        else:
            if log:
                log.warning("matplotlib could not be imported.")
            return False

    if interactive:
        if platform.system() == 'Darwin':
            backend = 'MacOSX'
        else:
            backend = 'TkAgg'
    else:
        backend = 'PDF'

    matplotlib.use(backend)

    return True


class Hooks(object):
    """
    Hooks class, static object merely existing to collect hook registrations.
    """
    main = []

    def __init__(self):
        pass


def banner():
    """
    Formats the banner.
    :return: banner
    """
    return r"""
     \   /\  /\  /                             -------------------------
      | | |O| | |    molyso                    Developed  2013 - 2021 by
      | | | | |O|                              Christian   C.  Sachs  at
      |O| |O| |O|    MOther    machine         ModSim / Microscale Group
      \_/ \_/ \_/    anaLYsis SOftware         Research  Center  Juelich
    --------------------------------------------------------------------
    If you use this software in a publication, cite our paper:

    %(citation)s

    --------------------------------------------------------------------
    """ % {'citation': __citation__}


def create_argparser():
    """


    :return:
    """
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
    argparser.add_argument('-agt', '--advanced-ground-truth', dest='advanced_ground_truth', type=str, default=None)
    argparser.add_argument('-ct', '--cache-token', dest='cache_token', type=str, default=None)
    argparser.add_argument('-tp', '--timepoints', dest='timepoints', default='0-', type=str)
    argparser.add_argument('-mp', '--multipoints', dest='multipoints', default='0-', type=str)
    argparser.add_argument('-o', '--table-output', dest='table_output', type=str, default=None)
    argparser.add_argument('--meta', '--meta', dest='meta', type=str, default=None)
    argparser.add_argument('-ot', '--output-tracking', dest='tracking_output', type=str, default=None)
    argparser.add_argument('-otf', '--output-tracking-format', dest='tracking_output_format',
                           type=list, default=None, action='append')
    argparser.add_argument('-nb', '--no-banner', dest='nb', default=False, action='store_true')
    argparser.add_argument('-cpu', '--cpus', dest='mp', default=-1, type=int)
    argparser.add_argument('-debug', '--debug', dest='debug', default=False, action='store_true')
    argparser.add_argument('-do', '--detect-once', dest='detect_once', default=False, action='store_true')
    argparser.add_argument('-nci', '--no-channel-images', dest='keepchan', default=True, action='store_false')
    argparser.add_argument('-cfi', '--channel-fluorescence-images', dest='keepfluorchan',
                           default=False, action='store_true')
    argparser.add_argument('-ccb', '--channel-image-channel-bits', dest='channel_bits',
                           default=np.uint8, type=bits_to_numpy_type)
    argparser.add_argument('-cfb', '--channel-image-fluorescence-bits', dest='channel_fluorescence_bits',
                           default=np.float32, type=bits_to_numpy_type)
    argparser.add_argument('-q', '--quiet', dest='quiet', default=False, action='store_true')
    argparser.add_argument('-nc', '--no-cache', dest='ignorecache', default='nothing',
                           const='everything', type=str, nargs='?')
    argparser.add_argument('-nt', '--no-tracking', dest='no_tracking', default=False, action='store_true')
    argparser.add_argument('-t', '--tunables', dest='tunables', type=str, default=None)
    argparser.add_argument('-s', '--set-tunable', dest='tunable_list',
                           type=list, nargs=2, default=None, action='append')
    argparser.add_argument('-pt', '--print-tunables', dest='print_tunables', default=False, action='store_true')
    argparser.add_argument('-rt', '--read-tunables', dest='read_tunables', type=str, default=None)
    argparser.add_argument('-wt', '--write-tunables', dest='write_tunables', type=str, default=None)

    return argparser


def setup_image(i, local_ims, t, pos):
    """

    :param i:
    :param local_ims:
    :param t:
    :param pos:
    """

    image = local_ims[pos, t, 0]

    left, right, top, bottom = (
        tunable('preprocess.crop.left', 0, description="Cropping, left border."),
        tunable('preprocess.crop.right', 0, description="Cropping, right border."),
        tunable('preprocess.crop.top', 0, description="Cropping, top border."),
        tunable('preprocess.crop.bottom', 0, description="Cropping, bottom border."))

    right = None if right == 0 else -right
    bottom = None if bottom == 0 else -bottom

    i.setup_image(image[top:bottom, left:right])

    if getattr(i, 'setup_add_fluorescence', False) and local_ims.size[Dimensions.Channel] > 1:
        for channel in range(1, local_ims.size[Dimensions.Channel]):
            fimg = local_ims[pos, t, channel]
            i.setup_add_fluorescence(fimg[top:bottom, left:right])

    i.multipoint = int(pos)
    i.timepoint_num = int(t)

    meta = local_ims.meta[pos, t, 0]

    i.timepoint = meta.time

    i.calibration_px_to_mu = meta.calibration

    i.metadata['x'], i.metadata['y'], i.metadata['z'] = meta.position

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


def check_or_get_first_frame(pos, args):
    """

    :param args:
    :param pos:
    :return:
    """
    global first_frame_cache

    if pos in first_frame_cache:
        return first_frame_cache[pos]
    else:
        if ims.size[Dimensions.Channel] > 1:
            image = FluorescentImage()
        else:
            image = Image()

        setup_image(image, ims, first_to_look_at, pos)

        image.autorotate()
        image.autoregistration(image)

        if args.detect_once:
            from .channel_detection import find_channels

            def _find_channels(im):
                image._find_channels_positions = find_channels(im)
                # noinspection PyProtectedMember
                return image._find_channels_positions

            image.find_channels_function = _find_channels

        image.find_channels()

        if args.detect_once:
            delattr(image, 'find_channels_function')

        first_frame_cache[pos] = image

        return image


def processing_frame(args, t, pos, clean=True):
    """

    :param args:
    :param t:
    :param pos:
    :param clean:
    :return:
    """

    first = check_or_get_first_frame(pos, args)

    if ims.size[Dimensions.Channel] > 1:
        image = FluorescentImage()
    else:
        image = Image()

    setup_image(image, ims, t, pos)

    DebugPlot.set_context(t=t, pos=pos)

    image.keep_channel_image = args.keepchan
    image.pack_channel_image = args.channel_bits

    if type(image) == FluorescentImage:
        image.keep_fluorescences_image = args.keepfluorchan
        image.pack_fluorescences_image = args.channel_fluorescence_bits

    if args.detect_once:
        image.angle = first.angle

    image.autorotate()
    image.autoregistration(first)

    if args.detect_once:
        from ..generic.registration import shift_image

        image.image = shift_image(image.image, image.shift)

        if type(image) == FluorescentImage:
            for n in range(len(image.image_fluorescences)):
                image.image_fluorescences[n] = shift_image(image.image_fluorescences[n], image.shift)

        image.shift = [0.0, 0.0]

        # noinspection PyProtectedMember
        def _find_channels_function(im):
            return first._find_channels_positions

        image.find_channels_function = _find_channels_function

    image.find_channels()

    if args.detect_once:
        delattr(image, 'find_channels_function')

    image.find_cells_in_channels()

    if clean:

        image.clean()

        image.flatten()

    return image


def setup_tunables(args, log=None):
    # load tunables
    if args.read_tunables:
        with open(args.read_tunables, 'r') as tunable_file:
            tunables = json.load(tunable_file)
            if log:
                log.info("Loaded tunable file \"%(filename)s\" with data: %(data)s" %
                         {'filename': args.read_tunables, 'data': repr(tunables)})
            TunableManager.load_tunables(tunables)

    if args.tunables:
        tunables = json.loads(args.tunables)
        if log:
            log.info("Loaded command line tunables: %(data)s" % {'data': repr(tunables)})
        TunableManager.load_tunables(tunables)

    if args.tunable_list:
        tunables = {}
        for k, v in args.tunable_list:
            tunables[''.join(k)] = ''.join(v)
        if log:
            log.info("Loaded command line tunables: %(data)s" % {'data': repr(tunables)})
        TunableManager.load_tunables(tunables)


def setup_modules(modules):
    """

    :param modules:
    """
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
    """

    :param args:
    """
    global ims
    global first_to_look_at

    if args.modules:
        setup_modules(args.modules)

    setup_tunables(args)

    if ims is None:
        ims = ImageStack(args.input).view(Dimensions.PositionXY, Dimensions.Time, Dimensions.Channel).filter(FloatFilter)

    if isinstance(args.multipoints, str):
        args.multipoints = parse_range(args.multipoints, maximum=ims.size[Dimensions.PositionXY])

    if isinstance(args.timepoints, str):
        args.timepoints = parse_range(args.timepoints, maximum=ims.size[Dimensions.Time])

    first_to_look_at = args.timepoints[0]

    correct_windows_signal_handlers()

    # OpenCV takes a moment and often nags about missing libraries, let's load it at the
    # beginning so the rest runs smoothly
    try:
        # noinspection PyPackageRequirements,PyUnresolvedReferences
        import cv2
    except ImportError:
        pass


def main():
    """


    :return: :raise:
    """
    global ims

    argparser = create_argparser()

    args = argparser.parse_args()

    if args.ground_truth or args.advanced_ground_truth:
        args.process = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s.%(msecs)03d %(name)s %(levelname)s %(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

    if args.quiet:  # silence the progress bar filter
        progress_bar = silent_progress_bar
        log.setLevel(logging.WARN)
    else:
        progress_bar = fancy_progress_bar

    if not args.nb:
        log.info(banner())

    log.info("Started analysis.")

    if args.modules:
        setup_modules(args.modules)

    setup_tunables(args, log)

    if sys.maxsize <= 2 ** 32:
        log.warning("Warning, running on a 32 bit Python interpreter! This is most likely not what you want,"
                    "and it will significantly reduce functionality!")

    for hook in Hooks.main:
        hook(args)

    if not args.process:
        setup_matplotlib(log=log)

        return interactive_main(args)

    try:
        if not args.ground_truth:
            # noinspection PyUnresolvedReferences
            import matplotlib

            matplotlib.use('PDF')
    except ImportError:
        if args.debug:
            args.debug = setup_matplotlib(throw=False, interactive=False, log=log)

    if args.debug:
        debug_init()
        if args.mp != 0:
            log.warning("Debugging enabled, concurrent processing disabled!")
            args.mp = 0

    if args.ground_truth or args.advanced_ground_truth:
        args.debug = False

    cache = Cache(args.input, ignore_cache=args.ignorecache, cache_token=args.cache_token)

    if 'tracking' not in cache:

        if 'imageanalysis' in cache:
            results = cache['imageanalysis']
        else:
            ims = ImageStack(args.input).view(Dimensions.PositionXY, Dimensions.Time, Dimensions.Channel).filter(FloatFilter)

            args.multipoints = parse_range(args.multipoints, maximum=ims.size[Dimensions.PositionXY])
            args.timepoints = parse_range(args.timepoints, maximum=ims.size[Dimensions.Time])

            positions_to_process = args.multipoints
            timepoints_to_process = args.timepoints

            log.info("Beginning Processing:")
            dummy = " " * len("XXXX-XX-XX XX:XX:XX.XXX molyso INFO ")
            log.info(prettify_numpy_array(positions_to_process,  dummy + "Positions : ").strip())
            log.info(prettify_numpy_array(timepoints_to_process, dummy + "Timepoints: ").strip())

            results = {pos: {} for pos in positions_to_process}

            total = len(timepoints_to_process) * len(positions_to_process)

            if args.mp < 0:
                args.mp = multiprocessing.cpu_count()

            log.info("Performing image analysis ...")

            to_process = list(itertools.product(timepoints_to_process, positions_to_process))

            if args.mp == 0:
                processing_setup(args)

                for t, pos in progress_bar(to_process):
                    results[pos][t] = processing_frame(args, t, pos)
            else:

                # ims = None

                log.info("... parallel with %(process_count)d processes" % {'process_count': args.mp})

                pool = multiprocessing.Pool(args.mp, processing_setup, [args])

                worker_states = []

                for t, pos in to_process:
                    worker_states.append((t, pos, pool.apply_async(processing_frame, (args, t, pos))))

                pool.close()

                progressbar_states = progress_bar(range(total))

                while len(worker_states) > 0:
                    for i, (t, pos, state) in reversed(list(enumerate(worker_states))):
                        if state.ready():
                            try:
                                results[pos][t] = state.get()
                            except Exception as e:
                                log.exception(
                                    "ERROR: Exception occurred at pos: %(pos)d, time %(time)d: %(e)s\n%(traceback)s" %
                                    {'pos': pos, 'time': t, 'e': str(e), 'traceback': traceback.format_exc()}
                                )

                            del worker_states[i]
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
            results = None
            del results  # free up some ram?
            tracked_results = cache['tracking']
        else:

            tracked_results = {}

            log.info("Set-up for tracking ...")

            # noinspection PyUnboundLocalVariable
            pi = progress_bar(range(sum([len(l) - 1 if len(l) > 0 else 0 for l in results.values()]) - 1))

            for pos, times in results.items():
                tracked_position = TrackedPosition()
                tracked_position.set_times(times)
                tracked_position.align_channels(progress_indicator=pi)
                tracked_position.remove_empty_channels()
                tracked_position.guess_channel_orientation()
                tracked_results[pos] = tracked_position

            log.info("Performing tracking ...")

            pi = progress_bar(range(sum([tp.get_tracking_work_size() for tp in tracked_results.values()]) - 1))

            for pos, times in results.items():
                tracked_position = tracked_results[pos]
                tracked_position.perform_tracking(progress_indicator=pi)
                tracked_position.remove_empty_channels_post_tracking()

            cache['tracking'] = tracked_results

        # ( Diversion into ground truth processing, if applicable )

        if args.ground_truth:
            setup_matplotlib(log=log)

            return interactive_ground_truth_main(args, tracked_results)

        if args.advanced_ground_truth:
            setup_matplotlib(log=log)

            return interactive_advanced_ground_truth_main(args, tracked_results)

        # ( Output of textual results: )################################################################################

        def each_pos_k_tracking_tracker_channels_in_results(inner_tracked_results):
            """

            :param inner_tracked_results:
            """
            for inner_pos in sorted(inner_tracked_results.keys()):
                inner_tracking = inner_tracked_results[inner_pos]
                for inner_k in sorted(inner_tracking.tracker_mapping.keys()):
                    inner_tracker = inner_tracking.tracker_mapping[inner_k]
                    inner_channels = inner_tracking.channel_accumulator[inner_k]
                    yield inner_pos, inner_k, inner_tracking, inner_tracker, inner_channels

        if args.table_output is None:
            recipient = sys.stdout
        else:
            recipient = codecs.open(args.table_output, 'wb+', 'utf-8')

        log.info("Outputting tabular data ...")

        flat_results = list(each_pos_k_tracking_tracker_channels_in_results(tracked_results))

        try:
            table_dumper = QuickTableDumper(recipient=recipient)

            iterable = progress_bar(flat_results) if recipient is not sys.stdout else silent_progress_bar(flat_results)

            for pos, k, tracking, tracker, channels in iterable:
                analyze_tracking(tracker_to_cell_list(tracker), lambda x: table_dumper.add(x), meta=args.meta)

        finally:
            if recipient is not sys.stdout:
                recipient.close()

        # ( Output of graphical tracking results: )#####################################################################

        if args.tracking_output is not None:

            setup_matplotlib(interactive=False, log=log)
            # noinspection PyUnresolvedReferences
            import matplotlib
            # noinspection PyUnresolvedReferences
            import matplotlib.pylab

            log.info("Outputting graphical tracking data ...")

            figures_directory = os.path.abspath(args.tracking_output)

            if not os.path.isdir(figures_directory):
                os.mkdir(figures_directory)

            if args.tracking_output_format is None:
                args.tracking_output_format = {'pdf'}
            else:
                args.tracking_output_format = set(''.join(sublist) for sublist in args.tracking_output_format)

            for pos, k, tracking, tracker, channels in progress_bar(flat_results):

                tracking_filename = "%(dir)s/tracking_pt_%(mp)02d_chan_%(k)02d" % \
                                    {'dir': figures_directory, 'mp': pos, 'k': k}

                cs = CallSerialization()

                plot_timeline(cs.get_proxy(), channels, tracker_to_cell_list(tracker),
                              figure_presetup=
                              lambda p: p.title("Channel #%02d (average cells = %.2f)" % (k, tracker.average_cells)),
                              figure_finished=
                              lambda p: p,
                              show_images=True, show_overlay=True, leave_open=True)

                if 'kymograph' in args.tracking_output_format:
                    with open(tracking_filename + '.kymograph', 'wb') as fp:
                        fp.write(cs.as_pickle)

                if 'pdf' in args.tracking_output_format:
                    pylab = matplotlib.pylab
                    inject_poly_drawing_helper(pylab)
                    cs.execute(pylab)
                    pylab.savefig(tracking_filename + '.pdf')
                    pylab.close('all')

                del cs

    # ( Post-Tracking: Just write some tunables, if desired )###########################################################

    if args.write_tunables:
        if os.path.isfile(args.write_tunables):
            log.warning("Tunable output will not overwrite existing files!")
            log.warning("NOT outputting tunables.")
        else:
            tunable_output_name = os.path.abspath(args.write_tunables)
            log.info("Writing tunables to \"%(tunable_output_name)s\"" % {'tunable_output_name': tunable_output_name})
            with codecs.open(tunable_output_name, 'wb+', 'utf-8') as fp:
                json.dump(TunableManager.get_defaults(), fp, indent=4, sort_keys=True)

    log.info("Analysis finished.")

