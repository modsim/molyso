# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

from .. import __citation__
import sys
import os
import numpy
import codecs

from .. import Debug

from ..generic.etc import debug_init, QuickTableDumper, silent_progress_bar, fancy_progress_bar, bits_to_numpy_type,\
    Cache

from .image import Image
from .fluorescence import FluorescentImage

from .tracking import TrackedPosition, analyze_tracking, plot_timeline, tracker_to_cell_list, \
    each_k_tracking_tracker_channels_in_results, each_pos_k_tracking_tracker_channels_in_results


from .highlevel_interactive_viewer import interactive_main
from .highlevel_interactive_ground_truth import interactive_ground_truth_main


from ..generic.pipeline import ImageProcessingPipeline

class MolysoPipeline(ImageProcessingPipeline):
    def internal_options(self):
        return {
            'name':
                "molyso",
            'description':
                "molyso: MOther machine anaLYsis SOftware",
            'banner':
                r"""
         \   /\  /\  /                             -------------------------
          | | |O| | |    molyso                    Developed  2013 - 2015 by
          | | | | |O|                              Christian   C.  Sachs  at
          |O| |O| |O|    MOther    machine         ModSim / Microscale Group
          \_/ \_/ \_/    anaLYsis SOftware         Research  Center  Juelich
        --------------------------------------------------------------------
        If you use this software in a publication, please cite our paper:

        %(citation)s

        --------------------------------------------------------------------
        """ % {'citation': __citation__},
            'tunables': True
        }

    def arguments(self, argparser):
        argparser.add_argument('-p', '--process', dest='process', default=False, action='store_true')
        argparser.add_argument('-gt', '--ground-truth', dest='ground_truth', type=str, default=None)
        argparser.add_argument('-ct', '--cache-token', dest='cache_token', type=str, default=None)
        argparser.add_argument('-o', '--table-output', dest='table_output', type=str, default=None)
        argparser.add_argument('-ot', '--output-tracking', dest='tracking_output', type=str, default=None)
        argparser.add_argument('-nb', '--no-banner', dest='nb', default=False, action='store_true')
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

    def before_processing(self):
        self.first_frames = {}

    def get_first(self, pos):
        if pos in self.first_frames:
            return self.first_frames[pos]
        else:
            if self.ims.get_meta('channels') > 1:
                image = FluorescentImage()
            else:
                image = Image()

            image.setup_image(self.ims.get_image(pos=pos, t=0, channel=self.ims.__class__.Phase_Contrast))

            # not necessary
            self.populate_metadata(image, pos, 0)

            image.autorotate()
            image.autoregister(image)

            self.first_frames[pos] = image

            return image

    def populate_metadata(self, image, pos, t):
        image.multipoint = int(pos)
        image.timepoint_num = int(t)

        image.timepoint = self.ims.get_meta('time', t=t, pos=pos)

        image.calibration_px_to_mu = self.ims.get_meta('calibration', t=t, pos=pos)

        image.metadata['x'], image.metadata['y'], image.metadata['z'] = self.ims.get_meta('position', t=t, pos=pos)

        image.metadata['time'] = image.timepoint
        image.metadata['timepoint'] = image.timepoint_num
        image.metadata['multipoint'] = image.multipoint

        image.metadata['calibration_px_to_mu'] = image.calibration_px_to_mu

        image.metadata['tag'] = ''
        image.metadata['tag_number'] = 0


    def map_image(self, meta, image_data):
        first = self.get_first(meta.pos)

        if self.ims.get_meta('channels') > 1:
            image = FluorescentImage()
        else:
            image = Image()

        image.setup_image(image_data)

        if self.ims.get_meta('channels') > 1:
            for channel in self.ims.get_meta('fluorescenceChannels'):
                fimg = local_ims.get_image(t=meta.t, pos=meta.pos, channel=channel, float=True)
                i.setup_add_fluorescence(fimg)

        self.populate_metadata(image, meta.pos, meta.t)

        Debug.set_context(t=meta.t, pos=meta.pos)

        image.keep_channel_image = self.args.keepchan
        image.pack_channel_image = self.args.channel_bits

        if type(image) == FluorescentImage:
            self.image.keep_fluorescences_image = self.args.keepfluorchan
            self.image.pack_fluorescences_image = self.args.channel_fluorescence_bits

        image.autorotate()
        image.autoregister(first)

        image.find_channels()
        image.find_cells_in_channels()

        image.clean()

        image.flatten()

        return image

    def reduce_timepoints(self, meta, results):
        #pi = progress_bar(range(sum([len(l) - 1 if len(l) > 0 else 0 for l in results.values()]) - 1))

        tracked_position = TrackedPosition()
        tracked_position.set_times(results)
        tracked_position.align_channels()  # progress_indicator=pi
        tracked_position.remove_empty_channels()
        tracked_position.guess_channel_orientation()

        #pi = progress_bar(range(sum([tp.get_tracking_work_size() for tp in tracked_results.values()]) - 1))

        tracked_position.perform_tracking()  # progress_indicator=pi
        tracked_position.remove_empty_channels_post_tracking()

        return tracked_position

    def output(self, results):
        if self.args.table_output is None:
            recipient = sys.stdout
        else:
            recipient = codecs.open(self.args.table_output, 'wb+', 'utf-8')

        self.log.info("Outputting tabular data ...")

        flat_results = list(each_pos_k_tracking_tracker_channels_in_results(results))

        try:
            table_dumper = QuickTableDumper(recipient=recipient)

            iterable = fancy_progress_bar(flat_results) if recipient is not sys.stdout else silent_progress_bar(flat_results)

            for pos, k, tracking, tracker, channels in iterable:
                analyze_tracking(tracker_to_cell_list(tracker), lambda x: table_dumper.add(x))

        finally:
            if recipient is not sys.stdout:
                recipient.close()

    def output_multipoint(self, meta, results):
        if self.args.tracking_output is not None:
            try:
                import matplotlib
                import matplotlib.pylab
            except ImportError:
                self.log.error("Tracking output enabled but matplotlib not found! Cannot proceed. Please install matplotlib ...")
                raise

            figure_directory = os.path.abspath(self.args.tracking_output)

            if not os.path.isdir(figure_directory):
                os.mkdir(figure_directory)

            for k, tracking, tracker, channels in each_k_tracking_tracker_channels_in_results(results):
                plot_timeline(matplotlib.pylab, channels, tracker_to_cell_list(tracker),
                              figure_presetup=
                              lambda p: p.title("Channel #%02d (average cells = %.2f)" % (k, tracker.average_cells)),
                              figure_finished=
                              lambda p: p.savefig("%(dir)s/tracking_pt_%(mp)02d_chan_%(k)02d.pdf" %
                                                  {'dir': figure_directory, 'mp': meta.pos, 'k': k}),
                              show_images=True, show_overlay=True)

    def before_main(self):
        if sys.maxsize <= 2 ** 32:
            self.log.warning("Running on a 32 bit Python interpreter! This is most likely not what you want,"
                             "and it will significantly reduce functionality!")

        if not self.args.process:
            interactive_main(self.args)
            sys.exit(1)

        if self.args.debug and not self.args.ground_truth:
            debug_init()
            if self.args.mp != 0:
                self.log.warning("Debugging enabled, concurrent processing disabled!")
                self.args.mp = 0





########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#
#
#
# def main():
#
#     Cache.printer = print_info
#     cache = Cache(args.input, ignore_cache=args.ignorecache, cache_token=args.cache_token)
#
#     if 'tracking' not in cache:
#
#         if 'imageanalysis' in cache:
#             results = cache['imageanalysis']
#         else:
#             cache['imageanalysis'] = results
#
#     if not args.no_tracking:
#
#         if 'tracking' in cache:
#             tracked_results = cache['tracking']
#         else:
#             cache['tracking'] = tracked_results
#
#         # ( Diversion into ground truth processing, if applicable )
#
#         if args.ground_truth:
#             interactive_ground_truth_main(args, tracked_results)
#             return
#
#
#
