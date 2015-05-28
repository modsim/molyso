
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

from .etc import parse_range, correct_windows_signal_handlers, debug_init, QuickTableDumper, \
    silent_progress_bar, fancy_progress_bar, replace_inf_with_maximum, prettify_numpy_array, bits_to_numpy_type, Cache, \
    ignorant_next

from ..imageio.imagestack import MultiImageStack
from ..imageio.imagestack_ometiff import OMETiffStack

OMETiffStack = OMETiffStack

from collections import namedtuple

class ImageProcessingPipelineInterface:
    def internal_options(self):
        pass

    def arguments(self, argparser):
        pass

    def map_image(self, meta, image_data):
        pass

    def reduce_timepoints(self, meta, results):
        pass

    def reduce_images(self, results):
        pass

    def reduce_multipoints(self, results):
        pass

    def before_processing(self):
        pass

    def after_processing(self):
        pass

    def output(self, results):
        pass

    def output_multipoint(self, meta, results):
        pass

    def before_main(self):
        pass

    def after_main(self):
        pass

Meta = namedtuple('Meta', ['pos', 't'])

class ImageProcessingPipeline(ImageProcessingPipelineInterface):

    instance = None

    @staticmethod
    def _internal_option_defaults():
        return {
            'name':
                "processor",
            'description':
                "processor",
            'order':
                'PT',
            'banner':
                "",
            'tunables': False
        }

    @property
    def options(self):
        if not getattr(self, '_options', False):
            self._options = self._internal_option_defaults().copy()
            if self.subclass_implements_function(self.internal_options):
                self._options.update(self.internal_options())
        return self._options

    def subclass_implements_function(self, what):
        return what.__name__ in self.__class__.__dict__

    def _iterate_in_order(self):
        if self.options['order'] == 'PT':
            for pos in self.positions:
                for tp in self.timepoints:
                    yield pos, tp
        elif self.options['order'] == 'TP':
            for tp in self.timepoints:
                for pos in self.positions:
                    yield pos, tp
        else:
            raise RuntimeError("Wrong order passed")



    def _create_argparser(self):
        argparser = argparse.ArgumentParser(description=self.options['description'])

        def _error(message=''):
            self.log.info(self.options['banner'])
            argparser.print_help()
            self.log.error("command line argument error: %s", message)
            sys.exit(1)

        argparser.error = _error

        argparser.add_argument('input', metavar='input', type=str, help="input file")
        argparser.add_argument('-m', '--module', dest='modules', type=str, default=None, action='append')
        argparser.add_argument('-cpu', '--cpus', dest='mp', default=-1, type=int)
        argparser.add_argument('-tp', '--timepoints', dest='timepoints', default=[0, float('inf')], type=parse_range)
        argparser.add_argument('-mp', '--multipoints', dest='multipoints', default=[0, float('inf')], type=parse_range)

        if self.options['tunables']:
            argparser.add_argument('-t', '--tunables', dest='tunables', type=str, default=None)
            argparser.add_argument('-pt', '--print-tunables', dest='print_tunables', default=False, action='store_true')
            argparser.add_argument('-rt', '--read-tunables', dest='read_tunables', type=str, default=None)
            argparser.add_argument('-wt', '--write-tunables', dest='write_tunables', type=str, default=None)

        return argparser

    def _parse_ranges(self, args, ims):
        self.positions = replace_inf_with_maximum(args.multipoints, ims.get_meta('multipoints'))
        self.timepoints = replace_inf_with_maximum(args.timepoints, ims.get_meta('timepoints'))

    def _print_ranges(self):
        self.log.info(
            "Beginning Processing:\n%s\n%s",
            prettify_numpy_array(self.positions,  "Positions : "),
            prettify_numpy_array(self.timepoints, "Timepoints: ")
        )

    @classmethod
    def _multiprocess_start(cls, what, args):
        cls.instance = what()
        cls.instance.args = args
        cls.instance.internal_before_processing()

    @classmethod
    def _multiprocess_map_image(cls, *args, **kwargs):
        return cls.instance.internal_map_image(*args, **kwargs)

    def internal_map_image(self, meta):
        image_data = self.ims.get_image(t=meta.t, pos=meta.pos, channel=self.ims.__class__.Phase_Contrast, raw=True)

        return self.map_image(meta, image_data)

    @classmethod
    def _multiprocess_output_multipoint(cls, *args, **kwargs):
        return cls.instance.internal_output_multipoint(*args, **kwargs)

    def internal_output_multipoint(self, meta, results):
        return self.output_multipoint(meta, results)

    def internal_before_processing(self):
        self._setup_modules()
        correct_windows_signal_handlers()

        if self.options['tunables']:
            # load tunables
            if self.args.read_tunables:
                with open(self.args.read_tunables, 'r') as tunable_file:
                    tunables = json.load(tunable_file)
                    self.log.info("Loaded tunable file \"%s\" with data: %s", self.args.read_tunables, repr(tunables))
                    TunableManager.load_tunables(tunables)

            if self.args.tunables:
                tunables = json.loads(self.args.tunables)
                self.log.info("Loaded command line tunables: %(data)s" % {'data': repr(tunables)})
                TunableManager.load_tunables(tunables)

            if self.args.print_tunables:
                TunableManager.set_printing(True)

        self.before_processing()

    @classmethod
    def _multiprocess_process_timepoints(cls, meta, results):
        return cls.instance.reduce_timepoints(meta, results)

    def _setup_modules(self):
        modules = self.args.modules
        if modules:
            import importlib
            for module in modules:
                try:
                    importlib.import_module("%s_%s" % (self.options['name'], module))
                except ImportError:
                    try:
                        importlib.import_module(module)
                    except ImportError:
                        self.log.warning("Could not load either module %s_%s or %s!",
                                      self.options['name'], module, module)

    @property
    def ims(self):
        if getattr(self, '_ims', None) is None:
            self._ims = MultiImageStack.open(self.args.input)

        return self._ims

    @property
    def log(self):
        if getattr(self, '_log', None) is None:
            self._log = logging.getLogger(self.options['name'])

        return self._log

    def main(self):

        logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)s %(levelname)s %(message)s")

        self.argparser = self._create_argparser()
        self.arguments(self.argparser)
        self.args = self.argparser.parse_args()

        self.log.info(self.options['banner'])
        self.log.info("Started %s.", self.options['name'])

        self._setup_modules()

        self._parse_ranges(self.args, self.ims)

        if self.args.mp <= 0:
            self.args.mp = multiprocessing.cpu_count()

        self.before_main()

        self._print_ranges()

        results = {}
        structured_results = {pos: {t: None for t in self.timepoints} for pos in self.positions}
        missing = {pos: len(self.timepoints) for pos in self.positions}

        multipoint_results = {pos: None for pos in self.positions}

        pbar = fancy_progress_bar(range(len(self.positions) * len(self.timepoints)))

        if self.args.mp < 2:

            self.internal_before_processing()

            for pos, tp in self._iterate_in_order():
                try:
                    result = self.internal_map_image(pos, tp)
                except Exception:
                    self.log.exception("Exception occurred at pos: %d, time %d", pos, tp)
                results[(pos, tp)] = result
                structured_results[pos][tp] = result
                missing[pos][tp] -= 1
                if missing[pos][tp] == 0:
                    multipoint_results[pos] = self.reduce_timepoints(Meta(pos, None), structured_results[pos])

                next(pbar)

            ignorant_next(pbar)

        else:
            self.pool = multiprocessing.Pool(processes=self.args.mp, initializer=ImageProcessingPipeline._multiprocess_start, initargs=(self.__class__, self.args,))

            jobs = {}

            mp_jobs = {}

            output_mp_jobs = {}

            for pos, tp in self._iterate_in_order():
                jobs[(pos, tp)] = self.pool.apply_async(ImageProcessingPipeline._multiprocess_map_image, args=(Meta(pos, tp),))

            while len(jobs) > 0:
                for (pos, tp), c in list(jobs.items()):
                    if c.ready():
                        try:
                            result = c.get()
                        except Exception:
                            self.log.exception("Exception occurred at pos: %d, time %d", pos, tp)
                        results[(pos, tp)] = result
                        del jobs[(pos, tp)]
                        structured_results[pos][tp] = result
                        missing[pos] -= 1
                        if missing[pos] == 0:
                            mp_jobs[pos] = self.pool.apply_async(ImageProcessingPipeline._multiprocess_process_timepoints, args=(Meta(pos, None), structured_results[pos]))

                        next(pbar)

            ignorant_next(pbar)

            pbar = fancy_progress_bar(range(len(self.positions)))

            while len(mp_jobs) > 0:
                for pos, c in list(mp_jobs.items()):
                    if c.ready():
                        try:
                            result = c.get()
                        except Exception:
                            self.log.exception("Exception occurred at pos: %d", pos)
                        multipoint_results[pos] = result
                        del mp_jobs[pos]
                        output_mp_jobs[pos] = self.pool.apply_async(ImageProcessingPipeline._multiprocess_output_multipoint, args=(Meta(pos, None), result))
                        next(pbar)

            ignorant_next(pbar)

            pbar = fancy_progress_bar(range(len(self.positions)))

            while len(output_mp_jobs) > 0:
                for pos, c in list(output_mp_jobs.items()):
                    if c.ready():
                        try:
                            result = c.get()
                        except Exception:
                            self.log.exception("Exception occurred at pos: %d", pos)
                        del output_mp_jobs[pos]

                        next(pbar)
            ignorant_next(pbar)
            self.pool.close()

        self.reduce_images(structured_results)
        self.reduce_multipoints(multipoint_results)

        self.after_processing()
        self.output(multipoint_results)

        if self.options['tunables']:
            if self.args.write_tunables:
                if os.path.isfile(self.args.write_tunables):
                    self.log.warning("Tunable output will not overwrite existing files! NOT tunables output.")
                else:
                    fname = os.path.abspath(self.args.write_tunables)
                    self.log.info("Writing tunables to \"%s\"", fname)
                    with codecs.open(fname, 'wb+', 'utf-8') as fp:
                        json.dump(TunableManager.get_defaults(), fp, indent=4, sort_keys=True)

        self.after_main()

        self.log.info("Finished %s.", self.options['name'])