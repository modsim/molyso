
import argparse
import sys
import os
import numpy
import itertools
import codecs
import json
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
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
from itertools import product
from collections import OrderedDict
from copy import deepcopy


def singleton_class_mapper(klass, what, args, kwargs, local_cache={}):
    try:
        if klass not in local_cache:
            local_cache[klass] = klass.__new__(klass)
        return getattr(local_cache[klass], what)(*args, **kwargs)

    except Exception as e:
        #print(traceback.print_exc())
        #print(e)
        raise

class Source:
    def __call__(self, meta):
        return []

class Mapper:
    def __call__(self, meta, parameter):
        return parameter

class Reducer:
    def __call__(self, meta, parameter):
        return parameter

class Sink:
    def __call__(self, meta, parameter):
        print(parameter)


class Collected:
    pass

class Every:
    pass


class Pipeline:


    class Environment:
        pass

    class DuckTypedApplyResult:
        def __init__(self, callable_):
            self.value = None
            self.called = False
            self.callable = callable_

        def ready(self):
            return True

        def wait(self, timeout):
            pass

        def get(self):
            if not self.called:
                self.value = self.callable()
            return self.value

    class NotDispatchedYet:
        pass

    def wrap(self, what):

        self.debug = False#True
        debug = self.debug

        if type(what) == type(Pipeline):
            instance = what()
            instance.my_env = self.__class__.Environment()
            instance.env = self.environment

            for k, v in self.shared_variables.items():
                instance.__dict__[k] = v

            for k in self.step_connected_variables:
                instance.__dict__[k] = getattr(self, k)

            def callable(step, meta, *args, **kwargs):
                instance.step = step
                if debug:
                    print("Entering " + repr(instance))
                _result = instance(meta, *args, **kwargs)
                if debug:
                    print("Leaving " + repr(instance))
                return _result
            return callable
        else:
            def callable(step, meta, *args, **kwargs):
                if debug:
                    print("Entering " + repr(what))
                _result = what(meta, *args, **kwargs)
                if debug:
                    print("Leaving " + repr(what))
                return _result
            return callable

    def add_step(self, when, what):
        if when not in self.steps:
            self.steps[when] = self.wrap(what)
        else:
            last = self.steps[when]
            new = self.wrap(what)

            def callable(step, meta, *args, **kwargs):
                result = last(step, meta, *args, **kwargs)
                if type(result) != tuple:
                    result = (result,)
                return new(step, meta, *result, **kwargs)

            self.steps[when] = callable

    def dispatch(self, step, *args, **kwargs):
        return self.steps[step](step, *args, **kwargs)

    def setup(self):
        pass

    def setup_call(self):
        self.steps = {}
        self.environment = self.__class__.Environment()
        self.shared_variables = {}
        if getattr(self, 'step_connected_variables', None) is None:
            self.step_connected_variables = set()
        self.synced_variables = {'meta_tuple', 'processing_order', 'item_counts', 'step_connected_variables'}

    def add_step_connected_variable(self, name):
        self.step_connected_variables |= {name}

    def add_shared_variable(self, name, value):
        self.shared_variables[name] = value

    def add_pipeline_synced_variable(self, name):
        self.synced_variables |= {name}

    def pre_setup(self):
        pass

    def post_setup(self):
        pass

    def worker_setup(self):
        pass

    def root_setup(self):
        pass

    def complete_setup(self, root=False):
        self.setup_call()
        self.pre_setup()
        if root:
            self.root_setup()
        self.worker_setup()
        self.post_setup()

    def __full_init__(self, synced_vars):
        self.__init__()

        for k, v in synced_vars.items():
            self.__dict__[k] = v

        self.complete_setup()

    def get_cache(self, key):
        if self.cache:
            return self.cache[key]
        else:
            raise RuntimeError('No cache defined')

    def set_cache(self, key, value):
        if self.cache:
            self.cache[key] = value

    def run(self, progress_bar=None):
        self.complete_setup(root=True)

        self.wait = 0.01

        meta_tuple = self.meta_tuple
        processing_order = self.processing_order
        item_counts = self.item_counts

        if getattr(self, 'cache', False) is False:
            self.cache = False

        sort_order = [index for index, _ in sorted(enumerate(processing_order), key=lambda p: p[0])]

        def prepare_steps(step, replace):
            return list(meta_tuple(*t) for t in sorted(product(*[
                item_counts[num] if value == replace else [value] for num, value in
                enumerate(step)
            ]), key=lambda t: [t[i] for i in sort_order]))

        todo = OrderedDict()

        reverse_todo = {}
        results = {}

        mapping = {}
        reverse_mapping = {}

        for step in self.steps.keys():
            order = prepare_steps(step, Every)
            reverse_todo.update({k: step for k in order})
            todo.update({k: self.__class__.NotDispatchedYet for k in order})
            deps = {t: set(prepare_steps(t, Collected)) for t in order}
            mapping.update(deps)
            for key, value in deps.items():
                for k in value:
                    if k not in reverse_mapping:
                        reverse_mapping[k] = {key}
                    else:
                        reverse_mapping[k] |= {key}

        mapping_copy = deepcopy(mapping)

        def is_concrete(t):
            for n in t:
                if n is Collected or n is Every:
                    return False
            return True

        if self.multiprocessing:
            if self.multiprocessing is True:
                self.multiprocessing = cpu_count()

            synced_vars = {k: getattr(self, k) for k in self.synced_variables}

            pool = Pool(
                processes=self.multiprocessing,
                initializer=singleton_class_mapper,
                initargs=(self.__class__, '__full_init__', (synced_vars,), {},)
            )
        else:
            pool = None

        initial_length = len(todo)

        if progress_bar:
            pbar = progress_bar(initial_length)

            def progressOne():
                try:
                    next(pbar)
                except StopIteration:
                    pass
        else:
            def progressOne():
                pass

        check = set()

        cache_originated = set()

        while len(todo) > 0 or len(check) > 0:
            for op in list(todo.keys()):
                parameter = None
                if is_concrete(op):
                    parameter = ([],)

                elif len(mapping[op]) != 0:
                    continue
                else:
                    parameter = ({fetch: results[fetch] for fetch in sorted(mapping_copy[op], key=lambda t: [t[i] for i in sort_order])},)

                token = repr(reverse_todo[op]) + ' ' + repr(op)
                if self.cache and token in self.cache:
                    print(token, token in self.cache)
                    if pool:
                        result = pool.apply_async(
                            singleton_class_mapper,
                            args=(self.__class__, 'get_cache', (token,), {},)
                        )
                    else:
                        def _cache_fetch_factory(what):
                            def _cache_fetch_function():
                                return self.get_cache(what)
                            return _cache_fetch_function
                        result = self.__class__.DuckTypedApplyResult(_cache_fetch_factory(token))

                    cache_originated |= {op}
                else:

                    complete_params = deepcopy((reverse_todo[op], op, ) + parameter)


                    if parameter is None:
                        parameter = tuple()
                    if pool:
                        result = pool.apply_async(
                            singleton_class_mapper,
                            args=(self.__class__, 'dispatch', complete_params, {},)
                        )
                    else:
                        result = self.__class__.DuckTypedApplyResult(lambda: self.dispatch(*complete_params))

                results[op] = result

                check |= {op}

                del todo[op]

            for op in list(check):
                result = results[op]
                if self.wait:
                    result.wait(self.wait)

                if result.ready():
                    try:
                        result = result.get()

                        if self.cache and op not in cache_originated:
                            # do not cache 'None' as a result
                            if result is not None:
                                # new cache supports 'arbitrary' keys, use a tuple TODO
                                token = repr(reverse_todo[op]) + ' ' + repr(op)
                                # so far, solely accessing (write) the cache from one process should mitigate locking issues
                                self.set_cache(token, result)

                    except Exception as e:
                        self.log.exception("Exception occurred at op: %s", repr(reverse_todo[op]) + ' ' + repr(op))
                        result = None

                    results[op] = result

                    if op in reverse_mapping:
                        for affected in reverse_mapping[op]:
                            mapping[affected] -= {op}

                    check -= {op}
                    progressOne()

        progressOne()

        if pool:
            pool.close()



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



class PipelineApplicationInterface(Pipeline):
    def internal_options(self):
        pass

    def arguments(self, argparser):
        pass

    def before_main(self):
        pass

    def after_main(self):
        pass

    def setup(self):  # repeat from Pipeline
        pass


from signal import signal, SIGUSR2
import traceback

def maintenance_interrupt(signal, frame):

    print("Interrupted at:")
    print(''.join(traceback.format_stack(frame)))
    print("Have a look at frame.f_globals and frame.f_locals")
    try:
        raise ImportError
        from IPython import embed
        embed()
    except ImportError:
        from code import interact
        interact(local=locals())
    print("... continuing")



class PipelineApplication(PipelineApplicationInterface, Pipeline):
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

    def pre_setup(self):
        self._setup_modules()

        correct_windows_signal_handlers()

        signal(SIGUSR2, maintenance_interrupt)

        self.add_pipeline_synced_variable('args')

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

        #self.before_processing()


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
                        self.log.warning(
                            "Could not load either module %s_%s or %s!",
                            self.options['name'], module, module
                        )

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

        if self.args.mp < 0:
            self.args.mp = cpu_count()

        self.multiprocessing = self.args.mp if self.args.mp > 1 else False

        self.before_main()

        self._print_ranges()

        def progress_bar(num):
            return fancy_progress_bar(range(num))

        self.run(progress_bar=progress_bar)

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
        return TestPipeline().main()
        logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)s %(levelname)s %(message)s")

        self.argparser = self._create_argparser()
        self.arguments(self.argparser)
        self.args = self.argparser.parse_args()

        self.log.info(self.options['banner'])
        self.log.info("Started %s.", self.options['name'])

        self._setup_modules()

        self._parse_ranges(self.args, self.ims)

        if self.args.mp <= 0:
            self.args.mp = cpu_count()

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
            self.pool = Pool(processes=self.args.mp, initializer=ImageProcessingPipeline._multiprocess_start, initargs=(self.__class__, self.args,))

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


class TestPipeline(PipelineApplication):

    class ImageSource(Source):
        def __call__(self, meta, image):
            image = self.ims.get_image(t=meta.t, pos=meta.pos)
            print(image)
            return image


    class MeanMapper(Mapper):
        def __call__(self, meta, image):
            return image.mean()

    class MeanReducer(Reducer):
        def __call__(self, meta, data):
            return numpy.array(list(data.values())).mean()

    class PrintSink(Sink):
        def __call__(self, meta, data):
            print("Reached the PrintSink. Meta is: %s Data is: %s" % (repr(meta), repr(data)))

    def root_setup(self):
        self.meta_tuple = Meta
        self.processing_order = Meta(t=1, pos=2)
        self.item_counts = Meta(t=self.timepoints, pos=self.positions)

        self.add_pipeline_synced_variable('args')
        self.add_step_connected_variable('ims')

    def worker_setup(self):
        self.add_step(Meta(t=Every, pos=Every), TestPipeline.ImageSource)
        self.add_step(Meta(t=Every, pos=Every), TestPipeline.MeanMapper)
        self.add_step(Meta(t=Collected, pos=Every), TestPipeline.PrintSink)
        self.add_step(Meta(t=Collected, pos=Collected), TestPipeline.PrintSink)

