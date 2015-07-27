
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

#####

from multiprocessing import Process, Pipe

class Future:

    def __init__(self):
        self.command = None
        self.args = None
        self.kwargs = None

        self.value = None
        self.error = None

        self.process = None
        self.pool = None

        self.status = None

        self.timeout = 0
        self.started_at = None

    def wait(self, time=None):
        pass

    def ready(self):
        if self.status:
            return self.status

        if self.process is None:
            # not scheduled yet
            return False

        if self.timeout > 0:
            now = datetime.datetime.now()
            if (now - self.started_at).total_seconds() > self.timeout:
                # we reached a hard timeout
                self.process.terminate()
                self.pool.report_broken_process(self.process)
                self.status, (self.value, self.error) = \
                    True, (None, RuntimeError('Process took longer than specified timeout and was terminated.'))
                return

        if not self.process.is_alive():
            self.pool.report_broken_process(self.process)
            self.status, (self.value, self.error) = True, (None, RuntimeError('Process trying to work on this future died.'))
            return

        self.status, (self.value, self.error) = self.process.ready()

        if self.status:
            self.pool.future_became_ready(self)

        return self.status

    def get(self):
        not_ready = True
        while not_ready:
            not_ready = not self.ready()

        if self.error:
            raise self.error
        else:
            return self.value

    def dispatch(self):
        self.started_at = datetime.datetime.now()
        self.process.dispatch()


import datetime
import traceback

class FutureProcess(Process):

    STARTUP = 0
    RUN = 1
    STOP = 2

    def __init__(self):
        super(FutureProcess, self).__init__()
        self.future = None
        self.pipe_parent_end, self.pipe_child_end = Pipe()

    def run(self):
        while True:
            command_type, command, args, kwargs = self.pipe_child_end.recv()

            if command_type == FutureProcess.STOP:
                break

            if command_type == FutureProcess.STARTUP and command is None:
                continue

            result = None
            exc = None

            try:
                result = command(*args, **kwargs)
            except Exception as e:
                message = traceback.print_exc()
                exc = RuntimeError("Exception during multiprocessing, original exception:\n" + message)

            if command_type == FutureProcess.STARTUP:
                continue

            self.pipe_child_end.send((result, exc,))

    def send_command(self, *args):
        self.pipe_parent_end.send(args)

    def dispatch(self):
        self.send_command(FutureProcess.RUN, self.future.command, self.future.args, self.future.kwargs)

    def ready(self):
        if self.pipe_parent_end.poll():
            return True, self.pipe_parent_end.recv()
        else:
            return False, (None, None,)



class SimpleProcessPool:

    def new_process(self):
        p = FutureProcess()
        p.start()
        p.send_command(*self.startup_message)
        return p

    def __init__(self, processes=0, initializer=None, initargs=[], initkwargs={}, future_timeout=0):

        self.startup_message = (FutureProcess.STARTUP, initializer, initargs, initkwargs)

        if processes == 0:
            processes = cpu_count()

        self.future_timeout = future_timeout
        self.count = processes

        self.waiting_processes = {self.new_process() for _ in range(processes)}
        self.active_processes = set()

        self.active_futures = set()
        self.waiting_futures = set()

        self.closing = False

    def close(self):
        self.closing = True

        self.schedule()

    def report_broken_process(self, p):
        f = p.future
        if p in self.active_processes:
            self.active_processes.remove(p)

        if p in self.waiting_processes:
            print("found a process where it does not belong", p)
            self.waiting_processes.remove(p)

        if f in self.active_futures:
            self.active_futures.remove(f)

        if f in self.waiting_futures:
            print("found a future where it does not belong", f)
            self.waiting_futures.remove(f)

        # and restart a new one
        self.waiting_processes.add(self.new_process())

        self.schedule()

    def apply(self, command, *args, **kwargs):
        f = Future()
        f.command = command
        f.args = args
        f.kwargs = kwargs

        f.timeout = self.future_timeout

        f.pool = self

        self.waiting_futures.add(f)

        self.schedule()

        return f

    # ugly signature
    def apply_async(self, fun, args=(), kwargs={}):
        return self.apply(fun, *args, **kwargs)

    def future_became_ready(self, f):
        if f in self.active_futures:
            self.active_futures.remove(f)
        if f.process in self.active_processes:
            self.active_processes.remove(f.process)
        self.waiting_processes.add(f.process)

        self.schedule()

    def schedule(self):
        for f in self.active_futures.copy():
            f.ready()

        while len(self.waiting_processes) > 0:
            if len(self.waiting_futures) == 0:
                break

            f = self.waiting_futures.pop()

            p = self.waiting_processes.pop()

            f.process = p
            p.future = f

            self.active_processes.add(p)
            self.active_futures.add(f)

            f.dispatch()

        if self.closing:
            while len(self.waiting_processes) > 0:
                p = self.waiting_processes.pop()
                p.send_command(FutureProcess.STOP, None, [], {})
                self.active_processes.add(p)


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

from inspect import getargspec, isclass

class Pipeline:
    debug = False
    strip = True

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

    class NeatDict(dict):
        def __getattr__(self, item):
            return self.get(item)

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, item):
            del self[item]

    def wrap(self, what, keep=None, delete=None):
        if isclass(what):
            argspec = getargspec(what.__call__)
        else:
            argspec = getargspec(what)

        debug = self.debug

        if isclass(what):
            #instance = what()
            # we create an instance without calling the constructor, so we can setup the environment first
            instance = what.__new__(what)
            instance.my_env = self.__class__.Environment()
            instance.env = self.environment

            for k, v in self.shared_variables.items():
                instance.__dict__[k] = v

            for k in self.step_connected_variables:
                instance.__dict__[k] = getattr(self, k)

            # now we call the constructor, and it has everything neatly set up already!
            instance.__init__()

            real_call = instance
        else:
            real_call = what

        NeatDict = self.__class__.NeatDict

        name = self.simplify_callable_name(what)

        if delete is None:
            _delete = set()
        else:
            _delete = set(delete)

        if keep is None:
            _keep = set()
        else:
            _keep = set(keep)

        KEY_COLLECTED = 'collected'
        KEY_RESULT = 'result'
        KEY_META = 'meta'

        set_of_keep_values = set([KEY_META, KEY_COLLECTED])

        def callable(step, meta, result):
            real_call.step = step
            if debug:
                print("Entering " + repr(instance))

            result = NeatDict(result)

            result.meta = meta
            result.result = True

            if argspec.args[0] == 'self':
                args = argspec.args[1:]
            else:
                args = argspec.args

            defaults = argspec.defaults if argspec.defaults else tuple()
            non_default_parameters = len(args) - len(defaults)

            if KEY_COLLECTED in result:
                wrapped = OrderedDict()
                for k, v in result[KEY_COLLECTED].items():
                    wrapped[k] = NeatDict(v)
                result[KEY_COLLECTED] = wrapped

            parameters = []
            for n, arg in enumerate(args):
                if arg == KEY_RESULT:
                    parameters.append(result)
                elif arg in result:
                    parameters.append(result[arg])
                else:
                    if n >= non_default_parameters:
                        parameters.append(defaults[n - non_default_parameters])
                    else:
                        # problem: pipeline step asks for a parameter we do not have
                        raise ValueError('[At %s]: Argument %r not in %r'  % (name, arg, result,))

            _call_return = real_call(*parameters)

            if type(_call_return) == dict and KEY_RESULT in _call_return:
                # if we get a dict back, we merge it with the ongoing result object
                result.update(_call_return)
            elif type(_call_return) == NeatDict:
                # if we get a neatdict back, we assume its the proper result object
                # and the pipeline step knew what it did ...
                # we continue with it as-is
                result = _call_return
            else:
                if type(_call_return) != tuple:
                    _call_return = (_call_return,)

                for n, item in enumerate(reversed(_call_return)):
                    k = args[-(n+1)]
                    if k == KEY_RESULT:
                        if type(item) == dict or type(item) == NeatDict:
                            result.update(item)
                    else:
                        result[k] = item

            if KEY_COLLECTED in result:
                unwrapped = OrderedDict()
                for k, v in result[KEY_COLLECTED].items():
                    unwrapped[k] = dict(v)
                result[KEY_COLLECTED] = unwrapped

            result = dict(result)

            all_to_delete = _delete.intersection(set(result.keys()))
            if len(_keep) > 0:
                all_to_delete |= set(result.keys()).difference(_keep | set_of_keep_values)

            for d in all_to_delete:
                del result[d]

            if KEY_RESULT in result:
                del result[KEY_RESULT]

            if debug:
                print("Leaving " + repr(instance))
            return result

        callable.__name__ = name
        callable.__qualname__ = callable.__name__

        return callable

    def simplify_callable_name(self, what):
        return ("Class_" if isclass(what) else "Function_") + ('__LAMBDA__' if what.__name__ == '<lambda>' else what.__name__)

    def add_step(self, when, what, keep=None, delete=None):
        what_wrapped = self.wrap(what, keep=keep, delete=delete)
        if when not in self.steps:
            self.steps[when] = what_wrapped
        else:
            last = self.steps[when]
            new = what_wrapped

            def callable(step, meta, *args, **kwargs):
                result = last(step, meta, *args, **kwargs)
                if type(result) != tuple:
                    result = (result,)
                return new(step, meta, *result, **kwargs)

            callable.__name__ = self.simplify_callable_name(what) + "_with_dependencies"
            callable.__qualname__ = callable.__name__

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
            for k in order:
                todo[k] = self.__class__.NotDispatchedYet
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

#            pool = Pool(
            pool = SimpleProcessPool(
                processes=self.multiprocessing,
                initializer=singleton_class_mapper,
                initargs=(self.__class__, '__full_init__', (synced_vars,), {},),
                #
                future_timeout=30.0*60,  # five minute timeout, only works with the self-written pool
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

        check = OrderedDict()

        cache_originated = set()

        while len(todo) > 0 or len(check) > 0:
            for op in list(todo.keys()):
                parameter = None
                if is_concrete(op):
                    parameter = ({},)

                elif len(mapping[op]) != 0:
                    continue
                else:
                    collected = OrderedDict()
                    for fetch in sorted(mapping_copy[op], key=lambda t: [t[i] for i in sort_order]):
                        collected[fetch] = results[fetch]
                    parameter = ({'collected': collected},)

                token = (reverse_todo[op], op,)
                if self.cache and token in self.cache:
                    #print(token, token in self.cache)
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
                    if parameter is None:
                        parameter = tuple({})

                    complete_params = (reverse_todo[op], op, ) + parameter  # deepcopy?

                    if pool:
                        result = pool.apply_async(
                            singleton_class_mapper,
                            args=(self.__class__, 'dispatch', complete_params, {},)
                        )
                    else:
                        def _dispatch_factory(what):
                            def _dispatch_function():
                                return self.dispatch(*what)
                            return _dispatch_function
                        result = self.__class__.DuckTypedApplyResult(_dispatch_factory(complete_params))

                results[op] = result

                check[op] = True

                del todo[op]

            for op in list(check.keys()):
                result = results[op]
                if self.wait:
                    result.wait(self.wait)

                if result.ready():
                    try:
                        result = result.get()

                        if self.cache and op not in cache_originated:
                            # do not cache 'None' as a result
                            if result is not None:
                                token = (reverse_todo[op], op,)
                                # so far, solely accessing (write) the cache from one process should mitigate locking issues
                                self.set_cache(token, result)

                    except Exception as e:
                        self.log.exception("Exception occurred at op: %s", repr(reverse_todo[op]) + ' ' + repr(op))
                        result = None

                    results[op] = result

                    if op in reverse_mapping:
                        for affected in reverse_mapping[op]:
                            mapping[affected] -= {op}

                    del check[op]
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
        argparser.add_argument('--prompt', '--prompt', dest='wait_on_start', default=False, action='store_true')
        argparser.add_argument('-tp', '--timepoints', dest='timepoints', default='0-', type=str)
        argparser.add_argument('-mp', '--multipoints', dest='multipoints', default='0-', type=str)

        if self.options['tunables']:
            argparser.add_argument('-t', '--tunables', dest='tunables', type=str, default=None)
            argparser.add_argument('-pt', '--print-tunables', dest='print_tunables', default=False, action='store_true')
            argparser.add_argument('-rt', '--read-tunables', dest='read_tunables', type=str, default=None)
            argparser.add_argument('-wt', '--write-tunables', dest='write_tunables', type=str, default=None)

        return argparser

    def _parse_ranges(self, args, ims):
        self.positions = parse_range(args.multipoints, maximum=ims.get_meta('multipoints'))
        self.timepoints = parse_range(args.timepoints, maximum=ims.get_meta('timepoints'))

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

        if self.args.wait_on_start:
            _ = input("Press enter to continue.")

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
        argparser.add_argument('-tp', '--timepoints', dest='timepoints', default='0-', type=str)
        argparser.add_argument('-mp', '--multipoints', dest='multipoints', default='0-', type=str)

        if self.options['tunables']:
            argparser.add_argument('-t', '--tunables', dest='tunables', type=str, default=None)
            argparser.add_argument('-pt', '--print-tunables', dest='print_tunables', default=False, action='store_true')
            argparser.add_argument('-rt', '--read-tunables', dest='read_tunables', type=str, default=None)
            argparser.add_argument('-wt', '--write-tunables', dest='write_tunables', type=str, default=None)

        return argparser

    def _parse_ranges(self, args, ims):
        self.positions = parse_range(args.multipoints, maximum=ims.get_meta('multipoints'))
        self.timepoints = parse_range(args.timepoints, maximum=ims.get_meta('timepoints'))

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

