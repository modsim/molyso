# -*- coding: utf-8 -*-
"""
etc.py contains various helper functions and classes, which are not directly related to data processing.
"""
from __future__ import division, unicode_literals, print_function
import os
import sys
import numpy
import hashlib
import time

from .. import Debug


def silent_progress_bar(iterable):
    return iter(iterable)


try:
    import clint.textui

    def fancy_progress_bar(iterable):
        return clint.textui.progress.bar(iterable, width=50)

        # raise ImportError
except ImportError:
    def fancy_progress_bar(iterable):
        times = numpy.zeros(len(iterable), dtype=float)
        for n, i in enumerate(iterable):
            start_time = time.time()
            yield i
            stop_time = time.time()
            times[n] = stop_time - start_time
            eta = " ETA %.2fs" % (numpy.mean(times[:n + 1]) * (len(iterable) - (n + 1)))
            print("processed %d/%d [took %.3fs%s]" % (n + 1, len(iterable), times[n], eta))

def iter_time(what):
    start_time = time.time()
    for n in what:
        yield n
    stop_time = time.time()
    print("whole step took %.3fs" % (stop_time - start_time,))

_fancy_progress_bar = fancy_progress_bar

fancy_progress_bar = lambda x: iter_time(_fancy_progress_bar(x))


class QuickTableDumper(object):
    def __init__(self, recipient=None):
        if recipient is None:
            recipient = sys.stdout

        self.recipient = recipient
        self.headers = []

        self.delim = '\t'
        self.lineend = '\n'
        self.precision = 8

    def write(self, s):
        self.recipient.write(s)

    def __iadd__(self, other):
        self.write(other)

    def add(self, row):
        if len(self.headers) == 0:
            self.headers = list(row.keys())

            self.write(self.delim.join(self.headers) + self.lineend)

        self.write(self.delim.join([self.stringer(row[k]) for k in self.headers]) + self.lineend)

    def stringer(self, obj):
        if type(obj) == float or type(obj) == numpy.float64:
            if self.precision:
                return str(round(obj, self.precision))
            else:
                return str(obj)
        else:
            return str(obj)


try:
    # noinspection PyUnresolvedReferences
    import cPickle

    pickle = cPickle
except ImportError:
    import pickle

try:
    import thread
except ImportError:
    import _thread as thread

if os.name != 'nt':
    def correct_windows_signal_handlers():
        pass
else:
    def correct_windows_signal_handlers():
        os.environ['PATH'] += os.path.pathsep + os.path.dirname(os.path.abspath(sys.executable))

        try:
            # noinspection PyUnresolvedReferences
            import win32api

            def handler(_, hook=thread.interrupt_main):
                hook()
                return 1

            win32api.SetConsoleCtrlHandler(handler, 1)

        except ImportError:
            print("Warning: Running on Windows, but module 'win32api' could not be imported to fix signal handler.")
            print("Ctrl-C might break the program ...")
            print("Fix: Install the module!")


def debug_init():
    Debug.enable('text', 'plot', 'plot_pdf')
    numpy.set_printoptions(threshold=numpy.nan)


def parse_range(s, allow_open_interval=True):
    ranges = []
    tail = []
    for frag in s.replace(';', ',').split(','):
        if '-' in frag:
            f, t = frag.split('-')
            if t == '':
                if allow_open_interval:
                    tail = [int(f), float('Inf')]
                else:
                    raise ValueError("parse_range called with allow_open_interval=False"
                                     " but open interval string was passed.")
            else:
                ranges += range(int(f), int(t) + 1)
        else:
            ranges += [int(frag)]
    ranges += tail
    return ranges


def replace_inf_with_maximum(array, maximum):
    if array[-1] == float('Inf'):
        f = array[-2]
        del array[len(array) - 2:len(array)]
        array += range(f, maximum)

    array = [e for e in array if 0 <= e <= maximum]

    return array


def prettify_numpy_array(arr, space_or_prefix):
    six_spaces = ' ' * 6
    prepared = repr(numpy.array(arr)).replace(')', '').replace('array(', six_spaces)
    if type(space_or_prefix) == int:
        return prepared.replace(six_spaces, ' ' * space_or_prefix)
    else:
        return space_or_prefix + prepared.replace(six_spaces, ' ' * len(space_or_prefix)).lstrip()



def bits_to_numpy_type(bits):
    # this is supposed to throw an error
    return {
        8: numpy.uint8,
        16: numpy.uint16,
        32: numpy.float32
    }[int(bits)]

class Cache(object):
    printer = print

    def build_cache_filename(self, suffix):
        return "%s.%s.cache" % (self.cache_token, suffix,)

    def __init__(self, file, ignore_cache, cache_token=None):
        self.filename = file

        if cache_token is None:
            self.cache_token = "%s.%s" % (
                os.path.basename(file).replace('.', '_').replace('?', '_').replace(',', '_'),
                hashlib.sha1(str(os.path.abspath(file).lower()).encode()).hexdigest()[:8])
        else:
            self.cache_token = cache_token

        if ignore_cache == 'everything':
            self.ignore_cache = True
        elif ignore_cache == 'nothing':
            self.ignore_cache = []
        else:
            self.ignore_cache = ignore_cache.split(',')

    def __contains__(self, suffix):
        if self.ignore_cache is True or suffix in self.ignore_cache:
            return False
        else:
            return os.path.isfile(self.build_cache_filename(suffix))

    def __getitem__(self, suffix):
        with open(self.build_cache_filename(suffix), 'rb') as fp:
            self.__class__.printer("Getting data for '%s'" % (suffix,))
            return pickle.load(fp)

    def __setitem__(self, suffix, data):
        if self.ignore_cache is True or suffix in self.ignore_cache:
            return
        else:
            with open(self.build_cache_filename(suffix), 'wb+') as fp:
                self.__class__.printer("Setting data for '%s'" % (suffix,))
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


class NotReallyATree(list):
    def __init__(self, iterable):
        super(NotReallyATree, self).__init__(self)
        for i in iterable:
            self.append(i)
        self.na = numpy.array(iterable)

    def query(self, q):  # w_numpy
        distances = numpy.sqrt(numpy.sum(numpy.power(self.na - q, 2.0), 1))
        pos = numpy.argmin(distances, 0)
        return distances[pos], pos