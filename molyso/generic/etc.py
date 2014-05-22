# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function
import os
import sys
import numpy
import hashlib
import time

from .. import Debug, Timed

def silent_progress_bar(iterable):
    return iterable

try:
    import clint.textui

    def fancy_progress_bar(iterable):
        return clint.textui.progress.bar(iterable, width=50)

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

progress_bar = fancy_progress_bar

class QuickTableDumper(object):
    def __init__(self, recipient=None):
        if recipient is None:
            recipient = sys.stdout

        self.recipient = recipient
        self.headers = []

        self.delim = "\t"
        self.lineend = "\n"
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

if os.name != "nt":
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
        if "-" in frag:
            f, t = frag.split("-")
            if t == "":
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


class Cache(object):
    ignore_cache = True
    printer = print

    @classmethod
    def build_cache_filename(cls, file, suffix):
        return "%s.%s.%s.cache" % (
            os.path.basename(file).replace('.', '_'),
            hashlib.sha1(str(os.path.abspath(file).lower()).encode()).hexdigest()[:8],
            suffix,)

    def __init__(self, file):
        self.filename = file

    def __contains__(self, suffix):
        if self.__class__.ignore_cache:
            return False
        else:
            return os.path.isfile(self.__class__.build_cache_filename(self.filename, suffix))

    def __getitem__(self, suffix):
        with open(self.__class__.build_cache_filename(self.filename, suffix), 'rb') as fp:
            self.__class__.printer("Getting")
            return pickle.load(fp)

    def __setitem__(self, suffix, data):
        if self.__class__.ignore_cache:
            return
        else:
            with open(self.__class__.build_cache_filename(self.filename, suffix), 'wb+') as fp:
                self.__class__.printer("Setting")
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
