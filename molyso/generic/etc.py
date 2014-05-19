# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function
import os
import sys
import numpy
import hashlib

from .. import Debug, Timed

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


def debug_init():
    Debug.enable("text", "plot", "plot_pdf")
    numpy.set_printoptions(threshold=numpy.nan)


def parse_range(s, allow_open_interval=True):
    ranges = []
    tail = []
    for frag in s.replace(";", ",").split(","):
        if "-" in frag:
            f, t = frag.split("-")
            if t == "":
                if allow_open_interval:
                    tail = [int(f), float("Inf")]
                else:
                    raise ValueError("parse_range called with allow_open_interval=False"
                                     " but open interval string was passed.")
            else:
                ranges += range(int(f), int(t) + 1)
        else:
            ranges += [int(frag)]
    ranges += tail
    return ranges


def hash_filename(fname):
    return hashlib.sha1(str(fname).encode()).hexdigest()


def does_pickled_hashed_exist(fname, suffix=""):
    return os.path.isfile(hash_filename(fname) + suffix)


def read_from_pickle_hashed(fname, suffix=""):
    f = hash_filename(fname) + suffix
    with Timed():
        with open(f, "rb") as fp:
            results = pickle.load(fp)
    return results


def write_to_pickle_hashed(fname, data=None, suffix=""):
    f = hash_filename(fname) + suffix
    with open(f, "wb+") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
