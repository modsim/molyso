# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from .. import Debug

import time
import sys
import os


class Timed(object):
    def __init__(self):
        self.ticks = None
        self.start = None
        self.last = None
        self.what = None
        self.quiet = None
        self.stop = None

    def __enter__(self, what=""):
        self.ticks = []
        self.start = time.time()
        self.last = time.time()
        self.what = what
        self.quiet = Debug.is_enabled("microbenchmarks")
        return self

    def tick(self, what=""):
        self.ticks += [(time.time(), self.last, what)]
        self.last = time.time()

    def elapsed(self):
        return time.time() - self.start

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = time.time()
        total = self.stop - self.start
        if not self.quiet:
            if self.what != "":
                sys.stderr.write("[%s]: " % (self.what,))
            sys.stderr.write("Took %.4fs " % (total,) + os.linesep)
            for step, newstop, what in self.ticks:
                temp = step - newstop
                sys.stderr.write("\t%.4fs (%2.2f%%) %s" % (temp, 100.0 * (temp / total), what) + os.linesep)
