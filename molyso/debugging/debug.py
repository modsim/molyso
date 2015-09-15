# -*- coding: utf-8 -*-
"""
documentation for debug
"""
from __future__ import division, unicode_literals, print_function


class Debug(object):
    enabled = {}
    context = ''

    default_filter = True
    filter_map = {}

    @classmethod
    def enable(cls, *what):
        for w in what:
            cls.enabled[w] = True

    @classmethod
    def disable(cls, *what):
        for w in what:
            cls.enabled[w] = True

    @classmethod
    def is_enabled(cls, what):
        try:
            return cls.enabled[what]
        except KeyError:
            return False

    @classmethod
    def set_context(cls, **kwargs):
        cls.context = ' '.join(["%s=%s" % x for x in kwargs.items()])

    @classmethod
    def get_context(cls):
        return cls.context

    @staticmethod
    def filter_to_str(what):
        return '.'.join([w.lower() for w in what])

    @classmethod
    def add_filter(cls, *what):
        what = list(what)
        last = what.pop()
        s = cls.filter_to_str(what)
        cls.filter_map[s] = last

    @classmethod
    def filter(cls, *what):
        what = list(what)
        for _ in range(len(what)):
            s = cls.filter_to_str(what)
            if s in cls.filter_map:
                return cls.filter_map[s]
            what.pop()
        return cls.default_filter
