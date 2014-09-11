# -*- coding: utf-8 -*-

"""
tunable.py contains a tunable (settings to be changed depending on input data) management class
"""
from __future__ import division, unicode_literals, print_function


class TunableManager(object):
    """
    TunableManager

    Static object handling tunables (that is, parameters used during various processing step, which the user might
    want to change)

    Tunables have default values, which must be set as a paremter with the function for asking a tunable.
    That way, default configuration is inlined, and does not need to be centrally managed.
    In order to collect all defaults, a typical run of the program has to be performed, and the collected default
    values to be dumped afterwards.
    """

    defaults = {}

    current = {}

    force_default = False

    @classmethod
    def load_tunables(cls, data):
        """
        Sets the tunables.

        :param data: set of tunables to load
        :type data: dict
        :rtype: None
        """
        cls.current = data

    @classmethod
    def get_defaults(cls):
        """
        Gets the defaults, which were collected during the calls asking for various tunables.

        :param what: tunable to look up
        :type what: str
        :return: either the overridden tunable or the default value
        :rtype: dependent on default
        """
        return cls.defaults

    @classmethod
    def get_tunable(cls, what, default):
        """
        Returns either an overriden tunable, or the default value.
        The result will be casted to the type of default.

        :param what: tunable to look up
        :type what: str
        :param default: default value
        :return: either the overridden tunable or the default value
        """
        cls.defaults[what] = default
        if cls.force_default:
            return default
        if what in cls.current:
            return type(default)(cls.current[what])
        else:
            return default


def tunable(what, default):
    """
    syntactic sugar helper function, to quickly get a tunable

    :param what: tunable to look up
    :type what: str
    :param default: default value
    :return: either the overridden tunable or the default value
    :rtype: dependent on default
    """
    return TunableManager.get_tunable(what, default)