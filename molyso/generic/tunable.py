# -*- coding: utf-8 -*-

"""
tunable.py contains a tunable (settings to be changed depending on input data) management class
"""
from __future__ import division, unicode_literals, print_function

import logging


class TunableManager(object):
    """
    TunableManager

    Static object handling tunables (that is, parameters which are user-changeable)

    Tunables have default values, which must be set as a parameter by the function asking for the tunable.
    That way, default configuration is inlined, and does not need to be centrally managed.
    In order to collect all defaults, a typical run of the program has to be performed, and the collected default
    values to be dumped afterwards.

    :cvar defaults: Internal map of defaults
    :vartype defaults: dict
    :cvar current: Current tunables, with possible overrides.
    :vartype current: dict
    :cvar force_default: Whether to force usage of default values (default: `False`)
    :vartype force_default: bool


    """

    defaults = {}

    current = {}

    descriptions = {}

    force_default = False

    logger = logging.getLogger(__name__ + '.' + 'TunableManager')

    @classmethod
    def set_description(cls, what, description):
        """
        Sets a description for a paremeter.

        :param what: parameter to describe
        :param description: description
        :return:
        """
        cls.descriptions[what] = description

    @classmethod
    def get_descriptions(cls):
        """
        Gets descriptions.

        :return: The descriptions.
        :rtype: dict
        """
        return cls.descriptions

    @classmethod
    def get_table(cls):
        descriptions = cls.get_descriptions()

        return [
            {
                'name': k,
                'default': v,
                'type_': type(v).__name__,
                'description': descriptions[k] if k in descriptions else ''
            }
            for k, v in cls.get_defaults().items()
        ]

    @classmethod
    def load_tunables(cls, data):
        """
        Sets the tunables.

        :param data: set of tunables to load
        :type data: dict
        :rtype: None

        >>> TunableManager.load_tunables({'foo': 'bar'})
        >>> tunable('foo', 'not bar')
        'bar'
        """
        cls.current = data

    @classmethod
    def get_defaults(cls):
        """
        Gets the defaults, which were collected during the calls asking for various tunables.

        :return: either the overridden tunable or the default value
        :rtype: dependent on default

        >>> TunableManager.defaults = {}
        >>> value = tunable('my.tunable', 3.1415)
        >>> TunableManager.get_defaults()
        {'my.tunable': 3.1415}
        """
        return cls.defaults

    @classmethod
    def get_tunable(cls, what, default):
        """
        Returns either an overridden tunable, or the default value.
        The result will be casted to the type of default.

        :param what: tunable to look up
        :type what: str
        :param default: default value
        :return: either the overridden tunable or the default value

        >>> tunable('my.tunable', 3.1415)
        3.1415
        """
        cls.defaults[what] = default
        if cls.force_default or what not in cls.current:
            result = default

            if cls.force_default:
                cls.logger.debug("Getting tunable \"%s\", forcing default: %s", what, repr(result))
            else:
                cls.logger.debug("Getting tunable \"%s\", using default: %s", what, repr(result))

        else:
            result = type(default)(cls.current[what])
            cls.logger.debug("Getting tunable \"%s\", using override: %s", what, repr(result))

        return result


def tunable(what, default, description=None):
    """
    Syntactic sugar helper function, to quickly get a tunable.
    Calls: :code:`TunableManager.get_tunable(what, default)`

    :param what: tunable to look up
    :type what: str or unicode
    :param default: default value
    :param description: description
    :return: either the overridden tunable or the default value
    :rtype: type(default)

    >>> tunable('my.tunable', 3.1415)
    3.1415
    """

    if description:
        TunableManager.set_description(what, description)

    return TunableManager.get_tunable(what, default)
