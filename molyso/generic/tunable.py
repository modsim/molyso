from __future__ import division, unicode_literals, print_function


class TunableManager(object):
    defaults = {}

    current = {}

    force_default = False

    @classmethod
    def load_tunables(cls, data):
        cls.current = data

    @classmethod
    def get_defaults(cls):
        return cls.defaults

    @classmethod
    def get_tunable(cls, what, default):
        cls.defaults[what] = default
        if cls.force_default:
            return default
        if what in cls.current:
            return type(default)(cls.current[what])


def tunable(what, default):
    return TunableManager.get_tunable(what, default)