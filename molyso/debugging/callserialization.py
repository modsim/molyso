# -*- coding: utf-8 -*-
"""
Some experimental code which helps in serializing calls (e.g. drawing instructions for matplotlib),
either for deferred rendering or storage with subsequent later visualization.

This code might be spun off into a distinct library, as it has probably wider applicability.
"""

import pickle
from collections import namedtuple

try:
    import jsonpickle
except ImportError:
    jsonpickle = None


class Step(namedtuple('Step', ['num', 'parent', 'call', 'name', 'args', 'kwargs', 'overrides'])):
    @property
    def formatted(self):
        if self.call:
            return 'results[%d] = results[%d](%s)' % (
                self.num,
                self.parent,
                ', '.join(
                    [repr(arg) for arg in self.args] +
                    ['%s=%s' % (k, repr(v)) for k, v in self.kwargs.items()] +
                    ['%s=results[%d]' % (str(k), v,) for k, v in self.overrides.items()]
                ))
        else:
            return 'results[%d] = results[%d].%s' % (self.num, self.parent, self.name,)


class Proxy(object):
    def __init__(self, cs, parent):
        self.cs = cs
        self.parent = parent

    def __getattr__(self, item):
        return Proxy(
            self.cs,
            self.cs.add_step(Step(-1, self.parent, False, item, [], {}, {}))
        )

    def __getitem__(self, item):
        return self.__getattr__('__getitem__').__call__(item)

    def __call__(self, *args, **kwargs):
        overrides = {}
        args = list(args)

        for n, arg in enumerate(args):
            if isinstance(arg, self.__class__):
                args[n] = None
                overrides[n] = arg.parent

        for key, value in list(kwargs.items()):
            if isinstance(value, self.__class__):
                del kwargs[key]
                overrides[key] = value.parent

        return Proxy(
            self.cs,
            self.cs.add_step(Step(-1, self.parent, True, '', args, kwargs, overrides))
        )


class CallSerialization(object):
    __slots__ = ['steps']

    @property
    def formatted(self):
        steps = self.steps
        return 'results = [None] * %d\n' % (len(steps)) + '\n'.join(
            step.formatted if step else 'results[0] = root'
            for step in steps
        )

    @classmethod
    def from_pickle(cls, data):
        return pickle.loads(data)

    @classmethod
    def from_json(cls, data):
        if jsonpickle:
            return jsonpickle.loads(data)

    @property
    def as_pickle(self):
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def as_json(self):
        if jsonpickle:
            return jsonpickle.dumps(self)

    def __init__(self):
        self.steps = [None]

    def add_step(self, what):
        num = len(self.steps)
        # noinspection PyProtectedMember
        self.steps.append(what._replace(num=num))
        return num

    def get_proxy(self):
        return Proxy(self, 0)

    def execute(self, root):

        results = [None] * len(self.steps)

        results[0] = root

        for step in self.steps[1:]:
            if step.call:
                args = step.args
                kwargs = step.kwargs

                for k, v in step.overrides.items():
                    if isinstance(k, int):
                        args[k] = results[v]
                    else:
                        kwargs[k] = results[v]

                results[step.num] = results[step.parent](*args, **kwargs)
            else:
                results[step.num] = getattr(results[step.parent], step.name)
