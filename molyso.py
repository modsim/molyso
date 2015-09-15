#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import os
import inspect
import sys

if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'molyso')):
    import molyso.mm.highlevel
else:
    sys.path = list(reversed(sys.path))
    import molyso.mm.highlevel

    sys.path = list(reversed(sys.path))

if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    molyso.mm.highlevel.main()
