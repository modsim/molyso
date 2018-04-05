# -*- coding: utf-8 -*-
"""

"""

import os
import sys
import argparse

from ...debugging.callserialization import CallSerialization
from ...debugging.debugplot import inject_poly_drawing_helper
from matplotlib import pylab


def create_argparser():
    argparser = argparse.ArgumentParser(
        description="viewkymograph can view or convert molyso's pickled kymograph representation")

    def _error(message=''):
        argparser.print_help()
        sys.stderr.write("%serror: %s%s" % (os.linesep, message, os.linesep,))
        sys.exit(1)

    argparser.error = _error

    argparser.add_argument('input', metavar='input', type=str, help="input file")
    argparser.add_argument('-o', '--output', dest='output', type=str, default=None)

    return argparser


def main():
    argparser = create_argparser()
    args = argparser.parse_args()

    inject_poly_drawing_helper(pylab)

    with open(args.input, 'rb') as fp:
        cs = CallSerialization.from_pickle(fp.read())

    cs.execute(pylab)

    if args.output:
        pylab.savefig(args.output)
    else:
        pylab.show()


if __name__ == '__main__':
    main()
