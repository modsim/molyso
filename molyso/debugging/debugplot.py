# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import sys
import atexit
from os.path import isfile

from tempfile import TemporaryFile

from . import Debug

def next_free_filename(prefix, suffix):
    n = 0
    while isfile(prefix + '%04d' % (n,) + suffix):
        n += 1
        if n > 9999:
            raise IOError('No free filename found.')
    return prefix + '%04d' % (n,) + suffix

def poly_drawing_helper(p, coords, **kwargs):
    gca = p.gca()

    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    actions = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 1)

    if 'closed' in kwargs:
        if kwargs['closed']:
            actions.append(Path.CLOSEPOLY)
            coords.append((0, 0))
        del kwargs['closed']

    gca.add_patch(PathPatch(Path(coords, actions), **kwargs))


class DebugPlotInterruptException(Exception):
    pass


class DebugPlot(object):
    """
        The DebugPlot class serves as an switchable abstraction layer to add plotting debug output facilities.
    """

    file_prefix = 'debug'

    active = True

    individual_and_merge = False
    individual_files = False
    individual_file_prefix = file_prefix

    file_suffix = '.pdf'

    throw_on_anything = True

    pp = None
    context = ''

    exp_plot_debugging = False  # True

    diverted_outputs = {}

    files_to_merge = []

    exit_handlers = []

    exit_handler_registered = False

    @classmethod
    def pdfopener(cls, filename):
        from matplotlib.backends.backend_pdf import PdfPages

        try:
            newoutput = PdfPages(filename)
        except IOError:
            basename, pdf = os.path.splitext(filename)

            filename = next_free_filename(basename, pdf)
            newoutput = PdfPages(filename)  # if it throws now, we won't care

        def close_at_exit():
            try:
                newoutput.close()
            except AttributeError:
                pass

        cls.exit_handlers.append(close_at_exit)

        return newoutput

    @classmethod
    def call_exit_handlers(cls):
        # perform cleanup, either explicitly,
        # or by atexit at the end  (__ini__ registers this function)
        for handler in cls.exit_handlers:
            handler()

        cls.exit_handlers = []
        cls.exit_handler_registered = False

    @classmethod
    def new_pdf_output(cls, filename, collected):
        newoutput = cls.pdfopener(filename)
        cls.diverted_outputs[newoutput] = collected

    def __init__(self, *args, **kwargs):
        """
        Creates an DebugPlot instance
        :param info: An additional information about the plot, currently shown in the title
        :return:
        """
        self.info = ''
        if 'info' in kwargs:
            self.info = kwargs['info']
        self.filter_okay = Debug.filter(*args)
        self.filter_str = Debug.filter_to_str(args)

        self.active = self.__class__.active and \
                      self.filter_okay and\
                      (Debug.is_enabled('plot') or Debug.is_enabled('plot_pdf'))

        if not DebugPlot.exit_handler_registered:
            atexit.register(DebugPlot.call_exit_handlers)

        if self.active:
            from matplotlib import pylab

            self.pylab = pylab

        if DebugPlot.individual_and_merge:
            try:
                import PyPDF2
                DebugPlot.individual_files = True
            except ImportError:
                DebugPlot.individual_and_merge = False

        if Debug.is_enabled('plot_pdf'):
            if not DebugPlot.individual_files:
                if DebugPlot.pp is None:
                    DebugPlot.pp = self.__class__.pdfopener('debug.pdf')

    def __getattr__(self, item):
        if self.active:
            if hasattr(self.pylab, item):
                if self.__class__.exp_plot_debugging:
                    def proxy(*args, **kwargs):
                        print("pylab.%s(%s%s%s)" % (
                            item, ','.join([repr(a) for a in args]), ',' if len(kwargs) > 0 else '',
                            ','.join(["%s=%s" % (a, repr(b)) for a, b in kwargs.items()])), file=sys.stderr)
                        return getattr(self.pylab, item)(*args, **kwargs)
                else:
                    def proxy(*args, **kwargs):
                        return getattr(self.pylab, item)(*args, **kwargs)
            else:
                raise NameError("name '%s' is not defined" % item)
        else:
            if self.__class__.throw_on_anything:
                raise DebugPlotInterruptException()
            else:
                # noinspection PyUnusedLocal
                def proxy(*args, **kwargs):
                    pass

        return proxy

    def poly_drawing_helper(self, coords, **kwargs):
        if self.gca():
            return poly_drawing_helper(self, coords, **kwargs)

    def __enter__(self):
        if self.active:
            # noinspection PyPep8Naming,PyAttributeOutsideInit
            self.rcParams = self.pylab.rcParams
            self.clf()
            self.close('all')
            self.rcParams['figure.figsize'] = (12, 8)
            self.rcParams['figure.dpi'] = 150
            self.rcParams['image.cmap'] = 'gray'
            self.figure()
            self.text(0.01, 0.01, "%s\n%s\n%s" % (self.info, Debug.get_context(), self.filter_str),
                      transform=self.gca().transAxes)
        return self

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == DebugPlotInterruptException:
            return True
        if self.active:
            if not DebugPlot.individual_files:
                if DebugPlot.pp:
                    self.savefig(DebugPlot.pp, format='pdf')
            else:
                if DebugPlot.individual_and_merge:
                    self.savefig(self.get_file_for_merge(), format='pdf')
                else:
                    self.savefig(next_free_filename(DebugPlot.individual_file_prefix, DebugPlot.file_suffix), format='pdf')

            for pp, okay in DebugPlot.diverted_outputs.items():
                if self.filter_str in okay:
                    self.savefig(pp, format='pdf')
            self.clf()
            self.close('all')
            self.figure()

    def get_file_for_merge(self):
        if len(DebugPlot.files_to_merge) == 0:
            def _merge_exit_handler():
                import PyPDF2

                with open(next_free_filename(DebugPlot.file_prefix, DebugPlot.file_suffix), 'wb+') as pdf_file:

                    pdf = PyPDF2.PdfFileMerger()

                    for individual_file in DebugPlot.files_to_merge:
                        individual_file.seek(0)
                        individual_pdf = PyPDF2.PdfFileReader(individual_file)

                        pdf.append(individual_pdf)

                    pdf.write(pdf_file)

                DebugPlot.files_to_merge = []
            DebugPlot.exit_handlers.append(_merge_exit_handler)

        f = TemporaryFile()
        DebugPlot.files_to_merge.append(f)
        return f