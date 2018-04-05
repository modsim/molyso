# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import os
import atexit

from os.path import isfile
from tempfile import TemporaryFile
from functools import partial

from .callserialization import CallSerialization


def next_free_filename(prefix, suffix):
    """

    :param prefix:
    :param suffix:
    :return: :raise IOError:
    """
    n = 0
    while isfile(prefix + '%04d' % (n,) + suffix):
        n += 1
        if n > 9999:
            raise IOError('No free filename found.')
    return prefix + '%04d' % (n,) + suffix


def poly_drawing_helper(p, coordinates, **kwargs):
    """

    :param p:
    :param coordinates:
    :param kwargs:
    """
    gca = p.gca()

    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    actions = [Path.MOVETO] + [Path.LINETO] * (len(coordinates) - 1)

    if 'closed' in kwargs:
        if kwargs['closed']:
            actions.append(Path.CLOSEPOLY)
            coordinates.append((0, 0))
        del kwargs['closed']

    gca.add_patch(PathPatch(Path(coordinates, actions), **kwargs))


def inject_poly_drawing_helper(p):
    """

    :param p:
    """
    p.poly_drawing_helper = partial(poly_drawing_helper, p)


class DebugPlotInterruptException(Exception):
    """
    Only for internal usage.
    Used to interrupt plot drawing early if it is disabled.
    """
    pass


class DebugPlotInterruptThrower(object):
    """
    Dummy object which raises an exception
    on every call. To be used when debug mode is deactivated.
    """
    def __getattr__(self, item):
        raise DebugPlotInterruptException()


class DebugPlot(object):
    """
        The DebugPlot class serves as an switchable abstraction layer to add plotting debug output facilities.
    """

    default_config = {
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'image.cmap': 'gray'
    }

    file_prefix = 'debug'

    active = True
    force_active = False

    post_figure = 'close'

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
    def set_context(cls, **kwargs):
        """

        :param kwargs:
        """
        cls.context = ' '.join(["%s=%s" % x for x in sorted(kwargs.items())])

    @classmethod
    def get_context(cls):
        """


        :return:
        """
        return cls.context

    @classmethod
    def pdfopener(cls, filename):
        """
        Opens a new PdfPages output, ensuring it will be closed at exit.

        :param filename: filename
        :return:
        """
        from matplotlib.backends.backend_pdf import PdfPages

        try:
            newoutput = PdfPages(filename)
        except IOError:
            basename, pdf = os.path.splitext(filename)

            filename = next_free_filename(basename, pdf)
            newoutput = PdfPages(filename)  # if it throws now, we won't care

        def _close_at_exit():
            try:
                newoutput.close()
            except AttributeError:
                pass

        cls.exit_handlers.append(_close_at_exit)

        return newoutput

    @classmethod
    def _call_exit_handlers(cls):
        # perform cleanup, either explicitly,
        # or by atexit at the end  (__init__ registers this function)
        """


        """
        for handler in cls.exit_handlers:
            handler()

        cls.exit_handlers = []
        cls.exit_handler_registered = False

    @classmethod
    def new_pdf_output(cls, filename, collected):
        """

        :param filename:
        :param collected:
        """
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

        # Note: DebugPlot had a sister class 'Debug' which took care of filtering
        # this was however not used in molyso. In a future rewrite, DebugPlot might be attached
        # more to the Python included log system / its filter capabilities...

        # self.filter_okay = Debug.filter(*args)
        self.filter_str = '.'.join([w.lower() for w in args])  # Debug.filter_to_str(args)

        self.active = self.force_active
        # or (self.active and
        # self.filter_okay and
        # (Debug.is_enabled('plot') or Debug.is_enabled('plot_pdf')))

        if not DebugPlot.exit_handler_registered:
            atexit.register(DebugPlot._call_exit_handlers)

        if self.active:
            from matplotlib import pylab

            self.pylab = pylab

        self.call_serialization = CallSerialization()

        if DebugPlot.individual_and_merge:
            try:
                # noinspection PyPackageRequirements,PyUnresolvedReferences
                import PyPDF2
                DebugPlot.individual_files = True
            except ImportError:
                DebugPlot.individual_and_merge = False

        # if Debug.is_enabled('plot_pdf'):
        if self.active:
            if not DebugPlot.individual_files:
                if DebugPlot.pp is None:
                    DebugPlot.pp = self.__class__.pdfopener('debug.pdf')

    def __enter__(self):
        # if self.active:
        #     # noinspection PyPep8Naming,PyAttributeOutsideInit

        if self.active:
            return self.call_serialization.get_proxy()
        else:
            return DebugPlotInterruptThrower()

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == DebugPlotInterruptException:
            return True

        if self.active:

            p = self.pylab

            p.rcParams.update(self.default_config)
            p.figure()
            p.text(0.01, 0.01, "%s\n%s\n%s" % (self.info, self.get_context(), self.filter_str),
                   transform=p.gca().transAxes)

            inject_poly_drawing_helper(p)

            self.call_serialization.execute(p)

            if not DebugPlot.individual_files:
                if DebugPlot.pp:
                    p.savefig(DebugPlot.pp, format='pdf')
            else:
                if DebugPlot.individual_and_merge:
                    p.savefig(
                        self.get_file_for_merge(),
                        format='pdf')
                else:
                    p.savefig(
                        next_free_filename(DebugPlot.individual_file_prefix, DebugPlot.file_suffix),
                        format='pdf')

            for pp, okay in DebugPlot.diverted_outputs.items():
                if self.filter_str in okay:
                    p.savefig(pp, format='pdf')

            if self.post_figure == 'show':
                p.show()
            else:
                p.close()

    @classmethod
    def get_file_for_merge(cls):
        """


        :return:
        """
        if len(cls.files_to_merge) == 0:
            def _merge_exit_handler():
                # noinspection PyPackageRequirements,PyUnresolvedReferences
                import PyPDF2

                with open(next_free_filename(cls.file_prefix, cls.file_suffix), 'wb+') as pdf_file:

                    pdf = PyPDF2.PdfFileMerger()

                    for individual_file in cls.files_to_merge:
                        individual_file.seek(0)
                        individual_pdf = PyPDF2.PdfFileReader(individual_file)

                        pdf.append(individual_pdf)

                    pdf.write(pdf_file)

                cls.files_to_merge = []
            cls.exit_handlers.append(_merge_exit_handler)

        f = TemporaryFile()
        cls.files_to_merge.append(f)
        return f
