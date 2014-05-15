# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from .. import Debug


class DebugPlotInterruptException(Exception):
    pass


class DebugPlot(object):
    """
        The DebugPlot class serves as an switchable abstraction layer to add plotting debug output facilities.
    """

    throw_on_anything = True

    pp = None
    context = ""

    exp_plot_debugging = False  # True

    diverted_outputs = {}

    @classmethod
    def pdfopener(cls, filename):
        from matplotlib.backends.backend_pdf import PdfPages

        try:
            newoutput = PdfPages(filename)
        except IOError:
            import os

            basename, pdf = os.path.splitext(filename)
            n = 1
            while os.path.isfile("%s-%d%s" % (basename, n, pdf)):
                n += 1
                if n > 999:
                    raise Exception("Something is going horribly wrong.")
            filename = "%s-%d%s" % (basename, n, pdf)
            newoutput = PdfPages(filename)  # if it throws now, we won't care

        def close_at_exit():
            newoutput.close()

        import atexit

        atexit.register(close_at_exit)
        return newoutput

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
        self.info = ""
        if "info" in kwargs:
            self.info = kwargs["info"]
        self.filter_okay = Debug.filter(*args)
        self.filter_str = Debug.filter_to_str(args)

        self.active = self.filter_okay and (Debug.is_enabled("plot") or Debug.is_enabled("plot_pdf"))

        if self.active:
            import pylab

            self.pylab = pylab

        if Debug.is_enabled("plot_pdf"):
            if DebugPlot.pp is None:
                DebugPlot.pp = self.__class__.pdfopener('debug.pdf')


    def __getattr__(self, item):
        if self.active:
            if hasattr(self.pylab, item):
                if self.__class__.exp_plot_debugging:
                    def proxy(*args, **kwargs):
                        import sys

                        print("pylab.%s(%s%s%s)" % (
                            item, ",".join([repr(a) for a in args]), "," if len(kwargs) > 0 else "",
                            ",".join(["%s=%s" % (a, repr(b)) for a, b in kwargs.items()])), file=sys.stderr)
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
                def proxy(*args, **kwargs):
                    pass

        return proxy

    def poly_drawing_helper(self, coords, **kwargs):
        gca = self.gca()
        if gca:
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch

            actions = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 1)
            gca.add_patch(PathPatch(Path(coords, actions), **kwargs))

    def __enter__(self):
        if self.active:
            self.rcParams = self.pylab.rcParams
            self.clf()
            self.close()
            self.rcParams['figure.figsize'] = (12, 8)
            self.rcParams['figure.dpi'] = 150
            self.rcParams['image.cmap'] = 'gray'
            self.figure()
            self.text(0.01, 0.01, "%s\n%s\n%s" % (self.info, Debug.get_context(), self.filter_str),
                      transform=self.gca().transAxes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == DebugPlotInterruptException:
            return True
        if self.active:
            if DebugPlot.pp:
                self.savefig(DebugPlot.pp, format='pdf')
            for pp, okay in DebugPlot.diverted_outputs.items():
                if self.filter_str in okay:
                    self.savefig(pp, format='pdf')
            self.clf()
            self.close()
            self.figure()