# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

from ..imageio.imagestack import MultiImageStack
from ..imageio.imagestack_ometiff import OMETiffStack
from .image import Image


OMETiffStack = OMETiffStack


def interactive_main(args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from .image import cell_color, channel_color

    ims = MultiImageStack.open(args.input, treat_z_as_mp=args.zm)

    mp_max = ims.get_meta('multipoints')
    tp_max = ims.get_meta('timepoints')

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0.25, bottom=0.25)

    fig.canvas.set_window_title("Image Viewer")

    ax_mp = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=channel_color)
    ax_tp = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=channel_color)

    multipoint = Slider(ax_mp, 'Multipoint', 1, mp_max, valinit=1, valfmt="%d", color=cell_color)
    timepoint = Slider(ax_tp, 'Timepoint', 1, tp_max, valinit=1, valfmt="%d", color=cell_color)


    def update(_):
        t = int(timepoint.val)
        pos = int(multipoint.val)

        fig.canvas.set_window_title("Image Viewer - [BUSY]")

        img = ims.get_image(t=t - 1, pos=pos - 1, channel=0)
        i = Image()
        i.setup_image(img)
        i.autorotate()
        i.find_channels()
        i.find_cells_in_channels()

        def pdh(coords, **kwargs):
            from ..debugging.debugplot import poly_drawing_helper as _pdh

            return _pdh(plt, coords, **kwargs)

        plt.poly_drawing_helper = pdh

        plt.rcParams['image.cmap'] = 'gray'

        plt.sca(ax)
        plt.cla()

        i.debug_print_cells(plt)

        fig.canvas.set_window_title("Image Viewer - %s timepoint %d/%d multipoint %d/%d" %
                                    (args.input, t, tp_max, pos, mp_max))

        plt.draw()

    update(None)

    multipoint.on_changed(update)
    timepoint.on_changed(update)

    def key_press(event):
        if event.key == 'left':
            timepoint.set_val(max(1, int(timepoint.val) - 1))
        elif event.key == 'right':
            timepoint.set_val(min(tp_max, int(timepoint.val) + 1))
        elif event.key == 'ctrl+left':
            timepoint.set_val(max(1, int(timepoint.val) - 10))
        elif event.key == 'ctrl+right':
            timepoint.set_val(min(tp_max, int(timepoint.val) + 10))
        elif event.key == 'down':
            multipoint.set_val(max(1, int(multipoint.val) - 1))
        elif event.key == 'up':
            multipoint.set_val(min(mp_max, int(multipoint.val) + 1))
        elif event.key == 'q':
            raise SystemExit

    fig.canvas.mpl_connect('key_press_event', key_press)

    fig.tight_layout()

    plt.show()