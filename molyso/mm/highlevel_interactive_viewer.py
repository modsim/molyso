# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

import warnings
from ..debugging.debugplot import inject_poly_drawing_helper


def interactive_main(args):
    """

    :param args:
    :raise SystemExit:
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from .image import cell_color, channel_color

    from .highlevel import processing_frame, processing_setup

    processing_setup(args)

    from .highlevel import ims, Dimensions

    mp_max = ims.size[Dimensions.PositionXY] - 1
    tp_max = ims.size[Dimensions.Time] - 1

    fluor_chan = list(range(ims.size[Dimensions.Channel] - 1))

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0.25, bottom=0.25)

    fig.canvas.set_window_title("Image Viewer")

    ax_mp = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=channel_color)
    ax_tp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=channel_color)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        multipoint = Slider(ax_mp, 'Multipoint', 0, mp_max, valinit=0, valfmt="%d", color=cell_color)
        timepoint = Slider(ax_tp, 'Timepoint', 0, tp_max, valinit=0, valfmt="%d", color=cell_color)

    env = {'show': True, 'rotated': True, 'fluor_ind': False}

    def update(_):
        """

        :param _:
        """
        t = int(timepoint.val)
        pos = int(multipoint.val)

        fig.canvas.set_window_title("Image Viewer - [BUSY]")

        inject_poly_drawing_helper(plt)

        plt.rcParams['image.cmap'] = 'gray'

        plt.sca(ax)
        plt.cla()

        plt.suptitle("[left/right] timepoint [up/down] multipoint [h] hide analysis [r] toggle rotated (in raw mode)")

        plt.xlabel("x [Pixel]")
        plt.ylabel("y [Pixel]")

        i = processing_frame(args, t, pos, clean=False)

        if env['fluor_ind'] is not False:
            if env['show']:
                i.debug_print_cells(plt)
            plt.title("Fluorescence Image (Fluorescence channel #%d)" % (env['fluor_ind'],))
            mapping = plt.imshow(i.image_fluorescences[env['fluor_ind']])
        elif env['show']:
            i.debug_print_cells(plt)
        else:
            if env['rotated']:
                plt.title("Image (rotated)")
                plt.imshow(i.image)
            else:
                plt.title("Image (raw)")
                plt.imshow(i.original_image)

        fig.canvas.set_window_title("Image Viewer - %s timepoint #%d %d/%d multipoint #%d %d/%d" %
                                    (args.input, t, 1+t, 1+tp_max, pos, 1+pos, 1+mp_max))

        plt.draw()

    update(None)

    multipoint.on_changed(update)
    timepoint.on_changed(update)

    def key_press(event):
        """

        :param event:
        :raise SystemExit:
        """
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
        elif event.key == 'h':
            env['show'] = not env['show']
            update(None)
        elif event.key == 'r':
            env['rotated'] = not env['rotated']
            update(None)
        elif event.key == 'F':
            if len(fluor_chan) > 0:
                if env['fluor_ind'] is False:
                    env['fluor_ind'] = fluor_chan[0]
                else:
                    env['fluor_ind'] = False if env['fluor_ind'] == len(fluor_chan) - 1 else current + 1
                update(None)
        elif event.key == 'q':
            raise SystemExit

    fig.canvas.mpl_connect('key_press_event', key_press)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig.tight_layout()

    plt.show()
