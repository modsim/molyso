# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

import numpy as np

from .tracking_output import s_to_h
from ..generic.etc import QuickTableDumper

try:
    # noinspection PyUnresolvedReferences
    import cPickle

    pickle = cPickle
except ImportError:
    import pickle


def interactive_ground_truth_main(args, tracked_results):
    """
    Ground truth mode entry function.

    :param args:
    :param tracked_results:
    :return: :raise SystemExit:
    """

    acceptable_pos_chans = \
        {p: list(range(len(tracked_results[list(tracked_results.keys())[p]].channel_accumulator.keys())))
         for p
         in range(len(tracked_results.keys()))
         if len(tracked_results[list(tracked_results.keys())[p]].channel_accumulator.keys()) > 0}

    def plots_info():
        """
        Outputs some information about the data set.
        """

        print("Positions " + str(list(tracked_results.keys())))
        print("Acceptable channels per position " + repr(acceptable_pos_chans))

    plots_info()

    ground_truth_data = args.ground_truth

    # noinspection PyUnresolvedReferences
    try:
        with open(ground_truth_data, 'rb') as fp:
            all_envs = pickle.load(fp)
    except FileNotFoundError:
        print("File did not exist, starting anew")
        all_envs = {}

    def save_data():
        """
        Saves the ground truth data to the file specified. (pickled data)

        """
        with open(ground_truth_data, 'wb+') as inner_fp:
            pickle.dump(all_envs, inner_fp, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved data to %s" % (ground_truth_data,))

    lowest_position = min(acceptable_pos_chans.keys())
    highest_position = max(acceptable_pos_chans.keys())

    next_dataset = [lowest_position, next(iter(acceptable_pos_chans[lowest_position]))]

    def perform_it():
        """
        Runs the ground truth mode.

        :return: :raise SystemExit:
        """
        next_pos, next_chan = next_dataset

        def empty_env():
            """
            Generates an empty environment.

            :return:
            """
            return {
                'points': np.ma.array(np.zeros((1024, 3))),
                'points_empty': np.ma.array(np.zeros((1024, 3))),
                'used': 0,
                'last_point_x': None,
                'last_point_y': None,
            }

        if (next_pos, next_chan) not in all_envs:
            all_envs[(next_pos, next_chan)] = empty_env()

        env = all_envs[(next_pos, next_chan)]

        pos = list(tracked_results.keys())[next_pos]
        tracking = tracked_results[pos]

        chan_num = list(tracking.channel_accumulator.keys())[next_chan]

        channels = tracking.channel_accumulator[chan_num]

        print("Opening position %d, channel %d" % (pos, chan_num,))

        data = np.zeros((len(channels), 6))

        n_timepoint, n_width, n_height, n_top, n_bottom, n_width_cumsum = 0, 1, 2, 3, 4, 5

        some_channel_image = None

        for n, cc in enumerate(channels):
            data[n, n_timepoint] = cc.image.timepoint
            data[n, n_width] = cc.channel_image.shape[1]
            data[n, n_height] = cc.channel_image.shape[0]
            data[n, n_top] = cc.top
            data[n, n_bottom] = cc.bottom
            some_channel_image = cc.channel_image

        data[:, n_width_cumsum] = np.cumsum(data[:, n_width])

        max_top, min_top = data[:, n_top].max(), data[:, n_top].min()
        max_bottom, min_bottom = data[:, n_bottom].max(), data[:, n_bottom].min()

        low, high = int(np.floor(min_top)), int(np.ceil(max_bottom))

        large_image = np.zeros((high - low, int(data[-1, n_width_cumsum])), dtype=some_channel_image.dtype)

        for n, cc in enumerate(channels):
            lower_border = int(np.floor(data[n, n_top] - low))
            large_image[
                lower_border:int(lower_border + data[n, n_height]),
                int(data[n, n_width_cumsum] - data[n, n_width]):int(data[n, n_width_cumsum])
            ] = cc.channel_image

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        plt.subplots_adjust(left=0.25, bottom=0.25)

        fig.canvas.set_window_title("Image Viewer")

        channels_per_inch = 5.0
        plt.rcParams['figure.figsize'] = (len(channels) / channels_per_inch, 4.0)
        plt.rcParams['figure.dpi'] = 150

        plt.rcParams['figure.subplot.top'] = 0.8
        plt.rcParams['figure.subplot.bottom'] = 0.2
        plt.rcParams['figure.subplot.left'] = 0.2
        plt.rcParams['figure.subplot.right'] = 0.8

        plt.rcParams['image.cmap'] = 'gray'

        plt.imshow(large_image)

        plt.title("Ground Truth — Position %d, channel %d" % (pos, chan_num,))

        plt.xlabel("x [Pixel]")
        plt.ylabel("y [Pixel]")

        fig.tight_layout()

        o_scatter = ax.scatter(0, 0)
        o_lines, = plt.plot(0, 0)

        def refresh():
            """
            Refreshes the overlay.

            """
            o_lines.set_data(env['points'][:env['used'], 0], env['points'][:env['used'], 1])

            o_scatter.set_offsets(env['points'][:env['used'], :2])
            fig.canvas.draw()

        def show_help():
            """
            Shows a help text for the ground truth mode.

            """
            print("""
            Ground Truth Mode:
            = Mouse =====================================
            Mark division events by right click:
            First a division, then a child's division.
            = Keys ======================================
            h       show  this help
            p       print growth rates
                    (last is based on mean division time)
            d       delete last division event
            n/N     next/previous multipoint
            m/M     next/previous channel
            o/O     output tabular data to console/file
            w       write data
                    (to previously specified filename)
            i       start interactive python console
            q       quit ground truth mode
            """)

        refresh()

        show_help()

        def click(e):
            """

            :param e:
            :return:
            """
            x, y = e.xdata, e.ydata
            if x is None or y is None:
                return
            if e.button == 3:
                last_point_x, last_point_y = env['last_point_x'], env['last_point_y']

                if last_point_x is not None:

                    if env['used'] + 3 >= env['points'].shape[0]:
                        oldmask = env['points'].mask[:env['used']]
                        env['points'] = np.ma.array(np.r_[env['points'], env['points_empty']])
                        env['points'].mask = np.zeros_like(env['points']).astype(np.dtype(bool))  # [:env['used']]
                        env['points'].mask[:env['used']] = oldmask

                    n_x = np.searchsorted(data[:, n_width_cumsum], x, side='right')
                    n_last_x = np.searchsorted(data[:, n_width_cumsum], last_point_x, side='right')

                    if x < last_point_x:
                        x, y, last_point_x, last_point_y = last_point_x, last_point_y, x, y
                        n_x, n_last_x = n_last_x, n_x

                    env['points'][env['used'], 0] = last_point_x
                    env['points'][env['used'], 1] = last_point_y
                    env['points'][env['used'], 2] = data[n_last_x, n_timepoint]
                    env['used'] += 1
                    env['points'][env['used'], 0] = x
                    env['points'][env['used'], 1] = y
                    env['points'][env['used'], 2] = data[n_x, n_timepoint]
                    env['used'] += 1
                    env['points'][env['used'], :] = np.ma.masked
                    env['used'] += 1

                    refresh()

                    # print(n, data[n, n_timepoint])
                    env['last_point_x'], env['last_point_y'] = None, None
                else:
                    env['last_point_x'], env['last_point_y'] = x, y

        def key_press(event):
            """

            :param event:
            :return: :raise SystemExit:
            """

            def show_stats():
                """
                Shows statistics.

                """
                inner_times = env['points'][:env['used'], 2].compressed()
                inner_times = inner_times.reshape(inner_times.size // 2, 2)
                inner_deltas = inner_times[:, 1] - inner_times[:, 0]
                inner_deltas /= 60.0 * 60.0
                inner_mu = np.log(2) / inner_deltas
                print(inner_mu, np.mean(inner_mu))

            def try_new_poschan(p, c):
                """

                :param p:
                :param c:
                :return:
                """

                next_pos, next_chan = next_dataset

                if p == 1:
                    while (next_pos + p) not in acceptable_pos_chans and (next_pos + p) < highest_position:
                        p += 1
                elif p == -1:
                    while (next_pos + p) not in acceptable_pos_chans and (next_pos + p) > lowest_position:
                        p -= 1

                if (next_pos + p) not in acceptable_pos_chans:
                    print("Position does not exist")
                    return

                if p != 0:
                    c = 0
                    next_chan = acceptable_pos_chans[next_pos + p][0]

                if c == 1:
                    while (next_chan + c) not in acceptable_pos_chans[next_pos + p] and \
                            (next_chan + c) < max(acceptable_pos_chans[next_pos + p]):
                        c += 1
                elif c == -1:
                    while (next_chan + c) not in acceptable_pos_chans[next_pos + p] and \
                            (next_chan + c) > min(acceptable_pos_chans[next_pos + p]):
                        c -= 1

                if (next_chan + c) not in acceptable_pos_chans[next_pos + p]:
                    print("Channel does not exist")
                    return

                next_dataset[0] = next_pos + p
                next_dataset[1] = next_chan + c

                plt.close()

            # if event.key == 'left':
            # timepoint.set_val(max(1, int(timepoint.val) - 1))
            # elif event.key == 'right':
            #     timepoint.set_val(min(tp_max, int(timepoint.val) + 1))
            # elif event.key == 'ctrl+left':
            #     timepoint.set_val(max(1, int(timepoint.val) - 10))
            # elif event.key == 'ctrl+right':
            #     timepoint.set_val(min(tp_max, int(timepoint.val) + 10))
            # elif event.key == 'down':
            #     multipoint.set_val(max(1, int(multipoint.val) - 1))
            # elif event.key == 'up':
            #     multipoint.set_val(min(mp_max, int(multipoint.val) + 1))
            if event.key == 'h':
                show_help()
            elif event.key == 'p':
                show_stats()
            elif event.key == 'd':
                env['used'] -= 3
                show_stats()
                refresh()
            # n next position, m next channel

            elif event.key == 'n':
                try_new_poschan(1, 0)
            elif event.key == 'N':
                try_new_poschan(-1, 0)
            elif event.key == 'm':
                try_new_poschan(0, 1)
            elif event.key == 'M':
                try_new_poschan(0, -1)
            elif event.key == 'o' or event.key == 'O':
                recipient = None

                if event.key == 'O':
                    print("Please enter file name for tabular output [will be overwritten if exists]:")
                    file_name = input()
                    recipient = open(file_name, 'w+')

                out = QuickTableDumper(recipient=recipient)

                for (t_pos, t_chan), t_env in all_envs.items():
                    x_pos = list(tracked_results.keys())[t_pos]
                    x_chan = list(tracked_results[x_pos].channel_accumulator.keys())[t_chan]
                    times = t_env['points'][:t_env['used'], 2].compressed()
                    times = times.reshape(times.size // 2, 2)
                    deltas = times[:, 1] - times[:, 0]
                    mu = np.log(2) / (deltas / (60.0 * 60.0))

                    for num in range(len(mu)):

                        out.add({
                            'position': x_pos,
                            'channel': x_chan,
                            'growth_rate': mu[num],
                            'growth_rate_channel_mean': np.mean(mu),
                            'division_age': s_to_h(deltas[num]),
                            'growth_start': s_to_h(times[num, 0]),
                            'growth_end': s_to_h(times[num, 1]),
                        })

                if event.key == 'O':
                    recipient.close()
                    print("File written.")

            elif event.key == 'w':
                save_data()
            elif event.key == 'i':
                import code
                code.InteractiveConsole(locals=globals()).interact()
            elif event.key == 'q':
                raise SystemExit

        fig.canvas.mpl_connect('key_press_event', key_press)
        fig.canvas.mpl_connect('button_press_event', click)

        plt.show()

    while True:
        perform_it()

