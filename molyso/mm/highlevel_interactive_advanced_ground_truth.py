# -*- coding: utf-8 -*-
"""
documentation
"""

from __future__ import division, unicode_literals, print_function

import numpy as np
import time

from .tracking_output import s_to_h
from ..generic.etc import QuickTableDumper

from .fluorescence import FluorescentChannel

import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()


class PolyLinesManager:
    double_click_timeout = 0.250

    def __getstate__(self):
        dict_copy = self.__dict__.copy()
        dict_copy['figure'] = None
        dict_copy['update_callback'] = None
        return dict_copy

    def __setstate__(self, state):
        self.__dict__ = state

    def __init__(self, figure, line_segments=[]):
        self.figure = figure

        self.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.figure.canvas.mpl_connect('button_release_event', self.on_button_release)

        self.mouse_move_handler = None

        self.artist_mapping = {}
        self.line_segments = []

        self.selected_artist = None
        self.selected_point = None
        self.last_click = 0.0

        self.update_callback = None

        for ls in line_segments:
            self.add(ls)

    def add(self, *lss):
        for ls in lss:
            ls.plot = None
            self.line_segments.append(ls)

    def delete(self, *lss):
        for ls in lss:
            if ls in self.line_segments:
                self.line_segments.remove(ls)

                if ls.plot:
                    ls.plot.remove()
                    if ls.plot in self.artist_mapping:
                        del self.artist_mapping[ls.plot]

    def draw(self, ax):
        self.artist_mapping.clear()
        for ls in self.line_segments:
            ls.draw(ax)
            self.artist_mapping[ls.plot] = ls

    def on_pick(self, event):
        if event.artist in self.artist_mapping:
            mouse_coords = np.array([event.mouseevent.xdata, event.mouseevent.ydata])

            ls = self.artist_mapping[event.artist]
            distances = np.sqrt(((ls.points - mouse_coords)**2).sum(axis=1))

            n_th_point = np.argmin(distances)

            distances[n_th_point] = np.inf

            n2_th_point = np.argmin(distances)

            now = time.time()

            if (now - self.last_click) > self.double_click_timeout:
                self.last_click = now

                self.selected_artist = ls
                self.selected_point = n_th_point

                self.track_mouse()
            else:
                self.last_click = now
                ls.insert(min(n_th_point, n2_th_point), max(n_th_point, n2_th_point), mouse_coords, figure=event.canvas.figure)

    def track_mouse(self):
        self.untrack_mouse()
        self.mouse_move_handler = self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def untrack_mouse(self):
        if self.mouse_move_handler:
            self.figure.canvas.mpl_disconnect(self.mouse_move_handler)
            self.mouse_move_handler = None

    def on_mouse_move(self, event):
        if event.inaxes:
            self.selected_artist.update(n=self.selected_point, xy=(event.xdata, event.ydata), figure=self.figure)

    def on_button_release(self, event):
        if self.selected_artist:
            self.selected_artist = None
            self.untrack_mouse()
            # some more?
        if self.update_callback:
            self.update_callback()


class PolyLine:

    def __getstate__(self):
        dict_copy = self.__dict__.copy()
        dict_copy['plot'] = None
        return dict_copy

    def __setstate__(self, state):
        self.__dict__ = state

    def __init__(self, points, closed=False):
        self.points = np.array(points)
        closed = False  # not working yet
        if closed:
            self.points = np.r_[self.points, [self.points[0]]]

        self.closed = closed
        self.plot = None
        self.plot_kwargs = dict(marker='o', picker=5)

    def draw(self, ax):
        if self.plot is None:
            self.plot, = ax.plot(self.points[:, 0], self.points[:, 1], **self.plot_kwargs)

    def redraw(self, figure=None):
        self.plot.set_xdata(self.points[:, 0])
        self.plot.set_ydata(self.points[:, 1])

        if figure:
            figure.canvas.draw_idle()

    def update(self, n=0, xy=(0.0, 0.0), figure=None):
        self.points[n] = xy

        if self.closed:
            if n == 0:
                self.points[-1] = self.points[0]
            elif n == len(self.points) - 1:
                self.points[0] = self.points[-1]

        self.redraw(figure)

    def insert_relative(self, lo=0, hi=1, relative=0.5, figure=None):

        lo_point, hi_point = self.points[lo], self.points[hi]

        xy = ((hi_point - lo_point) * relative + lo_point)

        self.points = np.r_[
            self.points[:lo+1],
            [xy],
            self.points[hi:]
        ]
        self.redraw(figure)

    def insert(self, lo=0, hi=1, xy=None, figure=None):
        self.points = np.r_[
            self.points[:lo+1],
            [xy],
            self.points[hi:]
        ]

        self.redraw(figure)


class PairedPolyLine(PolyLine):
    def __init__(self, points, closed=False):
        super().__init__(points, closed=closed)
        self.other = None
        self.pin = None

        self.plot_kwargs['c'] = np.random.rand(3)

    def connect(self, other, pin=None):
        self.other = other
        self.pin = pin
        other.other = self
        other.pin = pin
        other.plot_kwargs = self.plot_kwargs

    def insert(self, lo=0, hi=1, xy=None, figure=None):
        super().insert(lo, hi, xy=xy, figure=figure)
        self.other.insert_relative(lo, hi, relative=0.5, figure=figure)
        self.handle_pin(lo+1, figure=figure)

    def update(self, n=0, xy=(0.0, 0.0), figure=None):
        super().update(n=n, xy=xy, figure=figure)
        self.handle_pin(n, figure=figure)

    def handle_pin(self, n, figure=None):
        if self.pin is not None:
            self.other.points[n, self.pin] = self.points[n, self.pin]
            self.redraw(figure)
            self.other.redraw(figure)



def interactive_advanced_ground_truth_main(args, tracked_results):
    """
    Ground truth mode entry function.

    :param args:
    :param tracked_results:
    :return: :raise SystemExit:
    """

    calibration_px_to_mu = next(iter(tracked_results.values())).first.image.calibration_px_to_mu

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

    ground_truth_data = args.advanced_ground_truth

    # noinspection PyUnresolvedReferences
    try:
        with open(ground_truth_data, 'r') as fp:
            all_envs = jsonpickle.loads(fp.read())
    except FileNotFoundError:
        print("File did not exist, starting anew")
        all_envs = {}
    except json.decoder.JSONDecodeError:
        print("Corrupted (empty?) file, starting anew")
        all_envs = {}

    def save_data():
        """
        Saves the ground truth data to the file specified. (pickled data)

        """
        with open(ground_truth_data, 'w+') as inner_fp:
            inner_fp.write(jsonpickle.dumps(all_envs))
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
                'last_point_x': None,
                'last_point_y': None,
                'paired_polylines': [],
                'polyline_results': {}
            }

        key = repr((next_pos, next_chan))

        if key not in all_envs:
            all_envs[key] = empty_env()

        env = all_envs[key]

        pos = list(tracked_results.keys())[next_pos]
        tracking = tracked_results[pos]

        chan_num = list(tracking.channel_accumulator.keys())[next_chan]

        channels = tracking.channel_accumulator[chan_num]

        print("Opening position %d, channel %d" % (pos, chan_num,))

        data = np.zeros((len(channels), 6))

        n_timepoint, n_width, n_height, n_top, n_bottom, n_width_cumsum = 0, 1, 2, 3, 4, 5

        some_fluorescence_channel_image = some_channel_image = None
        fluorescence_count = 0

        for n, cc in enumerate(channels):
            data[n, n_timepoint] = cc.image.timepoint
            data[n, n_width] = cc.channel_image.shape[1]
            data[n, n_height] = cc.channel_image.shape[0]
            data[n, n_top] = cc.top
            data[n, n_bottom] = cc.bottom
            some_channel_image = cc.channel_image
            if isinstance(cc, FluorescentChannel):
                fluorescence_count = len(cc.fluorescences_channel_image)
                some_fluorescence_channel_image = cc.fluorescences_channel_image[0]

        if fluorescence_count > 0 and some_fluorescence_channel_image is None:
            print("File generated from fluorescence data, but no fluorescence channel information in cache.")
            print("Rerun analysis with -cfi/--channel-fluorescence-images option")

        data[:, n_width_cumsum] = np.cumsum(data[:, n_width])

        max_top, min_top = data[:, n_top].max(), data[:, n_top].min()
        max_bottom, min_bottom = data[:, n_bottom].max(), data[:, n_bottom].min()

        low, high = int(np.floor(min_top)), int(np.ceil(max_bottom))

        large_image = np.zeros((high - low, int(data[-1, n_width_cumsum])), dtype=some_channel_image.dtype)
        large_fluorescences_image = None
        if fluorescence_count and some_fluorescence_channel_image is not None:
            large_fluorescences_image = np.zeros(
                (fluorescence_count, high - low, int(data[-1, n_width_cumsum])),
                dtype=some_fluorescence_channel_image.dtype)

        large_image_min_max = [float('+Inf'), float('-Inf')]

        large_fluorescence_image_min_max = [[float('+Inf'), float('-Inf')]] * fluorescence_count

        fluorescence_backgrounds = [dict() for _ in range(fluorescence_count)]

        for n, cc in enumerate(channels):
            lower_border = int(np.floor(data[n, n_top] - low))
            large_image[
                lower_border:int(lower_border + data[n, n_height]),
                int(data[n, n_width_cumsum] - data[n, n_width]):int(data[n, n_width_cumsum])
            ] = cc.channel_image

            large_image_min_max = [min(cc.channel_image.min(), large_image_min_max[0]),
                                   max(cc.channel_image.max(), large_image_min_max[1])]

            if isinstance(cc, FluorescentChannel):
                for fluorescence_c in range(fluorescence_count):
                    if cc.fluorescences_channel_image[fluorescence_c] is not None:
                        large_fluorescences_image[
                            fluorescence_c,
                            lower_border:int(lower_border + data[n, n_height]),
                            int(data[n, n_width_cumsum] - data[n, n_width]):int(data[n, n_width_cumsum])
                        ] = cc.fluorescences_channel_image[fluorescence_c]

                        large_fluorescence_image_min_max[fluorescence_c] = [
                            min(cc.fluorescences_channel_image[fluorescence_c].min(), large_fluorescence_image_min_max[fluorescence_c][0]),
                            max(cc.fluorescences_channel_image[fluorescence_c].max(), large_fluorescence_image_min_max[fluorescence_c][1])
                        ]

                        fluorescence_backgrounds[fluorescence_c][n] = cc.image.background_fluorescences[fluorescence_c]

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

        axes_image = plt.imshow(large_image)
        axes_image.set_clim(vmin=large_image_min_max[0], vmax=large_image_min_max[1])
        axes_image._molyso_image_shown = -1  # yes, that's bad

        plt.title("Ground Truth — Position %d, channel %d" % (pos, chan_num,))

        plt.xlabel("x [Pixel]")
        plt.ylabel("y [Pixel]")

        fig.tight_layout()

        lm = PolyLinesManager(plt.gcf())
        for p1, p2 in env['paired_polylines']:
            lm.add(p1, p2)

        def update_callback():
            def pixels_to_timepoints(pixels):
                return np.array([
                    np.searchsorted(data[:, n_width_cumsum], pixel, side='right')
                    for pixel in pixels
                ])

            def timepoints_to_time(timepoints):
                return np.array([
                    data[timepoint, n_timepoint]
                    for timepoint in timepoints
                ])

            env['polyline_results'] = {}
            env['polyline_results_timestepwise'] = {}

            p_results = env['polyline_results']
            p_results_timestepwise = env['polyline_results_timestepwise']

            for polyline_num, (upper, lower) in enumerate(env['paired_polylines']):
                assert len(lower.points) == len(upper.points)

                x = lower.points[:, 0]

                u_y = upper.points[:, 1]
                l_y = lower.points[:, 1]

                if x[0] > x[-1]:  # the line is reversed!
                    x, u_y, l_y = x[::-1], u_y[::-1], l_y[::-1]

                timepoints = pixels_to_timepoints(x)

                t_deltas = timepoints[1:] - timepoints[:-1]
                indices_to_keep = np.r_[[True], t_deltas != 0]

                x, u_y, l_y = x[indices_to_keep], u_y[indices_to_keep], l_y[indices_to_keep]

                timepoints = pixels_to_timepoints(x)
                times = timepoints_to_time(timepoints)

                height_deltas = u_y - l_y

                height_deltas *= calibration_px_to_mu  # important

                height_development = np.c_[s_to_h(times), height_deltas]
                height_development = height_development[1:, :] - height_development[:-1, :]

                changes = height_development.copy()
                changes = changes[:, 1] / changes[:, 0]

                try:
                    average_elongation = np.average(changes, weights=height_development[:, 0])
                except ZeroDivisionError:
                    average_elongation = float('NaN')

                p_results[polyline_num] = {
                    'growth_start': s_to_h(times[0]),
                    'growth_end':  s_to_h(times[-1]),
                    'division_age': s_to_h(times[-1] - times[0]),
                    'growth_rate': np.log(2) / s_to_h(times[-1] - times[0]),
                    'average_elongation': average_elongation
                }

                # tricky: for every timepoint in between

                def interp(x1, x2, y1, y2, new_x):
                    return (((y2 - y1) / (x2 - x1)) * (new_x - x1)) + y1

                p_results_timestepwise[polyline_num] = []

                for n_t, t in enumerate(range(timepoints[0], timepoints[-1]+1)):
                    left_t = np.searchsorted(timepoints, t, side='left')

                    if left_t == 0:
                        left_t, right_t = left_t, left_t + 1
                    else:
                        left_t, right_t = left_t - 1, left_t

                    x_centered = data[t, n_width_cumsum] - data[t, n_width] / 2.0

                    new_upper = interp(x[left_t], x[right_t], u_y[left_t], u_y[right_t], x_centered)
                    new_lower = interp(x[left_t], x[right_t], l_y[left_t], l_y[right_t], x_centered)

                    # plt.plot([x_centered, x_centered], [new_upper, new_lower])

                    inner_results = {
                        'timepoint_num': t,
                        'timepoint': s_to_h(timepoints_to_time([t])[0]),
                        'length': (new_upper - new_lower) * calibration_px_to_mu
                    }

                    for fluorescence_c in range(fluorescence_count):
                        fimg = large_fluorescences_image[fluorescence_c,
                               int(new_lower):int(new_upper),
                               int(data[t, n_width_cumsum] - data[t, n_width]):int(data[t, n_width_cumsum])
                        ]

                        jobs = dict(min=lambda a: a.min(),
                                    max=lambda a: a.max(),
                                    mean=lambda a: a.mean(),
                                    std=lambda a: a.std(),
                                    median=lambda a: np.median(a))

                        for fun_name, fun_lambda in jobs.items():

                            try:
                                value = fun_lambda(fimg)
                            except ValueError:
                                value = float('NaN')

                            inner_results['fluorescence_%s_raw_%d' % (fun_name, fluorescence_c)] = value

                        inner_results['fluorescence_background_%d' % (fluorescence_c)] = fluorescence_backgrounds[
                            fluorescence_c][t]

                    p_results_timestepwise[polyline_num].append(inner_results)




        lm.update_callback = update_callback
        lm.update_callback()

        lm.draw(ax)

        def refresh():
            """
            Refreshes the overlay.

            """
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
            F       show/cycle fluorescence/brightfield
            o/O     output tabular data to console/file
            u/U     output tabular single cell data to console/file
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
                    lower = np.array([
                        [last_point_x, last_point_y],
                        [x, y]
                    ])

                    upper = lower.copy() - [0.0, 30.0]

                    p1, p2 = PairedPolyLine(lower), PairedPolyLine(upper)
                    p1.connect(p2, pin=0)

                    env['paired_polylines'].append((p1, p2))

                    lm.add(p1, p2)

                    lm.draw(ax)

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
                print()

                p_results = [ab[1] for ab in sorted(env['polyline_results'].items(), key=lambda ab: ab[0])]

                inner_mu = [res['growth_rate'] for res in p_results]
                print("µ = ", inner_mu, np.mean(inner_mu))

                inner_elo = [res['average_elongation'] for res in p_results]

                print("elongation rate = ", inner_elo, np.mean(inner_elo))

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

            if event.key == 'h':
                show_help()
            elif event.key == 'p':
                show_stats()
            elif event.key == 'd':
                lm.delete(*lm.line_segments[-2:])
                lm.draw(ax)
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
            elif event.key == 'F':
                if axes_image._molyso_image_shown < fluorescence_count:
                    axes_image._molyso_image_shown += 1  # TODO: Test with more than one fluorescence channels
                if axes_image._molyso_image_shown == fluorescence_count:
                    axes_image._molyso_image_shown = -1

                if axes_image._molyso_image_shown == -1:
                    axes_image.set_data(large_image)
                    # axes_image.autoscale()
                    axes_image.set_clim(vmin=large_image_min_max[0],
                                        vmax=large_image_min_max[1])
                else:
                    fluorescence_c = axes_image._molyso_image_shown
                    axes_image.set_data(large_fluorescences_image[fluorescence_c])
                    # axes_image.autoscale()
                    axes_image.set_clim(vmin=large_fluorescence_image_min_max[fluorescence_c][0],
                                        vmax=large_fluorescence_image_min_max[fluorescence_c][1])

                refresh()
            elif event.key == 'o' or event.key == 'O':
                # output

                recipient = None

                if event.key == 'O':
                    print("Please enter file name for tabular output [will be overwritten if exists]:")
                    file_name = input()
                    recipient = open(file_name, 'w+')

                out = QuickTableDumper(recipient=recipient)

                for key, t_env in all_envs.items():
                    t_pos, t_chan = map(int, key[1:-1].replace(' ', '').split(','))
                    x_pos = list(tracked_results.keys())[t_pos]
                    x_chan = list(tracked_results[x_pos].channel_accumulator.keys())[t_chan]

                    p_results = [ab[1] for ab in sorted(t_env['polyline_results'].items(), key=lambda ab: ab[0])]

                    inner_mu = [res['growth_rate'] for res in p_results]
                    mean_inner_mu = np.nanmean(inner_mu)

                    inner_elo = [res['average_elongation'] for res in p_results]
                    mean_inner_elo = np.nanmean(inner_elo)

                    for resultlet in p_results:
                        out.add({
                            'position': x_pos,
                            'channel': x_chan,
                            'growth_rate': resultlet['growth_rate'],
                            'growth_rate_channel_mean': mean_inner_mu,
                            'elongation_rate': resultlet['average_elongation'],
                            'elongation_rate_channel_mean': mean_inner_elo,
                            'division_age': resultlet['division_age'],
                            'growth_start': resultlet['growth_start'],
                            'growth_end': resultlet['growth_end'],
                        })

                if event.key == 'O':
                    recipient.close()
                    print("File written.")

            elif event.key == 'u' or event.key == 'U':
                # output

                recipient = None

                if event.key == 'U':
                    print("Please enter file name for tabular output [will be overwritten if exists]:")
                    file_name = input()
                    recipient = open(file_name, 'w+')

                out = QuickTableDumper(recipient=recipient)

                for key, t_env in all_envs.items():
                    t_pos, t_chan = map(int, key[1:-1].replace(' ', '').split(','))
                    x_pos = list(tracked_results.keys())[t_pos]
                    x_chan = list(tracked_results[x_pos].channel_accumulator.keys())[t_chan]

                    for line_num, rows_timestepwise in t_env['polyline_results_timestepwise'].items():
                        for row in rows_timestepwise:
                            result = row.copy()

                            result.update({
                                'position': x_pos,
                                'channel': x_chan,
                                'line_num': line_num,
                            })

                            out.add(result)

                if event.key == 'U':
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

