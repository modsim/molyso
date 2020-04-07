# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from ..generic.tunable import tunable

import numpy as np


def iterate_over_cells(cells):
    """

    :param cells:
    :return:
    """
    collector = []

    def _rec(another_cell):
        collector.append(another_cell)
        for yet_another_cell in another_cell.children:
            _rec(yet_another_cell)

    for a_cell in cells:
        _rec(a_cell)

    # for reproducible results, sort the cells

    collector = sorted(collector, key=lambda cell: (cell.seen_as[0].channel.image.timepoint, cell.seen_as[0].local_top))

    return collector


def tracker_to_cell_list(tracker):
    """


    :param tracker:
    :return:
    """
    return iterate_over_cells(tracker.origins)


def s_to_h(s):
    """
    converts seconds to hours
    :param s: seconds
    :return: hours
    """
    return s / (60.0 * 60.0)


# noinspection PyUnusedLocal
def s_to_h_str(s, *args, **kwargs):
    """
    converts seconds to hours as a rounded string
    :param s: seconds
    :return: hours
    :param s: seconds
    :return: hours string
    """
    return ("%.2f" % (s_to_h(s),)).replace('.00', '')


def catch_index_error(what, otherwise):
    """
    runs callable 'what' and catches IndexErrors, returning 'otherwise' if one occurred
    :param what: callable
    :param otherwise: alternate result in case of IndexError
    :return: result of 'what' or 'otherwise' in case of IndexError
    """
    try:
        return what()
    except IndexError:
        return otherwise


def catch_attribute_error(what, otherwise):
    """
    runs callable 'what' and catches AttributeError, returning 'otherwise' if one occurred
    :param what: callable
    :param otherwise: alternate result in case of IndexError
    :return: result of 'what' or 'otherwise' in case of IndexError
    """
    try:
        return what()
    except AttributeError:
        return otherwise


def plot_timeline(p, channels, cells,
                  figure_presetup=None, figure_finished=None,
                  show_images=True, show_overlay=True,
                  leave_open=False):
    """

    :param p:
    :param channels:
    :param cells:
    :param figure_presetup:
    :param figure_finished:
    :param show_images:
    :param show_overlay:
    :param leave_open:
    """
    from ..debugging.debugplot import poly_drawing_helper
    import matplotlib.colors

    colors = list(matplotlib.colors.cnames.keys())

    time_points = np.sort([cc.image.timepoint for cc in channels])

    channels_per_inch = 5.0

    p.rcParams.update({
        'figure.figsize': (len(time_points) / channels_per_inch, 4.0),
        'figure.dpi': 150,
        'figure.subplot.top': 0.8,
        'figure.subplot.bottom': 0.2,
        'figure.subplot.left': 0.2,
        'figure.subplot.right': 0.8
    })

    p.figure()

    if figure_presetup:
        figure_presetup(p)

    min_h = float('inf')
    max_h = 0

    for cc in channels:
        time_point = cc.image.timepoint
        bs = np.searchsorted(time_points, time_point, side='right')

        ne = time_points[bs] if 0 <= bs < len(time_points) else time_point

        pre = time_points[bs - 2] if 0 <= bs - 2 < len(time_points) else time_point

        left = max(0.0, time_point - abs(time_point - pre) / 2.0)
        right = min(time_points[-1], time_point + abs(ne - time_point) / 2.0)

        if show_images:
            channel_image_data = getattr(cc, 'channel_image', None)
            if channel_image_data is not None:
                p.imshow(channel_image_data, extent=(left, right, cc.top, cc.bottom),
                         origin='lower', cmap='gray', zorder=1.1)

        if cc.bottom > max_h:
            max_h = cc.bottom

        if cc.top < min_h:
            min_h = cc.top

        if show_overlay:
            for cell in cc.cells:
                coords = [[left, cell.bottom], [right, cell.bottom],
                          [right, cell.top], [left, cell.top], [left, cell.bottom]]
                poly_drawing_helper(p, coords,
                                    lw=0, edgecolor='r', facecolor='white', fill=True, alpha=0.25, zorder=1.2)

    p.gca().xaxis.set_major_formatter(p.FuncFormatter(s_to_h_str))
    p.gca().xaxis.set_major_locator(p.MultipleLocator(60.0 * 60.0 * 1))
    p.gca().xaxis.set_minor_locator(p.MultipleLocator(60.0 * 60.0 * 0.25))

    p.xlabel("Experiment Time [h]")
    p.ylabel("y [Pixel]")

    p.gca().set_aspect('auto')
    p.gca().set_autoscale_on(True)

    p.xlim(time_points[0], time_points[-1])
    p.ylim(min_h, max_h)

    p.tight_layout()

    if show_overlay:
        time_format_str = '#%0' + str(int(np.log10(len(time_points))) + 1) + 'd' + ' ' \
                          + '%0' + str(int(np.log10(time_points[-1])) + 1) + '.2fs'

        for n, time_point in enumerate(time_points):
            p.text(time_point, -max_h * 0.25, time_format_str % (n + 1, time_point),
                   rotation=90, verticalalignment='center', horizontalalignment='center', size=2.85)

    needed_length = sum(len(cell.seen_as) for cell in cells) + len(cells)

    scatter_collector = np.zeros((needed_length, 5), dtype=np.float32)  # type, x, y, fluor, length
    scatter_used = 0

    # constants

    pos_type, pos_time_point, pos_centroid, pos_length, pos_fluorescence_start = 0, 1, 2, 3, 4

    type_nothing, type_start, type_stop, type_junction = 0.0, 1.0, 2.0, 3.0

    np.random.seed(tunable('colors.visualization.track.random.seed', 3141592653,
                           description="Random seed for tracking visualization."))

    for cell in cells:
        old_scatter_used = scatter_used

        if cell.parent is not None:
            parent_cell = cell.parent.seen_as[-1]

            scatter_collector[scatter_used, pos_type] = type_nothing
            scatter_collector[scatter_used, pos_time_point] = parent_cell.channel.image.timepoint
            scatter_collector[scatter_used, pos_centroid] = parent_cell.centroid_1d
            # TODO different visualization?
            scatter_collector[scatter_used, pos_length] = parent_cell.length
            scatter_collector[scatter_used, pos_fluorescence_start] = \
                catch_index_error(lambda: getattr(parent_cell, 'fluorescences', [0.0])[0], 0.0)

            scatter_used += 1

        last_cell_number = len(cell.seen_as) - 1

        for nc, cell_appearance in enumerate(cell.seen_as):

            if nc == 0 and cell.parent is None:
                the_type = type_start
            elif nc == last_cell_number:
                the_type = type_junction if len(cell.children) > 0 else type_stop
            else:
                the_type = type_nothing

            scatter_collector[scatter_used, pos_type] = the_type
            scatter_collector[scatter_used, pos_time_point] = cell_appearance.channel.image.timepoint
            scatter_collector[scatter_used, pos_centroid] = cell_appearance.centroid_1d
            scatter_collector[scatter_used, pos_length] = cell_appearance.length
            scatter_collector[scatter_used, pos_fluorescence_start] = \
                catch_index_error(lambda: getattr(cell_appearance, 'fluorescences', [0.0])[0], 0.0)

            scatter_used += 1

        color = tunable('colors.visualization.track.color', '#005B82', description="Track color for visualization.")
        color_alpha = tunable('colors.visualization.track.alpha', 0.3, description="Track alpha for visualization.")

        if tunable('colors.visualization.track.random', 1, description="Randomize tracking color palette?") == 1:
            color = colors[np.random.randint(0, len(colors))]

        if show_overlay:
            slice_of_interest = scatter_collector[old_scatter_used:scatter_used, :]
            p.plot(
                slice_of_interest[:, pos_time_point],
                slice_of_interest[:, pos_centroid],
                marker=None, lw=0.5, c=color, zorder=1.4)  # marker='o', markersize=0.1
            p.fill_between(
                slice_of_interest[:, pos_time_point],
                slice_of_interest[:, pos_centroid] - 0.5 * slice_of_interest[:, pos_length],
                slice_of_interest[:, pos_centroid] + 0.5 * slice_of_interest[:, pos_length],
                lw=0, alpha=color_alpha, facecolor=color, zorder=1.3)

    scatter_collector = scatter_collector[:scatter_used, :]

    has_fluorescence = not (scatter_collector[:, pos_fluorescence_start] == 0.0).all()

    if show_overlay:
        starts = scatter_collector[scatter_collector[:, pos_type] == type_start, :]
        stops = scatter_collector[scatter_collector[:, pos_type] == type_stop, :]
        junctions = scatter_collector[scatter_collector[:, pos_type] == type_junction, :]

        p.scatter(starts[:, pos_time_point], starts[:, pos_centroid],
                  c='green', s=10, marker='>', lw=0, zorder=1.5)
        p.scatter(stops[:, pos_time_point], stops[:, pos_centroid],
                  c='red', s=10, marker='8', lw=0, zorder=1.5)
        p.scatter(junctions[:, pos_time_point], junctions[:, pos_centroid],
                  c='blue', s=10, marker='D', lw=0, zorder=1.5)

        if has_fluorescence:
            sc = p.scatter(scatter_collector[:, pos_time_point],
                           scatter_collector[:, pos_centroid],
                           c=scatter_collector[:, pos_fluorescence_start],
                           s=5, cmap='jet', lw=0, zorder=10.0)
            p.colorbar(sc)
        else:
            p.scatter(scatter_collector[:, pos_time_point],
                      scatter_collector[:, pos_centroid],
                      s=5, lw=0, zorder=10.0)

    if figure_finished:
        figure_finished(p)

    if not leave_open:
        p.close('all')


_unique_id_cache = {}
_unique_id_value = 1


def get_object_unique_id(obj):
    """

    :param obj:
    :return:
    """
    global _unique_id_cache, _unique_id_value

    if obj is None:
        return 0

    if id(obj) not in _unique_id_cache:
        _unique_id_cache[id(obj)] = _unique_id_value
        _unique_id_value += 1

    return _unique_id_cache[id(obj)]


def analyze_tracking(cells, receptor, meta=None):
    """

    :param meta:
    :param cells:
    :param receptor:
    """
    for cell in cells:
        for sn, sa in enumerate(cell.seen_as):
            tmp = {
                'cell_age': s_to_h(sa.channel.image.timepoint - cell.seen_as[0].channel.image.timepoint),
                'elongation_rate': catch_index_error(lambda: cell.raw_elongation_rates[sn], float('NaN')),
                'length': sa.channel.image.pixel_to_mu(sa.length),
                'uid_track': get_object_unique_id(cell.ultimate_parent),
                'uid_thiscell': get_object_unique_id(sa),
                'uid_cell': get_object_unique_id(cell),
                # None has an id !
                'uid_parent': get_object_unique_id(cell.parent),
                'timepoint': sa.channel.image.timepoint,
                'timepoint_num': sa.channel.image.timepoint_num,
                'cellyposition': sa.centroid_1d,
                'cellxposition': (sa.channel.left + sa.channel.right)/2,
                'multipoint': sa.channel.image.multipoint,
                'meta': str(meta) if meta else '',
                'channel_in_multipoint': sa.channel.image.channels.channels_list.index(sa.channel),
                'channel_average_cells': cell.tracker.average_cells,
                'channel_orientation': sa.channel.image.guess_channel_orientation(),
                'channel_width': sa.channel.image.pixel_to_mu(abs(sa.channel.left - sa.channel.right)),
                'about_to_divide': int(
                    ((sn + 1) == len(cell.seen_as)) and (cell.parent is not None) and (len(cell.children) > 0)
                ),
                'division_age': catch_index_error(
                    lambda: s_to_h(
                        cell.children[0].seen_as[0].channel.image.timepoint - cell.seen_as[0].channel.image.timepoint
                    ), float('NaN')),
                'fluorescence_count': len(getattr(sa, 'fluorescences', []))
            }

            for f in range(len(getattr(sa, 'fluorescences', []))):
                tmp['fluorescence_' + str(f)] = sa.fluorescences[f]
                tmp['fluorescence_raw_' + str(f)] = sa.fluorescences_raw[f]
                tmp['fluorescence_std_' + str(f)] = sa.fluorescences_std[f]
                tmp['fluorescence_raw_min_' + str(f)] = catch_attribute_error(lambda: sa.fluorescences_min[f], float('NaN'))
                tmp['fluorescence_raw_max_' + str(f)] = catch_attribute_error(lambda: sa.fluorescences_max[f], float('NaN'))
                tmp['fluorescence_raw_median_' + str(f)] = catch_attribute_error(lambda: sa.fluorescences_median[f], float('NaN'))
                tmp['fluorescence_background_' + str(f)] = sa.channel.image.background_fluorescences[f]

            receptor(tmp)
