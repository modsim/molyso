# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy


def iterate_over_cells(cells):
    collector = []

    def _rec(another_cell):
        collector.append(another_cell)
        for yet_another_cell in another_cell.children:
            _rec(yet_another_cell)

    for cell in cells:
        _rec(cell)
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


def plot_timeline(p, channels, cells,
                  figure_presetup=None, figure_finished=None,
                  show_images=True, show_overlay=True,
                  leave_open=False):
    from ..debugging.debugplot import poly_drawing_helper

    time_points = numpy.sort([cc.image.timepoint for cc in channels])

    channels_per_inch = 5.0
    p.rcParams['figure.figsize'] = (len(time_points) / channels_per_inch, 4.0)
    p.rcParams['figure.dpi'] = 150

    p.rcParams['figure.subplot.top'] = 0.8
    p.rcParams['figure.subplot.bottom'] = 0.2
    p.rcParams['figure.subplot.left'] = 0.2
    p.rcParams['figure.subplot.right'] = 0.8

    p.figure()

    if figure_presetup:
        figure_presetup(p)

    max_h = 0

    for cc in channels:
        time_point = cc.image.timepoint
        bs = numpy.searchsorted(time_points, time_point, side='right')

        ne = time_points[bs] if 0 <= bs < len(time_points) else time_point

        pre = time_points[bs - 2] if 0 <= bs < len(time_points) else time_point

        left = max(0.0, time_point - abs(time_point - pre) / 2.0)
        right = min(time_points[-1], time_point + abs(ne - time_point) / 2.0)

        if show_images:
            channel_image_data = getattr(cc, 'channel_image', None)
            if channel_image_data is not None:
                p.imshow(channel_image_data, extent=(left, right, cc.top, cc.bottom),
                         origin='lower', cmap='gray', zorder=1.1)

        if cc.bottom > max_h:
            max_h = cc.bottom

        if show_overlay:
            for cell in cc.cells:
                coords = [[left, cell.bottom], [right, cell.bottom],
                          [right, cell.top], [left, cell.top], [left, cell.bottom]]
                poly_drawing_helper(p, coords,
                                    lw=0, edgecolor='r', facecolor='white', fill=True, alpha=0.25, zorder=1.2)

    p.xlim(time_points[0], time_points[-1])
    p.ylim(0, max_h)
    p.gca().set_aspect('auto')
    p.gca().set_autoscale_on(True)

    if show_overlay:
        time_format_str = '#%0' + str(int(numpy.log10(len(time_points))) + 1) + 'd' + ' ' \
                          + '%0' + str(int(numpy.log10(time_points[-1])) + 1) + '.2fs'

        for n, time_point in enumerate(time_points):
            p.text(time_point, -max_h * 0.25, time_format_str % (n + 1, time_point),
                   rotation=90, verticalalignment='center', horizontalalignment='center', size=2.85)

    needed_length = sum(len(cell.seen_as) for cell in cells) + len(cells)

    # divisions = numpy.zeros((needed_length, 3))
    # divisions_used = 0

    # for cell in cells:
    # if cell.parent is not None:
    # if len(cell.children) > 0:
    # child = cell.children[0]
    #     # cell took first_occurrence_parent to first_occurrence_child time to divide!
    #     t1 = cell.seen_as[0].channel.image.timepoint
    #     t2 = child.seen_as[0].channel.image.timepoint
    #
    #     divisions[divisions_used, 0] = s_to_h(t2 - t1)
    #     divisions[divisions_used, 1] = t2
    #     divisions[divisions_used, 2] = cell.seen_as[-1].centroid1dloc
    #
    #     divisions_used += 1
    # divisions = divisions[:divisions_used, :]

    scatter_collector = numpy.zeros((needed_length, 5))  # type, x, y, int, length
    scatter_used = 0

    # "constants"

    type_nothing, type_start, type_stop, type_junction = 0.0, 1.0, 2.0, 3.0

    for cell in cells:
        old_scatter_used = scatter_used

        if cell.parent is not None:
            parent_cell = cell.parent.seen_as[-1]

            scatter_collector[scatter_used, 0] = type_nothing
            scatter_collector[scatter_used, 1] = parent_cell.channel.image.timepoint
            scatter_collector[scatter_used, 2] = parent_cell.centroid_1d
            scatter_collector[scatter_used, 3] = getattr(parent_cell, 'fluorescence', 0.0)
            scatter_collector[scatter_used, 4] = parent_cell.length

            scatter_used += 1

        last_cell_number = len(cell.seen_as) - 1

        for nc, cell_appearance in enumerate(cell.seen_as):

            if nc == 0 and cell.parent is None:
                the_type = type_start
            elif nc == last_cell_number:
                the_type = type_junction if len(cell.children) > 0 else type_stop
            else:
                the_type = type_nothing

            scatter_collector[scatter_used, 0] = the_type
            scatter_collector[scatter_used, 1] = cell_appearance.channel.image.timepoint
            scatter_collector[scatter_used, 2] = cell_appearance.centroid_1d
            scatter_collector[scatter_used, 3] = getattr(cell_appearance, 'fluorescence', 0.0)
            scatter_collector[scatter_used, 4] = cell_appearance.length

            scatter_used += 1

        col = '#005B82'
        if show_overlay:
            slice_of_interest = scatter_collector[old_scatter_used:scatter_used, :]
            p.plot(
                slice_of_interest[:, 1],
                slice_of_interest[:, 2],
                marker=None, lw=0.5, c=col, zorder=1.4)  # marker='o', markersize=0.1
            p.fill_between(
                slice_of_interest[:, 1],
                slice_of_interest[:, 2] - 0.5 * slice_of_interest[:, 4],
                slice_of_interest[:, 2] + 0.5 * slice_of_interest[:, 4],
                lw=0, alpha=0.3, facecolor=col, zorder=1.3)

    scatter_collector = scatter_collector[:scatter_used, :]

    sc = None

    if show_overlay:
        starts = scatter_collector[scatter_collector[:, 0] == type_start, :]
        stops = scatter_collector[scatter_collector[:, 0] == type_stop, :]
        junctions = scatter_collector[scatter_collector[:, 0] == type_junction, :]

        p.scatter(starts[:, 1], starts[:, 2],
                  c='green', s=10, marker='>', lw=0, zorder=1.5)
        p.scatter(stops[:, 1], stops[:, 2],
                  c='red', s=10, marker='8', lw=0, zorder=1.5)
        p.scatter(junctions[:, 1], junctions[:, 2],
                  c='blue', s=10, marker='D', lw=0, zorder=1.5)

        sc = p.scatter(scatter_collector[:, 1],
                       scatter_collector[:, 2],
                       c=scatter_collector[:, 3],
                       s=5, cmap='jet', lw=0, zorder=10.0)
        p.colorbar(sc)

    if show_images and sc is None:
        sc = p.scatter([time_points[0], time_points[0]], [max_h * 0.5, max_h * 0.5], c=[0.0, 1.0],
                       s=5, cmap='jet', lw=0, zorder=-1.0)
        p.colorbar(sc)

    if figure_finished:
        figure_finished(p)

    if not leave_open:
        p.close('all')


def analyze_tracking(cells, receptor):
    for cell in cells:
        for sn, sa in enumerate(cell.seen_as):
            receptor({
                'cell_age': s_to_h(sa.channel.image.timepoint - cell.seen_as[0].channel.image.timepoint),
                'elongation_rate': catch_index_error(lambda: cell.raw_elongation_rates[sn], float('NaN')),
                'length': sa.channel.image.pixel_to_mu(sa.length),
                'uid_track': id(cell),
                'uid_thiscell': id(sa),
                'uid_cell': id(cell),
                'uid_parent': id(cell.parent),
                'timepoint': sa.channel.image.timepoint,
                'timepoint_num': sa.channel.image.timepoint_num,
                'cellyposition': sa.centroid_1d,
                'multipoint': sa.channel.image.multipoint,
                'channel_in_multipoint': sa.channel.image.channels.channels_list.index(sa.channel),
                'channel_average_cells': cell.tracker.average_cells,
                'channel_orientation': sa.channel.image.guess_channel_orientation(),
                'about_to_divide': int(
                    ((sn + 1) == len(cell.seen_as)) and (cell.parent is not None) and (len(cell.children) > 0)
                ),
                'division_age': catch_index_error(
                    lambda: s_to_h(
                        cell.children[0].seen_as[0].channel.image.timepoint - cell.seen_as[0].channel.image.timepoint
                    ), float('NaN')),
                'fluorescence': getattr(sa, 'fluorescence', float('NaN')),
                'fluorescence_background': getattr(sa.channel.image, 'background_fluorescence', float('NaN')),
            })
