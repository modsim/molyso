from __future__ import division, unicode_literals, print_function

import random
import numpy

try:
    import matplotlib.colors

    colors = list(matplotlib.colors.cnames.keys())
except ImportError:
    colors = [""]

from .. import DebugPlot, tunable
from ..generic.util import remove_outliers


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
    return iterate_over_cells(tracker.origins)


def s_to_h(s):
    """

    :param s:
    :return:
    """
    return s / (60.0 * 60.0)


def plot_timeline(p, channels, cells,
                  figure_presetup=None, figure_finished=None,
                  show_images=True, show_overlay=True,
                  leave_open=False):
    def poly_drawing_helper(p, coords, **kwargs):
        gca = p.gca()
        if gca:
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch

            actions = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 1)
            gca.add_patch(PathPatch(Path(coords, actions), **kwargs))

    tps = numpy.sort([cc.image.timepoint for cc in channels])

    # channel images per inch
    cpi = 5.0
    p.rcParams['figure.figsize'] = (len(tps) / cpi, 4.0)
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


        tp = cc.image.timepoint
        bs = numpy.searchsorted(tps, tp, side='right')

        try:
            ne = tps[bs]
        except IndexError:
            ne = tp

        try:
            pre = tps[bs - 2]
        except IndexError:
            pre = tp

        left = tp - abs(tp - pre) / 2
        right = tp + abs(ne - tp) / 2

        if left < 0:
            left = 0
        if right > tps[-1]:
            right = tps[-1]

        if show_images:
            try:
                if cc.channelimage is not None:
                    cdata = cc.channelimage
                    p.imshow(cdata, extent=(left, right, cc.top, cc.bottom), origin='lower', cmap='gray')

            except AttributeError:
                pass

        if cc.bottom > max_h:
            max_h = cc.bottom

        if show_overlay:
            for cell in cc.cells:
                coords = [[left, cell.bottom], [right, cell.bottom],
                          [right, cell.top], [left, cell.top], [left, cell.bottom]]
                poly_drawing_helper(p, coords,
                                    lw=0, edgecolor='r', facecolor='gray', fill=True, alpha=0.25)
    print(tps)
    p.xlim(tps[0], tps[-1])
    p.ylim(0, max_h)
    p.gca().set_aspect('auto')
    p.gca().set_autoscale_on(True)

    if show_overlay:
        for n, tp in enumerate(tps):
            p.text(tp, 0, "%d/%.2f" % (n + 1, tp),
                   rotation=90, verticalalignment='center', horizontalalignment='center', size=4)

    division_times = []
    division_timepoints = []
    division_positions = []
    scatter_collector_x = [0.0, 0.0]
    scatter_collector_y = [0.0, 0.0]
    scatter_collector_intensity = [1.0, 0.0]

    starts_x = []
    starts_y = []
    stops_x = []
    stops_y = []
    junctions_x = []
    junctions_y = []

    for cell in cells:
        xpts = [c.channel.image.timepoint for c in cell.seen_as]
        ypts = [c.centroid1dloc for c in cell.seen_as]

        try:
            fints = [c.fluorescence for c in cell.seen_as]
        except:
            fints = [0.0] * len(cell.seen_as)

        y1pts = [c.top for c in cell.seen_as]
        y2pts = [c.bottom for c in cell.seen_as]
        if cell.parent is not None:
            xpts = [cell.parent.seen_as[-1].channel.image.timepoint] + xpts
            ypts = [cell.parent.seen_as[-1].centroid1dloc] + ypts
            try:
                fints = [cell.parent.seen_as[-1].fluorescence] + fints
            except:
                fints = [0.0] + fints
            y1pts = [cell.parent.seen_as[-1].top] + y1pts
            y2pts = [cell.parent.seen_as[-1].bottom] + y2pts
            # this cell has a parent!
            if len(cell.children) > 0:
                child = cell.children[0]
                # cell took first_occurrence_parent to first_occurrence_child time to divide!
                t1 = cell.seen_as[0].channel.image.timepoint
                t2 = child.seen_as[0].channel.image.timepoint

                division_times.append(s_to_h(t2 - t1))
                division_timepoints.append(t2)
                division_positions.append(cell.seen_as[-1].centroid1dloc)

                junctions_x.append(cell.seen_as[-1].channel.image.timepoint)
                junctions_y.append(cell.seen_as[-1].centroid1dloc)
            else:
                stops_x.append(cell.seen_as[-1].channel.image.timepoint)
                stops_y.append(cell.seen_as[-1].centroid1dloc)
        else:
            starts_x.append(cell.seen_as[0].channel.image.timepoint)
            starts_y.append(cell.seen_as[0].centroid1dloc)

        scatter_collector_x.extend(xpts)
        scatter_collector_y.extend(ypts)
        scatter_collector_intensity.extend(fints)

        col = random.choice(colors)
        if show_overlay:
            p.plot(xpts, ypts, marker='o', markersize=0.1, lw=0.5, c=col)
            p.fill_between(xpts, y1pts, y2pts, lw=0, cmap='jet', alpha=0.5, facecolor=col)

    sc = None

    if show_overlay:
        scatter_collector_intensity = numpy.array(scatter_collector_intensity)

        p.scatter(starts_x, starts_y, c='green', s=10, marker='>', lw=0)
        p.scatter(stops_x, stops_y, c='red', s=10, marker='8', lw=0)
        p.scatter(junctions_x, junctions_y, c='blue', s=10, marker='D', lw=0)

        sc = p.scatter(scatter_collector_x,
                       scatter_collector_y,
                       c=scatter_collector_intensity,
                       s=6, cmap='jet', lw=0)
        p.colorbar(sc)

    if show_images and sc is None:
        sc = p.scatter([0, 0], [0, 0], c=[0, 1], s=6, cmap='jet', lw=0)
        p.colorbar(sc)

    if figure_finished:
        figure_finished(p)

    if not leave_open:
        p.close('all')


def analyze_tracking(tracker, t=None):
    k = None

    for cell in iterate_over_cells(tracker.origins):
        sl = len(cell.seen_as)

        for sn, sa in enumerate(cell.seen_as):

            if k is None:
                k = sa.channel.image.channels.index(sa.channel)

            about_to_divide = ((sn + 1) == sl) and (cell.parent is not None) and (len(cell.children) > 0)

            r = {
                "cell_age": s_to_h(sa.channel.image.timepoint - cell.seen_as[0].channel.image.timepoint),
                "elongation_rate": float("NaN"),
                "length": sa.length,
                "uid_track": id(cell),
                "uid_thiscell": id(sa),
                "uid_cell": id(cell),
                "uid_parent": id(cell.parent),
                "timepoint": sa.channel.image.timepoint,
                "timepoint_num": sa.channel.image.timepoint_num,
                "cellyposition": sa.centroid1dloc,
                "multipoint": sa.channel.image.multipoint,
                "channel_in_multipoint": k,
                "channel_average_cells": tracker.average_cells,
                "about_to_divide": (1 if about_to_divide else 0),
                "division_age": float("NaN"),
                "fluorescence": float("NaN"),
                "fluorescence_background": float("NaN"),
            }

            try:
                r["elongation_rate"] = cell.raw_elongation_rates[sn]
            except IndexError:
                pass

            try:
                r["division_age"] = s_to_h(cell.children[0].seen_as[0].channel.image.timepoint - cell.seen_as[
                    0].channel.image.timepoint)
            except IndexError:
                pass

            try:
                r["fluorescence"] = sa.fluorescence
            except AttributeError:
                pass

            try:
                r["fluorescence_background"] = sa.channel.image.background_fluorescence
            except AttributeError:
                pass

            t.add(r)


"""

    divtimes = numpy.array(divtimes)
    #xdivtimes = divtimes[divtimes > 1.0]
    xdivtimes = remove_outliers(divtimes, times_std=1.0)
    #xdivtimes = divtimes
    import math

    ln2 = math.log(2)
    print(len(divtimes), len(xdivtimes))
    s = "counts=%d mean division time=%f 1/mean(d)=%f mean(1/d)=%f" % \
        (len(xdivtimes), xdivtimes.mean(), 1 / xdivtimes.mean(), (1 / xdivtimes).mean())
    print(s)
    print("***")
    s = "counts=%d mean division time=%f ln(2)/mean(d)=%f mean(ln(2)/d)=%f" % \
        (len(xdivtimes), xdivtimes.mean(), ln2 / xdivtimes.mean(), (ln2 / xdivtimes).mean())
    print(s)
    print("/***")

    s = "counts=%d median division time=%f 1/median(d)=%f median(1/d)=%f" % \
        (len(xdivtimes), numpy.median(xdivtimes), 1 / numpy.median(xdivtimes),
         numpy.median(1 / xdivtimes))
    print(s)
    print(repr(xdivtimes))

    print("log(2)/...")
    print(repr(ln2 / xdivtimes))

    xdivtimes = numpy.array(remove_outliers(xdivtimes, 1.0))
    print("OUTLIERS REMOVED")
    print(len(divtimes), len(xdivtimes))
    s = "counts=%d mean division time=%f 1/mean(d)=%f mean(1/d)=%f" % \
        (len(xdivtimes), xdivtimes.mean(), 1 / xdivtimes.mean(), (1 / xdivtimes).mean())
    print(s)

    p.title("c%d (avg c=%f)\n%s" % (k, tracker.average_cells, s))
"""