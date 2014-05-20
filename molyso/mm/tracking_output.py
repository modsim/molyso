from __future__ import division, unicode_literals, print_function

import random
import numpy

from ..generic.etc import QuickTableDumper


try:
    import matplotlib.colors

    colors = list(matplotlib.colors.cnames.keys())
except ImportError:
    colors = [""]

from .. import DebugPlot, tunable
from ..generic.util import remove_outliers


def visualize_tracking(tracked_results):
    tr = tracked_results[list(sorted(tracked_results.keys()))[0]]
    tracker_mapping = tr["tracking"]
    channel_accumulator = tr["accumulator"]

    for k, tracker in tracker_mapping.items():

        minimum_average_cells = 2.0
        if tracker.average_cells < minimum_average_cells:  #0.5:
            print(
                "Channel %(channel_num)d has less than %(minimum_average_cells).2f cells per channel detection on average (=%(average_cells).2f), skipping!"
                % {"minimum_average_cells": minimum_average_cells, "channel_num": k,
                   "average_cells": tracker.average_cells})
            continue

        try:
            for displaymode in ["both"]:
                with DebugPlot() as p:
                    chans = channel_accumulator[k]
                    tps = numpy.sort([cc.image.timepoint for cc in chans])


                    # channel images per inch
                    cpi = 5.0
                    p.rcParams['figure.figsize'] = (len(tps) / cpi, 4.0)
                    p.rcParams['figure.dpi'] = 150

                    p.rcParams['figure.subplot.top'] = 0.8
                    p.rcParams['figure.subplot.bottom'] = 0.2
                    p.rcParams['figure.subplot.left'] = 0.2
                    p.rcParams['figure.subplot.right'] = 0.8

                    p.figure()
                    p.title("Channel #%d (average cells = %f)" % (k, tracker.average_cells))

                    maxh = 0

                    for cc in chans:


                        tp = cc.image.timepoint
                        bs = numpy.searchsorted(tps, tp, side="right")

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

                        if displaymode in ["both", "images"]:
                            try:
                                if cc.channelimage is not None:
                                    cdata = cc.channelimage
                                    p.imshow(cdata, extent=(left, right, cc.top, cc.bottom), origin="lower")

                            except AttributeError:
                                pass

                        if cc.bottom > maxh:
                            maxh = cc.bottom
                        if displaymode in ["both", "plots"]:
                            for cell in cc.cells:
                                coords = [[left, cell.bottom], [right, cell.bottom],
                                          [right, cell.top], [left, cell.top], [left, cell.bottom]]
                                p.poly_drawing_helper(coords, lw=0, edgecolor='r', facecolor='gray',
                                                      fill=True, alpha=0.25)

                    p.xlim([0, tps[-1]])
                    p.ylim([0, maxh])
                    p.gca().set_aspect('auto')
                    p.gca().set_autoscale_on(True)

                    if displaymode in ["both", "plots"]:
                        for n, tp in enumerate(tps):
                            p.text(tp, 0, "%d/%.2f" % (n + 1, tp), rotation=90, verticalalignment='center',
                                   horizontalalignment='center', size=4)

                    divtimes = []
                    divtimepoints = []
                    divpositions = []
                    scattercollectorx = []
                    scattercollectory = []
                    scattercollectorint = []

                    startsx = []
                    startsy = []
                    stopsx = []
                    stopsy = []
                    junctionx = []
                    junctiony = []

                    def collector(cell):
                        xpts = [c.channel.image.timepoint for c in cell.seen_as]
                        ypts = [c.centroid1dloc for c in cell.seen_as]

                        try:
                            fints = [c.fluorescence for c in cell.seen_as]
                        except:
                            fints = [0.0 for c in cell.seen_as]

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
                                delta = t2 - t1
                                delta /= 60.0 * 60.0
                                divtimes.append(delta)
                                divtimepoints.append(t2)
                                divpositions.append(cell.seen_as[-1].centroid1dloc)

                                junctionx.append(cell.seen_as[-1].channel.image.timepoint)
                                junctiony.append(cell.seen_as[-1].centroid1dloc)
                            else:
                                stopsx.append(cell.seen_as[-1].channel.image.timepoint)
                                stopsy.append(cell.seen_as[-1].centroid1dloc)
                        else:
                            startsx.append(cell.seen_as[0].channel.image.timepoint)
                            startsy.append(cell.seen_as[0].centroid1dloc)

                        scattercollectorx.extend(xpts)
                        scattercollectory.extend(ypts)
                        scattercollectorint.extend(fints)

                        col = random.choice(colors)
                        if displaymode in ["both", "plots"]:
                            p.plot(xpts, ypts, marker="o", markersize=0.1, lw=0.5, c=col)
                            p.fill_between(xpts, y1pts, y2pts, lw=0, cmap="jet", alpha=0.5, facecolor=col)

                        map(collector, cell.children)

                    map(collector, tracker.origins)

                    scattercollectorint = numpy.array(scattercollectorint)
                    # ints below background (0) are useless
                    scattercollectorint[scattercollectorint < 0] = 0

                    if displaymode in ["both", "plots"]:
                        p.scatter(startsx, startsy, c="green", s=10, marker=">", lw=0)
                        p.scatter(stopsx, stopsy, c="red", s=10, marker="8", lw=0)
                        p.scatter(junctionx, junctiony, c="blue", s=10, marker="D", lw=0)

                        sc = p.scatter(scattercollectorx, scattercollectory, c=scattercollectorint, s=6,
                                       cmap="jet",
                                       lw=0)
                        p.colorbar(sc)

                    if displaymode is "images":
                        sc = p.scatter([0], [0], c=[1], s=6, cmap="jet", lw=0)
                        p.colorbar(sc)

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

        except:
            pass




def dump_tracking(tracked_results, basename="_OUT"):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    def poly_drawing_helper(p, coords, **kwargs):
        gca = p.gca()
        if gca:
            actions = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 1)
            gca.add_patch(PathPatch(Path(coords, actions), **kwargs))

    multipoints = list(sorted(tracked_results.keys()))

    for mp in multipoints:
        channel_accumulator = tracked_results[mp]["accumulator"]
        for k, tracker in tracked_results[mp]["tracking"].items():
            try:
                p = matplotlib.pyplot
                chans = channel_accumulator[k]
                tps = numpy.sort([cc.image.timepoint for cc in chans])


                # chanimages per inch
                cpi = 5.0
                p.rcParams['figure.figsize'] = (len(tps) / cpi, 4)
                p.rcParams['figure.dpi'] = 150

                #p.rcParams['figure.autolayout'] = True

                p.rcParams['figure.subplot.top'] = 0.8
                p.rcParams['figure.subplot.bottom'] = 0.2
                p.rcParams['figure.subplot.left'] = 0.2
                p.rcParams['figure.subplot.right'] = 0.8

                p.figure()
                p.title("Multipoint %d Channel %d (average cells = %f)" % (mp, k, tracker.average_cells))
                maxh = 0
                for cc in chans:

                    tp = cc.image.timepoint
                    bs = numpy.searchsorted(tps, tp, side="right")

                    try:
                        ne = tps[bs]
                    except IndexError:
                        #print("EXCP")
                        ne = tp
                    try:
                        pre = tps[bs - 2]
                    except IndexError:
                        #print("EXCP")
                        pre = tp
                    left = tp - abs(tp - pre) / 2
                    right = tp + abs(ne - tp) / 2

                    if left < 0:
                        left = 0
                    if right > tps[-1]:
                        right = tps[-1]
                        #print(left, tp, right)

                    if cc.bottom > maxh:
                        maxh = cc.bottom
                    for cell in cc.cells:
                        coords = [[left, cell.bottom], [right, cell.bottom],
                                  [right, cell.top], [left, cell.top], [left, cell.bottom]]
                        poly_drawing_helper(p, coords, lw=0, edgecolor='r', facecolor='gray', fill=True, alpha=0.25)

                p.xlim([0, tps[-1]])
                p.ylim([0, maxh])
                p.gca().set_aspect('auto')
                p.gca().set_autoscale_on(True)

                for n, tp in enumerate(tps):
                    p.text(tp, 0, "%d/%.2f" % (n + 1, tp), rotation=90, verticalalignment='center',
                           horizontalalignment='center', size=4)

                divtimes = []
                divtimepoints = []
                divpositions = []
                scattercollectorx = []
                scattercollectory = []
                scattercollectorint = []

                startsx = []
                startsy = []
                stopsx = []
                stopsy = []
                junctionx = []
                junctiony = []

                def collector(cell):
                    xpts = [c.channel.image.timepoint for c in cell.seen_as]
                    ypts = [c.centroid1dloc for c in cell.seen_as]

                    try:
                        fints = [c.fluorescence for c in cell.seen_as]
                    except:
                        fints = [0.0 for c in cell.seen_as]

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
                            delta = t2 - t1
                            delta /= 60.0 * 60.0
                            divtimes.append(delta)
                            divtimepoints.append(t2)
                            divpositions.append(cell.seen_as[-1].centroid1dloc)

                            junctionx.append(cell.seen_as[-1].channel.image.timepoint)
                            junctiony.append(cell.seen_as[-1].centroid1dloc)
                        else:
                            stopsx.append(cell.seen_as[-1].channel.image.timepoint)
                            stopsy.append(cell.seen_as[-1].centroid1dloc)
                    else:
                        startsx.append(cell.seen_as[0].channel.image.timepoint)
                        startsy.append(cell.seen_as[0].centroid1dloc)

                    scattercollectorx.extend(xpts)
                    scattercollectory.extend(ypts)
                    scattercollectorint.extend(fints)

                    col = colors[int(random.random() * len(colors))]
                    p.plot(xpts, ypts, marker="o", markersize=0.1, lw=0.5, c=col)
                    p.fill_between(xpts, y1pts, y2pts, lw=0, cmap="jet", alpha=0.5, facecolor=col)

                    map(collector, cell.children)

                map(collector, tracker.origins)

                scattercollectorint = numpy.array(scattercollectorint)
                # ints below background (0) are useless
                scattercollectorint[scattercollectorint < 0] = 0

                p.scatter(startsx, startsy, c="green", s=10, marker=">", lw=0)
                p.scatter(stopsx, stopsy, c="red", s=10, marker="8", lw=0)
                p.scatter(junctionx, junctiony, c="blue", s=10, marker="D", lw=0)
                try:
                    sc = p.scatter(scattercollectorx, scattercollectory, c=scattercollectorint, s=6, cmap="jet", lw=0)
                    p.colorbar(sc)
                except:
                    pass
                p.savefig("%s-mp-%d-c-%d.pdf" % (basename, mp, k))
                p.close('all')
            except:
                pass


def iterate_over_cells(cells):
    collector = []

    def _rec(another_cell):
        collector.append(another_cell)
        for yet_another_cell in another_cell.children:
            _rec(yet_another_cell)

    for cell in cells:
        _rec(cell)
    return collector


def analyze_tracking(tracked_results, recipient=None):
    s_to_h = lambda s: s / (60.0 * 60.0)

    shouldskip = False

    t = QuickTableDumper(recipient=recipient)
    multipoints = list(sorted(tracked_results.keys()))

    for mp in multipoints:
        for k, tracker in tracked_results[mp]["tracking"].items():
            minimum_average_cells = 2.0
            if shouldskip and tracker.average_cells < minimum_average_cells:  #0.5:
                print("Skipping an apparently empty channel...")
                continue

            for cell in iterate_over_cells(tracker.origins):
                sl = len(cell.seen_as)

                for sn, sa in enumerate(cell.seen_as):

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
                        "multipoint": mp,
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

