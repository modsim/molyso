import numpy
import numpy.random
#DuoOptimizerQueue, GlobalDuoOptimizerQueue,
from .tracking_infrastructure import CellTracker, CellCrossingCheckingGlobalDuoOptimizerQueue

from .tracking_output import dump_tracking, visualize_tracking, analyze_tracking


def track_complete_channel_timeline(times):
    previous = None
    tracker_mapping = {}

    timeslist = list(sorted(times.keys()))
    tl = enumerate(timeslist)

    n = 0

    keylist = []

    while len(keylist) == 0:

        first_time = timeslist[n]
        previous = times[first_time]
        first = times[first_time].channels

        keylist = list(range(len(first)))

        tracker_mapping = {c: CellTracker() for c in keylist}
        # helper
        channel_accumulator = {c: {} for c in keylist}

        cell_counts = {c: 0.0 for c in keylist}

        try:
            next(tl)
        except StopIteration:
            print("SEVERE WARNING: Apparently NO frame contained channels!")
            return {"tracking": {}, "accumulator": {}}

        if len(keylist) > 0:
            break

        print("Warning, Frame %d apparently without channels, skipping to the next one!" % (n))

        n += 1

    for timecounter, t in tl:
        i = times[t]
        alignment = previous.channels.align_with_and_return_indices(i.channels)
        for pind, cind in alignment:

            #if pind != 10:
            #    continue

            pcells = previous.channels[pind].cells
            ccells = i.channels[cind].cells

            for n in range(len(pcells) - 1, -1, -1):
                if pcells[n].length < 15:
                    del pcells[n]
            for n in range(len(ccells) - 1, -1, -1):
                if ccells[n].length < 15:
                    del ccells[n]

            fa = dict(i.channels.align_with_and_return_indices(first))

            #            tracker = tracker_mapping[fa[cind]]
            #            tracker.tick()
            # this is not perfectly right, but it's enough work with chan accum already
            cell_counts[fa[cind]] += len(i.channels[cind].cells)

            if t not in channel_accumulator[fa[cind]]:
                channel_accumulator[fa[cind]][t] = i.channels[cind]
            else:
                # somehow two channels were assumed to be the same, skipping the second one
                pass

        #            analyse_cell_fates(tracker, pcells, ccells)

        previous = i

    for ind in channel_accumulator.keys():
        channel_accumulator[ind] = [channel_accumulator[ind][n] for n in sorted(channel_accumulator[ind].keys())]

    for k, count in list(cell_counts.items()):
        if (count / timecounter) < 0.5:
            del tracker_mapping[k]
            del channel_accumulator[k]
            del cell_counts[k]

    for c in tracker_mapping.keys():
        tracker = tracker_mapping[c]
        channellist = channel_accumulator[c]

        ii = iter(channellist)

        if len(channellist) == 0:
            continue

        previous = next(ii)

        for i in ii:
            tracker.tick()
            analyse_cell_fates(tracker, previous.cells, i.cells)
            previous = i

    return {"tracking": tracker_mapping, "accumulator": channel_accumulator}
    # visualization!


def analyse_cell_fates(tracker, pcells, ccells):
    def it_got_lost(pcell, ccell):
        pass

    def it_is_same(pcell, ccell):
        tracker.get_cell_by_observation(pcell).add_observation(ccell)

    def it_is_children(pcell, ccells):
        c1, c2 = ccells
        oc = tracker.get_cell_by_observation(pcell)
        oc.add_child(tracker.new_observed_cell(c1))
        oc.add_child(tracker.new_observed_cell(c2))

    def it_is_new(pcell, ccell):
        tracker.new_observed_origin(ccell)

    opt = CellCrossingCheckingGlobalDuoOptimizerQueue()

    try:
        chan_traj = numpy.mean([tracker.get_cell_by_observation(pc).trajectories[-1] for pc in pcells])
    except KeyError:
        chan_traj = 0.0

    for pn, pc in enumerate(pcells):
        if not tracker.is_tracked(pc):  # this probably only occurs on the first attempt
            tracker.new_observed_origin(pc)
        for n, cc in enumerate(ccells):
            cc1 = cc

            tc = tracker.get_cell_by_observation(pc)


            def calc_trajectory(c1, c2):
                return (c1.centroid1dloc - c2.centroid1dloc) / (c1.channel.image.timepoint - c2.channel.image.timepoint)


            def _get_some_linspace(n):
                return numpy.linspace(1, n, n)


            end_weighted_average_tail_len = 5
            end_weighted_average_tail = 1.0 / numpy.linspace(1, end_weighted_average_tail_len,
                                                             end_weighted_average_tail_len)
            end_weighted_average_tail_sums = numpy.cumsum(end_weighted_average_tail)

            def end_weighted_average(arr):
                return arr[-1]
                to_use = min(len(arr), end_weighted_average_tail_len)
                return numpy.sum(end_weighted_average_tail[:to_use] * arr[-1:-to_use - 1:-1]) \
                       / end_weighted_average_tail_sums[to_use - 1]

                #return numpy.sum(arr * _get_some_linspace(len(arr))) / (len(arr) * (len(arr) + 1) * 0.5)
                #tmp = numpy.linspace(1, len(arr), len(arr))
                #return numpy.sum(arr * tmp) / numpy.sum(tmp)

            def calc_last_traj(tc):
                trajr = tc.trajectories
                #return numpy.mean(trajr)
                #print("***")
                #print(trajr)
                #print(end_weighted_average(trajr), calc_trajectory(pc,cc), end_weighted_average(trajr)/calc_trajectory(pc,cc))
                return end_weighted_average(trajr)

            last_traj = calc_last_traj(tc)

            def calc_last_elo(tc):
                elor = tc.elongation_rates
                #return numpy.mean(elor)
                return end_weighted_average(elor)


            last_elo = calc_last_elo(tc)

            #if chan_traj != 0.0:
            #    last_traj = (last_traj + chan_traj) / 2


            wneighbor = 0.40

            try:
                decp = tracker.get_cell_by_observation(pc.dec_neighbor)
                btraj = calc_last_traj(decp)
                belo = calc_last_elo(decp)

                last_traj = ((1 - wneighbor) * last_traj + wneighbor * btraj)
                last_elo = ((1 - wneighbor) * last_elo + wneighbor * belo)
            except KeyError:
                pass

            try:
                incp = tracker.get_cell_by_observation(pc.inc_neighbor)
                btraj = calc_last_traj(incp)
                belo = calc_last_elo(incp)

                last_traj = ((1 - wneighbor) * last_traj + wneighbor * btraj)
                last_elo = ((1 - wneighbor) * last_elo + wneighbor * belo)
            except KeyError:
                pass

            tdelta = cc.channel.image.timepoint - pc.channel.image.timepoint

            putative_shift = last_traj * tdelta
            putative_elongation = last_elo * tdelta

            su = putative_shift + 0.5 * putative_elongation
            sd = putative_shift - 0.5 * putative_elongation

            #print(tdelta, last_traj, last_elo, putative_shift, putative_elongation, su,sd)

            #su = sd = 0


            def calc_cost_same(onecell, othercell):
                #    angle = math.atan((((othercell.top + othercell.bottom) / 2) - (
                #        onecell.top + onecell.bottom) / 2) / (
                #                          othercell.channel.image.timepoint - onecell.channel.image.timepoint))
                #    angle /= (math.pi / 180.0)
                #    if abs(angle) > 10.0:
                #        return 999999999.0
                #print(angle)
                #    print(angle)
                value = (abs(onecell.top + su - othercell.top)
                         + abs(onecell.bottom + sd - othercell.bottom)) / 2
                shrinkage = othercell.length - onecell.length
                if shrinkage < 0:
                    shrinkage *= 5
                value -= shrinkage


                #value *= (onecell.centroid1dloc / 100.0)**1
                value += 2 * (onecell.centroid1dloc / 10.0) ** 1
                return 1.0 * value

            def calc_cost_child(onecell, othercell):
                tonecell = tracker.get_cell_by_observation(onecell)
                if len(tonecell.seen_as) < 2:
                    return 1000000000.0
                    #if tonecell
                c1d = (onecell.top + onecell.bottom) / 2 + 0.5 * putative_elongation
                value = 0.0
                value += abs(onecell.top + su - othercell.top)
                value += abs(c1d - othercell.bottom)
                value += abs(onecell.bottom + sd - othercell.bottom)
                value += abs(c1d - othercell.top)

                value /= 4.0

                shrinkage = othercell.length - onecell.length
                if shrinkage > 0:
                    shrinkage *= 5
                value += shrinkage
                #value *= (onecell.centroid1dloc / 100.0)**1
                value += 2 * (onecell.centroid1dloc / 10.0) ** 1
                return 1.15 * value

            cost_same = calc_cost_same(pc, cc)

            if (pc.length - cc.length) < 25:
                opt.add_outcome(cost_same, pc, cc, it_is_same)

            large_value = 1000000.0
            opt.add_outcome(large_value, [], cc, it_is_new)

            another_large_value = 1.0 * large_value

            opt.add_outcome(another_large_value, pc, [], it_got_lost)

            if n < len(ccells) - 1:
                cc2 = ccells[n + 1]

                #space_between = abs(cc1.bottom - cc2.top)
                #if space_between < 10:
                ccost = (calc_cost_child(pc, cc1) + calc_cost_child(pc, cc2)) / 2

                opt.add_outcome(ccost, pc, [cc1, cc2], it_is_children)

    opt.perform_optimal()
