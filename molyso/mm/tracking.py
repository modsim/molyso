import numpy
import numpy.random
#DuoOptimizerQueue, GlobalDuoOptimizerQueue,
from .tracking_infrastructure import CellTracker, CellCrossingCheckingGlobalDuoOptimizerQueue

from .tracking_output import dump_tracking, visualize_tracking, analyze_tracking


def track_complete_channel_timeline(times):
    previous = None
    tracker_mapping = {}

    timeslist = list(sorted(times.keys()))

    first_num = None

    keylist = []

    for n, t in enumerate(timeslist):
        first_time = t
        previous = times[first_time]

        first = times[first_time].channels

        if len(first) > 0:
            break

        print("Skipping channel")

    keylist = list(range(len(first)))

    tracker_mapping = {c: CellTracker() for c in keylist}
    # helper
    channel_accumulator = {c: {} for c in keylist}

    cell_counts = {c: 0.0 for c in keylist}

    image = None

    for t in timeslist[n + 1:]:
        if image:
            previous = image

        image = times[t]
        alignment = previous.channels.align_with_and_return_indices(image.channels)
        alignment_with_first = dict(image.channels.align_with_and_return_indices(first))

        for _, current_index in alignment:
            # ths is not perfectly right, but it's enough work with chan accum already
            cell_counts[alignment_with_first[current_index]] += len(image.channels[current_index].cells)

            if t not in channel_accumulator[alignment_with_first[current_index]]:
                channel_accumulator[alignment_with_first[current_index]][t] = image.channels[current_index]

    for index in channel_accumulator.keys():
        channel_accumulator[index] = [channel_accumulator[index][n] for n in sorted(channel_accumulator[index].keys())]

    for k, count in list(cell_counts.items()):
        if (count / len(timeslist)) < 0.5:
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

        for image in ii:
            tracker.tick()
            analyse_cell_fates(tracker, previous.cells, image.cells)
            previous = image

    return {"tracking": tracker_mapping, "accumulator": channel_accumulator}
    # visualization!


def analyse_cell_fates(tracker, pcells, ccells):

    opt = CellCrossingCheckingGlobalDuoOptimizerQueue()

    try:
        chan_traj = numpy.mean([tracker.get_cell_by_observation(pc).trajectories[-1] for pc in pcells])
    except KeyError:
        chan_traj = 0.0

    for pn, pc in enumerate(pcells):
        if not tracker.is_tracked(pc):  # this probably only occurs on the first attempt
            tracker.new_observed_origin(pc)
        for n, cc in enumerate(ccells):


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

            time_delta = cc.channel.image.timepoint - pc.channel.image.timepoint

            putative_shift = last_traj * time_delta
            putative_elongation = last_elo * time_delta

            putative_shift = 0.0
            putative_elongation = 0.0

            su = putative_shift + 0.5 * putative_elongation
            sd = putative_shift - 0.5 * putative_elongation

            def calc_cost_same(onecell, othercell):
                value = (abs(onecell.top + su - othercell.top)
                         + abs(onecell.bottom + sd - othercell.bottom)) / 2
                shrinkage = othercell.length - onecell.length
                if shrinkage < 0:
                    shrinkage *= 5
                value -= shrinkage

                value += 2 * (onecell.centroid1dloc / 10.0) ** 1
                return 1.0 * value

            def calc_cost_child(onecell, othercell):
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

            opt.add_outcome(cost_same, {pc}, {cc}, lambda pcell, ccell: \
                tracker.get_cell_by_observation(pcell[0]).add_observation(ccell[0]))

            large_value = 1000000.0

            cost_new_cell = 1.0 * large_value

            opt.add_outcome(cost_new_cell, set(), {cc}, lambda pcell, ccell: tracker.new_observed_origin(ccell[0]))

            cost_lost_cell = 1.0 * large_value
            opt.add_outcome(cost_lost_cell, {pc}, set(), lambda pcell, ccell: None)

            if n < len(ccells) - 1:
                cc1 = ccells[n + 0]
                cc2 = ccells[n + 1]

                cost_children = (calc_cost_child(pc, cc1) + calc_cost_child(pc, cc2)) / 2.0

                opt.add_outcome(cost_children, {pc}, {cc1, cc2},
                                lambda pcell, ccell: tracker.get_cell_by_observation(pcell[0]).
                                add_children(tracker.new_observed_cell(ccell[0]), tracker.new_observed_cell(ccell[1])))

    opt.perform_optimal()

