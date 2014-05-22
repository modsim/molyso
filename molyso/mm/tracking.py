# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

from .tracking_infrastructure import CellTracker, CellCrossingCheckingGlobalDuoOptimizerQueue

from .tracking_output import *


def dummy_progress_indicator():
    pass


def ignorant_next(iterable):
    try:
        return next(iterable)
    except StopIteration:
        return None


class TrackedPosition(object):
    def __init__(self):
        self.times = None
        self.n = 0

        self.timeslist = []

        self.first = []

        self.tracker_mapping = {}
        self.channel_accumulator = {}
        self.cell_counts = {}

    def set_times(self, times):
        self.times = times
        self.find_first_valid_time()

    def find_first_valid_time(self):
        self.timeslist = list(sorted(self.times.keys()))

        for self.n, t in enumerate(self.timeslist):
            first_time = t

            self.first = self.times[first_time].channels

            if len(self.first) > 0:
                break

            print("Skipping channel")

        key_list = list(range(len(self.first)))

        self.tracker_mapping = {c: CellTracker() for c in key_list}
        self.channel_accumulator = {c: {} for c in key_list}
        self.cell_counts = {c: [] for c in key_list}

    def align_channels(self, progress_indicator=dummy_progress_indicator):
        image = None

        for _ in range(self.n):
            ignorant_next(progress_indicator)

        for t in self.timeslist[self.n + 1:]:


            if image is None:
                image = self.times[t]

            previous = image

            image = self.times[t]
            alignment = previous.channels.align_with_and_return_indices(image.channels)
            alignment_with_first = dict(image.channels.align_with_and_return_indices(self.first))

            for _, current_index in alignment:
                # ths is not perfectly right, but it's enough work with chan accum already
                self.cell_counts[alignment_with_first[current_index]].append(len(image.channels[current_index].cells))

                if t not in self.channel_accumulator[alignment_with_first[current_index]]:
                    self.channel_accumulator[alignment_with_first[current_index]][t] = image.channels[current_index]

            ignorant_next(progress_indicator)
        for index in self.channel_accumulator.keys():
            self.channel_accumulator[index] = [self.channel_accumulator[index][n]
                                               for n in sorted(self.channel_accumulator[index].keys())]

    def remove_empty_channels(self):
        cell_means = {k: float(sum(v)) / len(v) for k, v in self.cell_counts.items()}

        for k, mean_cellcount in cell_means.items():
            if mean_cellcount < 0.5:
                del self.tracker_mapping[k]
                del self.channel_accumulator[k]
                del self.cell_counts[k]

    def get_tracking_work_size(self):
        return sum([len(ca) - 1 if len(ca) > 0 else 0 for ca in self.channel_accumulator.values()])

    def perform_tracking(self, progress_indicator=dummy_progress_indicator):
        for c in self.tracker_mapping.keys():
            tracker = self.tracker_mapping[c]
            channel_list = self.channel_accumulator[c]

            if len(channel_list) == 0:
                continue

            previous = channel_list[0]

            for current in channel_list[1:]:
                tracker.tick()
                analyse_cell_fates(tracker, previous.cells, current.cells)
                ignorant_next(progress_indicator)
                previous = current

    def remove_empty_channels_post_tracking(self):
        minimum_average_cells = 2.0
        shouldskip = True
        for k, tracker in list(self.tracker_mapping.items()):
            if shouldskip and tracker.average_cells < minimum_average_cells:  # 0.5:
                del self.tracker_mapping[k]
                del self.channel_accumulator[k]
                del self.cell_counts[k]

    def perform_everything(self, times):
        self.set_times(times)
        self.align_channels()
        self.remove_empty_channels()
        self.perform_tracking()
        self.remove_empty_channels_post_tracking()
        return self


def analyse_cell_fates(tracker, pcells, ccells):
    outcome_it_is_same = lambda pcell, ccell: tracker.get_cell_by_observation(pcell[0]).add_observation(ccell[0])
    outcome_it_is_children = \
        lambda pcell, ccell: tracker.get_cell_by_observation(pcell[0]). \
            add_children(tracker.new_observed_cell(ccell[0]), tracker.new_observed_cell(ccell[1]))
    outcome_it_is_new = lambda pcell, ccell: tracker.new_observed_origin(ccell[0])
    outcome_it_is_lost = None  # lambda pcell, ccell: None

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

            opt.add_outcome(cost_same, {pc}, {cc}, outcome_it_is_same)

            large_value = 1000000.0

            cost_new_cell = 1.0 * large_value

            opt.add_outcome(cost_new_cell, set(), {cc}, outcome_it_is_new)

            cost_lost_cell = 1.0 * large_value
            opt.add_outcome(cost_lost_cell, {pc}, set(), outcome_it_is_lost)

            if n < len(ccells) - 1:
                cc1 = ccells[n + 0]
                cc2 = ccells[n + 1]

                cost_children = (calc_cost_child(pc, cc1) + calc_cost_child(pc, cc2)) / 2.0

                opt.add_outcome(cost_children, {pc}, {cc1, cc2}, outcome_it_is_children)

    opt.perform_optimal()

