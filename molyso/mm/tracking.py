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

        for image in self.times.values():
            if image.flattened:
                image.unflatten()

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
                self.cell_counts[alignment_with_first[current_index]].append(
                    len(image.channels.channels_list[current_index].cells))

                if t not in self.channel_accumulator[alignment_with_first[current_index]]:
                    self.channel_accumulator[alignment_with_first[current_index]][t] = image.channels.channels_list[
                        current_index]

            ignorant_next(progress_indicator)
        for index in self.channel_accumulator.keys():
            self.channel_accumulator[index] = [self.channel_accumulator[index][n]
                                               for n in sorted(self.channel_accumulator[index].keys())]

    def remove_empty_channels(self):
        cell_means = {k: (float(sum(v)) / len(v)) if len(v) > 0 else 0.0 for k, v in self.cell_counts.items()}

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


def analyse_cell_fates(tracker, previous_cells, current_cells):
    # original_current_cells = current_cells
    current_cells = current_cells.cells_list

    def outcome_it_is_same(previous_cell, current_cell):
        tracker.get_cell_by_observation(previous_cell).add_observation(current_cell)


    def outcome_it_is_children(previous_cell, current_cells):
        tracker.get_cell_by_observation(previous_cell). \
            add_children(tracker.new_observed_cell(current_cells[0]), tracker.new_observed_cell(current_cells[1]))


    def outcome_it_is_new(_, current_cell):
        tracker.new_observed_origin(current_cell)

    outcome_it_is_lost = None  # lambda pcell, ccell: None

    opt = CellCrossingCheckingGlobalDuoOptimizerQueue()

    try:
        trajs = [tracker.get_cell_by_observation(pc).trajectories[-1] for pc in previous_cells]
        if len(trajs) > 0:
            chan_traj = numpy.mean(trajs)
        else:
            chan_traj = 0.0
    except KeyError:
        chan_traj = 0.0

    for previous_number, previous_cell in enumerate(previous_cells):
        if not tracker.is_tracked(previous_cell):  # this probably only occurs on the first attempt
            tracker.new_observed_origin(previous_cell)
        for current_number, current_cell in enumerate(current_cells):
            tracked_previous_cell = tracker.get_cell_by_observation(previous_cell)

            trajectories = tracked_previous_cell.trajectories
            elongation_rates = tracked_previous_cell.elongation_rates

            last_traj = numpy.mean(trajectories[-min(len(trajectories), 5):])
            last_elo = numpy.mean(elongation_rates[-min(len(elongation_rates), 5):])

            time_delta = current_cell.channel.image.timepoint - previous_cell.channel.image.timepoint

            putative_shift = last_traj * time_delta
            putative_elongation = last_elo * time_delta

            #putative_shift = 0.0
            #putative_elongation = 0.0

            su = putative_shift + 0.5 * putative_elongation
            sd = putative_shift - 0.5 * putative_elongation

            def calc_cost_same(one_cell, other_cell):
                value = (abs(one_cell.top + su - other_cell.top)
                         + abs(one_cell.bottom + sd - other_cell.bottom)) / 2
                shrinkage = other_cell.length - one_cell.length
                if shrinkage < 0:
                    shrinkage *= 5
                value -= shrinkage

                value += 2 * (one_cell.centroid1dloc / 10.0) ** 1
                return 1.0 * value

            def calc_cost_children(one_cell, one_child, another_child):

                def calc_cost_child(one_cell, other_cell):
                    c1d = (one_cell.top + one_cell.bottom) / 2 + 0.5 * putative_elongation
                    value = 0.0
                    value += abs(one_cell.top + su - other_cell.top)
                    value += abs(c1d - other_cell.bottom)
                    value += abs(one_cell.bottom + sd - other_cell.bottom)
                    value += abs(c1d - other_cell.top)

                    value /= 4.0

                    shrinkage = other_cell.length - one_cell.length
                    if shrinkage > 0:
                        shrinkage *= 5
                    value += shrinkage
                    #value *= (onecell.centroid1dloc / 100.0)**1
                    value += 2 * (one_cell.centroid1dloc / 10.0) ** 1
                    return 1.5 * value

                return (calc_cost_child(one_cell, one_child) + calc_cost_child(one_cell, another_child)) / 2.0


            cost_same = calc_cost_same(previous_cell, current_cell)
            opt.add_outcome(cost_same, {previous_cell}, {current_cell}, outcome_it_is_same)

            large_value = 1000000.0

            cost_new_cell = 1.0 * large_value
            opt.add_outcome(cost_new_cell,
                            set(), {current_cell},
                            outcome_it_is_new)

            cost_lost_cell = 1.0 * large_value
            opt.add_outcome(cost_lost_cell,
                            {previous_cell}, set(),
                            outcome_it_is_lost)

            if current_number < len(current_cells) - 1:
                putative_first_child = current_cells[current_number + 0]
                putative_other_child = current_cells[current_number + 1]

                cost_children = calc_cost_children(previous_cell, putative_first_child, putative_other_child)

                opt.add_outcome(cost_children,
                                {previous_cell}, {putative_first_child, putative_other_child},
                                outcome_it_is_children)

    opt.perform_optimal()

