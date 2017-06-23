# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import logging

import numpy as np

from .tracking_infrastructure import CellTracker, CellCrossingCheckingGlobalDuoOptimizerQueue
from ..generic.signal import find_extrema_and_prominence, hamming_smooth
from ..generic.etc import ignorant_next, dummy_progress_indicator

from .tracking_output import *


def each_k_tracking_tracker_channels_in_results(tracking):
    """

    :param tracking:
    """
    for inner_k in sorted(tracking.tracker_mapping.keys()):
        tracker = tracking.tracker_mapping[inner_k]
        channels = tracking.channel_accumulator[inner_k]
        yield inner_k, tracking, tracker, channels


def each_pos_k_tracking_tracker_channels_in_results(inner_tracked_results):
    """

    :param inner_tracked_results:
    """
    for pos, outer_tracking in inner_tracked_results.items():
        for inner_k, inner_tracking, tracker, channels in each_k_tracking_tracker_channels_in_results(outer_tracking):
            yield pos, inner_k, inner_tracking, tracker, channels


class TrackedPosition(object):
    """
    A TrackedPosition object contains various CellTracker objects for each channel within a multipoint position,
    as well as other information associated with the position.
    """

    def __init__(self):
        self.times = None
        self.n = 0

        self.timeslist = []

        self.first = []

        self.tracker_mapping = {}
        self.channel_accumulator = {}
        self.cell_centroid_accumulator = {}
        self.cell_counts = {}

    # logger is not set as a regular instance variable,
    # as it would make the serialization of the class unpleasant ...
    @property
    def logger(self):
        return logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def set_times(self, times):
        """

        :param times:
        """
        self.times = times

        for image in self.times.values():
            if image.flattened:
                image.unflatten()

        self.find_first_valid_time()

    def find_first_valid_time(self):
        """
        Finds the first valid time point for the position.

        """
        self.timeslist = list(sorted(self.times.keys()))

        for self.n, t in enumerate(self.timeslist):
            first_time = t

            self.first = self.times[first_time].channels

            if len(self.first) > 0:
                break

            self.logger.info("Skipping channel")

        key_list = list(range(len(self.first)))

        self.tracker_mapping = {c: CellTracker() for c in key_list}
        self.channel_accumulator = {c: {} for c in key_list}
        self.cell_centroid_accumulator = {c: {} for c in key_list}
        self.cell_counts = {c: [] for c in key_list}

    def align_channels(self, progress_indicator=dummy_progress_indicator()):
        """

        :param progress_indicator:
        """
        image = None

        for _ in range(self.n):
            ignorant_next(progress_indicator)

        for t in self.timeslist[self.n:]:
            if image is None:
                image = self.times[t]

            previous = image

            image = self.times[t]
            alignment = previous.channels.align_with_and_return_indices(image.channels)
            alignment_with_first = dict(image.channels.align_with_and_return_indices(self.first))

            for _, current_index in alignment:
                # ths is not perfectly right, but it's enough work with chan accumulator already
                self.cell_counts[alignment_with_first[current_index]].append(
                    len(image.channels.channels_list[current_index].cells))

                centroid_accumulator = self.cell_centroid_accumulator[alignment_with_first[current_index]]
                for cell in image.channels.channels_list[current_index].cells:
                    centroid = int(round(cell.centroid_1d))
                    if centroid in centroid_accumulator:
                        centroid_accumulator[centroid] += 1
                    else:
                        centroid_accumulator[centroid] = 1

                if t not in self.channel_accumulator[alignment_with_first[current_index]]:
                    self.channel_accumulator[alignment_with_first[current_index]][t] = image.channels.channels_list[
                        current_index]

            ignorant_next(progress_indicator)

        for index in self.channel_accumulator.keys():
            self.channel_accumulator[index] = [self.channel_accumulator[index][n]
                                               for n in sorted(self.channel_accumulator[index].keys())]

    def remove_empty_channels(self):
        """
        Removes empty channels from the data set.

        """
        cell_means = {k: (float(sum(v)) / len(v)) if len(v) > 0 else 0.0 for k, v in self.cell_counts.items()}

        for k, mean_cell_count in cell_means.items():
            if mean_cell_count < tunable('tracking.empty_channel_filtering.minimum_mean_cells', 2.0,
                                         description="For empty channel removal, minimum of cell mean per channel."):
                del self.tracker_mapping[k]
                del self.channel_accumulator[k]
                del self.cell_counts[k]
                del self.cell_centroid_accumulator[k]

    def guess_channel_orientation(self):
        """
        Tries to guess the channel orientation.
        1 if the closed end ('mother side') is the high coordinates,
        -1 if low ...

        """
        for channel_num in self.channel_accumulator.keys():
            cells_in_channel = self.cell_centroid_accumulator[channel_num]
            minpos = min(cells_in_channel.keys())
            signal = np.zeros(max(cells_in_channel.keys()) + 1 - minpos)
            for pos, times in cells_in_channel.items():
                signal[pos - minpos] = times

            signal_len = len(signal)

            # noinspection PyTypeChecker
            helper_parabola = np.linspace(-signal_len / 2, +signal_len / 2, signal_len) ** 2 / signal_len ** 2

            signal = hamming_smooth(signal, 15)
            signal *= helper_parabola

            try:
                extrema = find_extrema_and_prominence(signal)
                maxy = extrema.signal[extrema.maxima]

                centroid = np.sum(extrema.maxima * maxy) / np.sum(maxy)

                # mean = np.mean(maxy)
                result = 1 if centroid >= signal.size / 2 else -1
            except IndexError:
                result = 0

            for channel in self.channel_accumulator[channel_num]:
                channel.putative_orientation = result

    def get_tracking_work_size(self):
        """


        :return:
        """
        return sum([len(ca) - 1 if len(ca) > 0 else 0 for ca in self.channel_accumulator.values()])

    def perform_tracking(self, progress_indicator=dummy_progress_indicator()):
        """

        :param progress_indicator:
        """
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
        """
        Removes empty channels after tracking.

        """
        minimum_average_cells = tunable('tracking.empty_channel_filtering.minimum_mean_cells', 2.0,
                                        description="For empty channel removal, minimum of cell mean per channel.")
        should_skip = True
        for k, tracker in list(self.tracker_mapping.items()):
            if should_skip and tracker.average_cells < minimum_average_cells:
                del self.tracker_mapping[k]
                del self.channel_accumulator[k]
                del self.cell_counts[k]

    def perform_everything(self, times):
        """

        :param times:
        :return:
        """
        self.set_times(times)
        self.align_channels()
        self.remove_empty_channels()
        self.perform_tracking()
        self.remove_empty_channels_post_tracking()
        return self


def analyse_cell_fates(tracker, previous_cells, current_cells):
    # original_current_cells = current_cells
    """

    :param tracker:
    :param previous_cells:
    :param current_cells:
    :return:
    """
    current_cells = current_cells.cells_list

    def outcome_it_is_same(the_previous_cell, the_current_cell):
        """

        :param the_previous_cell:
        :param the_current_cell:
        """
        tracker.get_cell_by_observation(the_previous_cell).add_observation(the_current_cell)

    def outcome_it_is_children(the_previous_cell, the_current_cells):
        """

        :param the_previous_cell:
        :param the_current_cells:
        """
        tracker.get_cell_by_observation(the_previous_cell). \
            add_children(tracker.new_observed_cell(the_current_cells[0]),
                         tracker.new_observed_cell(the_current_cells[1]))

    def outcome_it_is_new(_, the_current_cell):
        """

        :param _:
        :param the_current_cell:
        """
        tracker.new_observed_origin(the_current_cell)

    outcome_it_is_lost = None

    opt = CellCrossingCheckingGlobalDuoOptimizerQueue()

    for previous_number, previous_cell in enumerate(previous_cells):
        if not tracker.is_tracked(previous_cell):  # this probably only occurs on the first attempt
            tracker.new_observed_origin(previous_cell)
        for current_number, current_cell in enumerate(current_cells):
            tracked_previous_cell = tracker.get_cell_by_observation(previous_cell)

            trajectories = tracked_previous_cell.trajectories
            elongation_rates = tracked_previous_cell.elongation_rates

            last_traj = np.mean(trajectories[-min(len(trajectories), 5):])
            last_elo = np.mean(elongation_rates[-min(len(elongation_rates), 5):])

            time_delta = current_cell.channel.image.timepoint - previous_cell.channel.image.timepoint

            putative_shift = last_traj * time_delta
            putative_elongation = last_elo * time_delta

            # putative_shift, putative_elongation = 0.0, 0.0

            shift_upper = putative_shift + 0.5 * putative_elongation
            shift_lower = putative_shift - 0.5 * putative_elongation

            shrinkage_penalty = 5.0

            large_value = 1000000.0
            cost_new_cell = 1.0 * large_value
            cost_lost_cell = 1.0 * large_value

            def calc_cost_same(one_cell, other_cell):
                """

                :param one_cell:
                :param other_cell:
                :return:
                """
                cost = \
                    0.5 * abs(one_cell.top + shift_upper - other_cell.top) + \
                    0.5 * abs(one_cell.bottom + shift_lower - other_cell.bottom)

                shrinkage = other_cell.length - (one_cell.length + putative_elongation)

                cost -= shrinkage * shrinkage_penalty if shrinkage < 0.0 else 0.0
                return cost

            def calc_cost_children(one_cell, upper_child, lower_child):

                """

                :param one_cell:
                :param upper_child:
                :param lower_child:
                :return:
                """
                one_cell_centroid = 0.5 * one_cell.top + 0.5 * one_cell.bottom + 0.5 * putative_elongation

                cost = \
                    0.5 * abs(one_cell.top + shift_upper - upper_child.top) + \
                    0.5 * abs(one_cell_centroid - upper_child.bottom) + \
                    0.5 * abs(one_cell_centroid - lower_child.top) + \
                    0.5 * abs(one_cell.bottom - lower_child.bottom)

                shrinkage = (one_cell.length + putative_elongation) - upper_child.length - lower_child.length

                cost += shrinkage * shrinkage_penalty if shrinkage > 0.0 else 0.0
                return cost

            opt.add_outcome(calc_cost_same(previous_cell, current_cell),
                            {previous_cell}, {current_cell},
                            outcome_it_is_same)

            opt.add_outcome(cost_new_cell,
                            set(), {current_cell},
                            outcome_it_is_new)

            opt.add_outcome(cost_lost_cell,
                            {previous_cell}, set(),
                            outcome_it_is_lost)

            if current_number < len(current_cells) - 1:
                putative_first_child = current_cells[current_number + 0]
                putative_other_child = current_cells[current_number + 1]

                opt.add_outcome(calc_cost_children(previous_cell, putative_first_child, putative_other_child),
                                {previous_cell}, {putative_first_child, putative_other_child},
                                outcome_it_is_children)

    opt.perform_optimal()

