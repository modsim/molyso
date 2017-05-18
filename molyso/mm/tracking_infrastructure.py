# -*- coding: utf-8 -*-
"""
This module contains cell tracking infrastructure.
"""
from __future__ import division, unicode_literals, print_function

import numpy as np


class CellTracker(object):
    """
    The CellTracker contains all tracks of a channel.

    """
    __slots__ = ['all_tracked_cells', 'origins', 'timepoints']

    def __init__(self):
        self.all_tracked_cells = {}
        self.origins = []
        self.timepoints = 0

    def tick(self):
        """
        Ticks the clock. Sets the internal timepoint counter forward by one.

        """
        self.timepoints += 1

    @property
    def average_cells(self):
        """
        Returns the average count of cells present in this tracked channel.

        :return:
        """
        if self.timepoints > 0:
            return float(len(self.all_tracked_cells)) / self.timepoints
        else:
            return 0.0

    def new_cell(self):
        """
        Creates a new TrackedCell object associated with this tracker.

        :return:
        """
        return TrackedCell(self)

    def new_observed_cell(self, where):
        """
        Creates a new TrackedCell object, with added observation.

        :param where:
        :return:
        """
        return self.new_cell().add_observation(where)

    def new_origin(self):
        """
        Creates a new TrackedCell object and adds it as an origin.

        :return:
        """
        t = self.new_cell()
        self.origins.append(t)
        return t

    def new_observed_origin(self, where):
        """
        Creates a new TrackedCell object and adds it as an origin, with added observation.
        :param where:
        :return:
        """
        return self.new_origin().add_observation(where)

    def is_tracked(self, cell):
        """
        Returns whether the cell is tracked.

        :param cell:
        :return:
        """
        return cell in self.all_tracked_cells

    def get_cell_by_observation(self, where):
        """
        Returns the associated cell by its observation.

        :param where:
        :return:
        """
        return self.all_tracked_cells[where]


class TrackedCell(object):
    """

    :param tracker:
    """
    __slots__ = ['tracker', 'parent', 'children', 'seen_as', 'raw_elongation_rates', 'raw_trajectories']

    def __init__(self, tracker):
        self.tracker = tracker
        self.parent = None
        self.children = []  # [None, None]

        self.seen_as = []

        self.raw_elongation_rates = [0.0]
        self.raw_trajectories = [0.0]

    @property
    def ultimate_parent(self):
        """


        :return:
        """
        if self.parent is None:
            return self
        else:
            return self.parent.ultimate_parent

    @property
    def elongation_rates(self):
        # if self.parent:
        # return self.parent.elongation_rates + self.raw_elongation_rates
        # else:
        #     return self.raw_elongation_rates
        """


        :return:
        """
        return self.raw_elongation_rates

    @property
    def trajectories(self):
        # if self.parent:
        # return self.parent.trajectories + self.raw_trajectories
        # else:
        #     return self.raw_trajectories
        """


        :return:
        """
        return self.raw_trajectories

    def add_child(self, tcell):
        """

        :param tcell:
        :return:
        """
        tcell.parent = self
        self.children.append(tcell)
        return self

    def add_children(self, *children):
        """

        :param children:
        """
        for child in children:
            self.add_child(child)

    def add_observation(self, cell):
        """

        :param cell:
        :return:
        """
        self.seen_as.append(cell)
        self.tracker.all_tracked_cells[cell] = self

        if len(self.seen_as) > 1:
            current = self.seen_as[-1]
            previous = self.seen_as[-2]

            assert (current != previous)

            self.raw_elongation_rates.append(
                (current.length - previous.length) /
                (current.channel.image.timepoint - previous.channel.image.timepoint))
            self.raw_trajectories.append(
                (current.centroid_1d - previous.centroid_1d) /
                (current.channel.image.timepoint - previous.channel.image.timepoint))

        return self


def to_list(x):
    """

    :param x:
    :return:
    """
    return x if type(x) == list else [x]


class CellCrossingCheckingGlobalDuoOptimizerQueue(object):
    """

    """

    def __init__(self):
        self.set_a = set()
        self.set_b = set()

        self.data = []

        self.run = []

        self.debug_output = ''

    def add_outcome(self, cost, involved_a, involved_b, what):
        # nan check
        """

        :param cost:
        :param involved_a:
        :param involved_b:
        :param what:
        :return:
        """
        if cost != cost:
            return

        self.data.append((cost, (involved_a, involved_b, what)))

        self.set_a |= involved_a
        self.set_b |= involved_b

    def perform_optimal(self):
        """


        :return:
        """
        ordered_a = list(sorted(self.set_a))
        ordered_b = list(sorted(self.set_b))

        lookup_a = {i: n for n, i in enumerate(ordered_a)}
        lookup_b = {i: n for n, i in enumerate(ordered_b)}

        len_a = len(ordered_a)
        len_b = len(ordered_b)

        rows = len(self.data)
        cols = len(ordered_a) + len(ordered_b)

        matrix = np.zeros((rows, cols,), dtype=bool)

        dependencies = np.zeros((rows, len_a, len_b,), dtype=int)

        costs = np.zeros(rows)

        data = sorted(self.data, key=lambda x: (x[0], x[1][0], x[1][1]))

        for i, (cost, (involved_a, involved_b, what)) in enumerate(data):

            for a in involved_a:
                matrix[i, lookup_a[a]] = True

            for b in involved_b:
                matrix[i, lookup_b[b] + len_a] = True

            for a in involved_a:
                for b in involved_b:
                    dependencies[i, lookup_a[a], lookup_b[b]] = 1

            costs[i] = cost

        def crossing_check(used_rows, row_to_add):
            """

            :param used_rows:
            :param row_to_add:
            :return:
            """
            local_used_rows = used_rows.copy()
            local_used_rows[row_to_add] = True
            summed_deps = dependencies[local_used_rows, :, :].sum(axis=0)

            last = -1

            for m in range(len_a):
                non_zero_positions, = np.nonzero(summed_deps[m, :])
                if len(non_zero_positions) == 0:
                    continue
                if non_zero_positions[0] > last:
                    last = non_zero_positions[0]
                else:
                    return False
            return True

        used = np.zeros(rows, dtype=bool)
        collector = np.zeros(cols, dtype=bool)
        cost_accumulator = 0.0

        for w in range(rows):
            matrix_row = matrix[w, :]
            if (~(matrix_row & collector)).all() and crossing_check(used, w):
                collector |= matrix_row
                cost_accumulator += costs[w]
                used[w] = True

        for c, (_, (involved_a, involved_b, what)) in enumerate(data):
            if used[c] and what:
                involved_a = None if len(involved_a) == 0 else \
                    (next(iter(involved_a)) if len(involved_a) == 1 else list(involved_a))
                involved_b = None if len(involved_b) == 0 else \
                    (next(iter(involved_b)) if len(involved_b) == 1 else list(involved_b))

                what(involved_a, involved_b)
