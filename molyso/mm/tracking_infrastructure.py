# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy


class CellTracker(object):
    """

    """

    __slots__ = ['all_tracked_cells', 'origins', 'timepoints']

    def __init__(self):
        self.all_tracked_cells = {}
        self.origins = []
        self.timepoints = 0

    def tick(self):
        self.timepoints += 1

    @property
    def average_cells(self):
        if self.timepoints > 0:
            return float(len(self.all_tracked_cells)) / self.timepoints
        else:
            return 0.0

    def new_cell(self):
        return TrackedCell(self)

    def new_observed_cell(self, where):
        return self.new_cell().add_observation(where)

    def new_origin(self):
        t = self.new_cell()
        self.origins.append(t)
        return t

    def new_observed_origin(self, where):
        return self.new_origin().add_observation(where)

    def is_tracked(self, cell):
        return cell in self.all_tracked_cells

    def get_cell_by_observation(self, where):
        return self.all_tracked_cells[where]


class TrackedCell(object):
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
        return self.raw_elongation_rates

    @property
    def trajectories(self):
        # if self.parent:
        # return self.parent.trajectories + self.raw_trajectories
        # else:
        #     return self.raw_trajectories
        return self.raw_trajectories

    def add_child(self, tcell):
        tcell.parent = self
        self.children.append(tcell)
        return self

    def add_children(self, *children):
        for child in children:
            self.add_child(child)

    def add_observation(self, cell):
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
    return x if type(x) == list else [x]


class CellCrossingCheckingGlobalDuoOptimizerQueue(object):
    def __init__(self):
        self.set_a = set()
        self.set_b = set()

        self.data = []

        self.run = []

        self.debug_output = ''

    def add_outcome(self, cost, involved_a, involved_b, what):
        # nan check
        if cost != cost:
            return

        self.data.append((cost, (involved_a, involved_b, what)))

        self.set_a |= involved_a
        self.set_b |= involved_b

    def perform_optimal(self):
        ordered_a = list(sorted(self.set_a))
        ordered_b = list(sorted(self.set_b))

        lookup_a = {i: n for n, i in enumerate(ordered_a)}
        lookup_b = {i: n for n, i in enumerate(ordered_b)}

        len_a = len(ordered_a)
        len_b = len(ordered_b)

        rows = len(self.data)
        cols = len(ordered_a) + len(ordered_b)

        matrix = numpy.zeros((rows, cols,), dtype=bool)

        dependencies = numpy.zeros((rows, len_a, len_b,), dtype=int)

        costs = numpy.zeros(rows)

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
            local_used_rows = used_rows.copy()
            local_used_rows[row_to_add] = True
            summed_deps = dependencies[local_used_rows, :, :].sum(axis=0)

            last = -1

            for m in range(len_a):
                non_zero_positions, = numpy.nonzero(summed_deps[m, :])
                if len(non_zero_positions) == 0:
                    continue
                if non_zero_positions[0] > last:
                    last = non_zero_positions[0]
                else:
                    return False
            return True

        used = numpy.zeros(rows, dtype=bool)
        collector = numpy.zeros(cols, dtype=bool)
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
