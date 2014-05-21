# -*- coding: utf-8 -*-
"""
documentation
"""
from __future__ import division, unicode_literals, print_function

import numpy
import itertools


class CellTracker(object):
    """

    """

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
    def __init__(self, tracker):
        self.tracker = tracker
        self.parent = None
        self.children = []  # [None, None]

        self.seen_as = []

        self.raw_elongation_rates = [0.0]
        self.raw_trajectories = [0.0]

    @property
    def elongation_rates(self):
        if self.parent:
            return self.parent.elongation_rates + self.raw_elongation_rates
        else:
            return self.raw_elongation_rates

    @property
    def trajectories(self):
        if self.parent:
            return self.parent.trajectories + self.raw_trajectories
        else:
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
                (current.centroid1dloc - previous.centroid1dloc) /
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

    # forgive me ...
    # hacking this together quickly as I want to see if the crossing checking
    # helps somewhat significantly
    # its totally unabstracted (and unoptimized). bleargh!

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

        costs = numpy.zeros(rows)

        actions = [()] * rows

        for i, (cost, (involved_a, involved_b, what)) in enumerate(
                sorted(self.data, key=lambda x: (x[0], x[1][0], x[1][1]))):
            actions[i] = (involved_a, involved_b, what)

            for a in involved_a:
                matrix[i][lookup_a[a]] = True

            for b in involved_b:
                matrix[i][lookup_b[b] + len_a] = True

            costs[i] = cost

        baseline = []
        unused = []

        collector = numpy.zeros(cols, dtype=bool)
        accumulator = 0.0

        def would_that_work_without_crossings(used_rows, current_row):
            # wow, so bruteforce.
            def row_to_inuse(n):
                tmp_a = []
                tmp_b = []
                for num, v in enumerate(matrix[n]):
                    if v:
                        if num < len_a:
                            tmp_a.append(ordered_a[num])
                        else:
                            tmp_b.append(ordered_b[num - len_a])
                return itertools.product(tmp_a, tmp_b)

            tocheck = row_to_inuse(current_row)

            for ur in used_rows:
                for a, b in row_to_inuse(ur):
                    for ta, tb in tocheck:
                        if (a.centroid1dloc < ta.centroid1dloc and b.centroid1dloc > tb.centroid1dloc) or \
                                (a.centroid1dloc > ta.centroid1dloc and b.centroid1dloc < tb.centroid1dloc):
                            return False

            return True


        for w in range(rows):
            row = matrix[w]
            cost = costs[w]

            if (~(row & collector)).all() and would_that_work_without_crossings(baseline, w):
                collector |= row
                accumulator += cost
                baseline += [w]
            else:
                unused += [w]

        current = baseline[:]
        current_cost = accumulator

        things_to_try = sorted([(costs[c], c) for c in current], reverse=True)
        #shuffle(things_to_try)

        self.debug_output += "start opti = %f\n" % current_cost
        iteration_count = 1  # 10
        for iteration in range(iteration_count):


            new_try = current[:]

            new_collector = collector
            new_unused = unused[:]
            new_accumulator = new_cost = current_cost

            for n in range(2):
                try:
                    me_cost, me_item = things_to_try.pop(0)
                except IndexError:
                    break

                new_try.remove(me_item)
                new_collector -= matrix[me_item]
                new_accumulator -= me_cost
                new_cost -= me_cost

                new_unused += [me_item]

                unused += [me_item]

            change = []

            collector_backup = collector.copy()
            accumulator_backup = accumulator

            for w in unused:
                row = matrix[w]
                cost = costs[w]

                if (~(row & new_collector)).all() and would_that_work_without_crossings(change + new_try, w):
                    new_collector |= row
                    new_accumulator += cost
                    change += [w]
                    new_unused.remove(w)


            #print("iteration = %d" % (iteration,))
            #print("%d: accumulator (%f) < current_cost (%f) " % (iteration, accumulator, current_cost,))
            if new_accumulator < current_cost:
                self.debug_output += "      opti = %f\n" % new_accumulator
                current_cost = new_accumulator
                collector = new_collector
                current = new_try + change
                unused = new_unused
                things_to_try = sorted([(costs[c], c) for c in current], reverse=True)
                #shuffle(things_to_try)


        ####

        if False:

            header = ["Cost"] + ["1"] + ["O"] + ["What"] + ["A"] + [str(s) for s in range(1, len_a + 1)] \
                     + ["B"] + [str(s) for s in range(1, len_b + 1)]
            self.debug_output += "\t".join([str(s) for s in header])
            self.debug_output += "\n"
            #)
            for n in range(rows):
                trow = [costs[n]] + ["*" if n in baseline else ""] + ["*" if n in current else ""] + [
                    actions[n][2].func_name] + [""] + list(matrix[n][0:len_a]) + [""] + list(matrix[n][len_a:])  #)
                self.debug_output += "\t".join([str(s) for s in trow])
                self.debug_output += "\n"

        ###

        for c in current:
            involved_a, involved_b, what = actions[c]
            if what is not None:
                what(list(involved_a), list(involved_b))
