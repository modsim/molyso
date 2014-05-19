class CellTracker(object):
    def __init__(self):
        self.all_tracked_cells = {}
        self.origins = []
        self.timepoints = 0

    def tick(self):
        self.timepoints += 1

    @property
    def average_cells(self):
        try:
            return float(len(self.all_tracked_cells)) / self.timepoints
        except ZeroDivisionError:
            return 0.0

    def new_origin(self):
        t = self.new_cell()
        self.origins += [t]
        return t

    def new_cell(self):
        t = TrackedCell(self)
        return t

    def new_observed_cell(self, where):
        t = self.new_cell()
        t.add_observation(where)
        return t

    def new_observed_origin(self, where):
        t = self.new_origin()
        t.add_observation(where)
        return t

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

        self.raw_elongation_rates = []
        self.raw_trajectories = []

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
        self.children += [tcell]

    def add_observation(self, cell):
        self.seen_as += [cell]
        try:
            if self.seen_as[-2] == self.seen_as[-1]:
                self.raw_elongation_rates.append(0.0)
                self.raw_trajectories.append(0.0)
                return  # wtf?
        except IndexError:
            pass
            #print(self.seen_as)
        self.tracker.all_tracked_cells[cell] = self
        try:
            self.raw_elongation_rates += [
                (self.seen_as[-1].length - self.seen_as[-2].length) /
                (self.seen_as[-1].channel.image.timepoint - self.seen_as[-2].channel.image.timepoint)]
        except (IndexError, ZeroDivisionError):
            self.raw_elongation_rates.append(0.0)

        try:
            self.raw_trajectories += [
                (self.seen_as[-1].centroid1dloc - self.seen_as[-2].centroid1dloc) /
                (self.seen_as[-1].channel.image.timepoint - self.seen_as[-2].channel.image.timepoint)]
        except (IndexError, ZeroDivisionError):
            self.raw_trajectories.append(0.0)


import numpy


class CellCrossingCheckingGlobalDuoOptimizerQueue(object):
    def __init__(self):
        self.set_a = set()
        self.set_b = set()

        self.data = []

        self.run = []

        self.debug_output = ""

    def _aslist(self, something):
        if type(something) == list:
            return something
        else:
            return [something]

    def add_outcome(self, cost, involved_a, involved_b, what):

        # nan check
        if cost != cost:
            return

        #if cost > 100:
        #    return
        self.data.append((cost, (involved_a, involved_b, what)))

        involved_a = self._aslist(involved_a)
        involved_b = self._aslist(involved_b)

        for a in involved_a:
            self.set_a.add(a)
        for b in involved_b:
            self.set_b.add(b)


    def getlookups(self):
        alookup = {v: k + 1 for k, v in
                   enumerate(c for pos, c in sorted((c.centroid1dloc, c) for c in self.set_a)[::-1])}
        blookup = {v: k + 1 for k, v in
                   enumerate(c for pos, c in sorted((c.centroid1dloc, c) for c in self.set_b)[::-1])}
        return alookup, blookup

    # forgive me ...
    # hacking this together quickly as I want to see if the crossing checking
    # helps somewhat significantly
    # its totally unabstracted (and unoptimized). bleargh!

    def perform_optimal(self, dryRun=False):
        orderedA = list(sorted(self.set_a))
        orderedB = list(sorted(self.set_b))

        lookupA = {i: n for n, i in enumerate(orderedA)}
        lookupB = {i: n for n, i in enumerate(orderedB)}

        lena = len(orderedA)
        lenb = len(orderedB)

        rows = len(self.data)
        cols = len(orderedA) + len(orderedB)

        matrix = numpy.zeros((rows, cols), dtype=bool)

        costs = numpy.zeros(rows)

        actions = [()] * rows

        to_list = lambda x: x if type(x) == list else [x]

        for i, (cost, (involved_a, involved_b, what)) in enumerate(
                sorted(self.data, key=lambda x: (x[0], to_list(x[1][0]), to_list(x[1][1])))):
            actions[i] = (involved_a, involved_b, what)

            for a in self._aslist(involved_a):
                matrix[i][lookupA[a]] = True

            for b in self._aslist(involved_b):
                matrix[i][lookupB[b] + lena] = True

            costs[i] = cost

        baseline = []
        unused = []

        collector = numpy.zeros(cols, dtype=bool)
        accumulator = 0.0

        import itertools

        def would_that_work_without_crossings(used_rows, current_row):
            # wow, so bruteforce.
            def row_to_inuse(n):
                a = []
                b = []
                for num, v in enumerate(matrix[n]):
                    if v:
                        if num < lena:
                            a += [orderedA[num]]
                        else:
                            b += [orderedB[num - lena]]
                return list(itertools.product(a, b))

            tocheck = row_to_inuse(current_row)

            for ur in used_rows:
                for a, b in row_to_inuse(ur):
                    for ta, tb in tocheck:
                        if a.centroid1dloc < ta.centroid1dloc and b.centroid1dloc > tb.centroid1dloc:
                            #print("it has happened")
                            return False
                        if a.centroid1dloc > ta.centroid1dloc and b.centroid1dloc < tb.centroid1dloc:
                            #print("it has happened, a bit diffrently")
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

            header = ["Cost"] + ["1"] + ["O"] + ["What"] + ["A"] + [str(s) for s in range(1, lena + 1)] \
                     + ["B"] + [str(s) for s in range(1, lenb + 1)]
            self.debug_output += "\t".join([str(s) for s in header])
            self.debug_output += "\n"
            #)
            for n in range(rows):
                trow = [costs[n]] + ["*" if n in baseline else ""] + ["*" if n in current else ""] + [
                    actions[n][2].func_name] + [""] + list(matrix[n][0:lena]) + [""] + list(matrix[n][lena:])  #)
                self.debug_output += "\t".join([str(s) for s in trow])
                self.debug_output += "\n"

        ###

        for c in current:
            involved_a, involved_b, what = actions[c]
            what(involved_a, involved_b)
