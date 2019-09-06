# -*- coding: utf-8 -*-
"""
etc.py contains various helper functions and classes, which are not directly related to data processing.
"""
from __future__ import division, unicode_literals, print_function
import os
import sys
import hashlib
import time
import logging
import sqlite3

import numpy as np

from io import BytesIO
from ..debugging import DebugPlot

try:
    import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger(__name__)


def silent_progress_bar(iterable):
    """
    Dummy function, just returns an iterator.

    :param iterable: the iterable to turn into an iterable
    :type iterable: iterable
    :return: iterable
    :rtype: iterable

    >>> next(silent_progress_bar([1, 2, 3]))
    1
    """
    return iter(iterable)


def fancy_progress_bar(iterable):
    """
    Returns in iterator which will show progress as well.
    Will either use the tqdm module when available, or a simpler implementation.

    :param iterable: the iterable to progress-ify
    :type iterable: iterable
    :rtype: iterable
    :return: progress-ified iterable
    """

    if tqdm:
        # weird bug: if the threading magic in tqdm is active, multiprocessing in molyso gets stuck!
        # should be investigated further, but for now, let us just disable it ...

        tqdm.tqdm.monitor_interval = 0

        for item in tqdm.tqdm(iterable):
            yield item
    else:
        times = np.zeros(len(iterable), dtype=float)
        for n, i in enumerate(iterable):
            start_time = time.time()
            yield i
            stop_time = time.time()
            times[n] = stop_time - start_time
            eta = " ETA %.2fs" % float(np.mean(times[:n + 1]) * (len(iterable) - (n + 1)))
            logger.info("processed %d/%d [took %.3fs%s]" % (n + 1, len(iterable), times[n], eta))


def iter_time(iterable):
    """
    Will print the total time elapsed during iteration of ``iterable`` afterwards.

    :param iterable: iterable
    :type iterable: iterable
    :rtype: iterable
    :return: iterable
    """
    start_time = time.time()
    for n in iterable:
        yield n
    stop_time = time.time()
    logger.info("whole step took %.3fs" % (stop_time - start_time,))


_fancy_progress_bar = fancy_progress_bar


def fancy_progress_bar(iterable):
    """

    :param iterable:
    :return:
    """
    return iter_time(_fancy_progress_bar(iterable))


def dummy_progress_indicator():
    """


    :return:
    """
    return iter(int, 1)


def ignorant_next(iterable):
    """
    Will try to iterate to the next value, or return None if none is available.

    :param iterable:
    :return:
    """
    try:
        return next(iterable)
    except StopIteration:
        return None


class QuickTableDumper(object):

    """

    :param recipient:
    """
    delimiter = '\t'
    line_end = '\n'

    precision = 8

    def __init__(self, recipient=None):
        if recipient is None:
            recipient = sys.stdout

        self.recipient = recipient
        self.headers = []

    def write_list(self, the_list):
        """

        :param the_list:
        """
        self.recipient.write(self.delimiter.join(map(self.stringify, the_list)) + self.line_end)

    def add(self, row):
        """

        :param row:
        """
        if len(self.headers) == 0:
            self.headers = list(sorted(row.keys()))
            self.write_list(self.headers)

        self.write_list(row[k] for k in self.headers)

    def stringify(self, obj):
        """

        :param obj:
        :return:
        """
        if type(obj) in (float, np.float32, np.float64) and self.precision:
            return str(round(obj, self.precision))
        else:
            return str(obj)


try:
    # noinspection PyUnresolvedReferences
    import cPickle

    pickle = cPickle
except ImportError:
    import pickle

try:
    import _thread
except ImportError:
    import thread as _thread

if os.name != 'nt':
    def correct_windows_signal_handlers():
        """
        Dummy for non-windows os.
        """
        pass
else:
    def correct_windows_signal_handlers():
        """
        Corrects Windows signal handling, otherwise multiprocessing solutions will not correctly
        exit if Ctrl-C is used to interrupt them.

        :return:
        """
        os.environ['PATH'] += os.path.pathsep + os.path.dirname(os.path.abspath(sys.executable))

        try:
            # noinspection PyUnresolvedReferences
            import win32api

            def _handler(_, hook=_thread.interrupt_main):
                hook()
                return 1

            win32api.SetConsoleCtrlHandler(_handler, 1)

        except ImportError:
            logger.warning("Running on Windows, but module 'win32api' could not be imported to fix signal handler.\n" +
                           "Ctrl-C might break the program ..." +
                           "Fix: Install the module!")


def debug_init():
    """
    Initialized debug mode, as of now this means that DebugPlot is set to active (it will produce a debug.pdf)

    """
    DebugPlot.force_active = True
    np.set_printoptions(threshold=sys.maxsize)


def parse_range(s, maximum=0):
    """

    :param s:
    :param maximum:
    :return:
    """
    maximum -= 1
    splits = s.replace(' ', '').replace(';', ',').split(',')

    ranges = []
    remove = []

    not_values = False

    for frag in splits:
        if frag[0] == '~':
            not_values = not not_values
            frag = frag[1:]

        if '-' in frag:
            f, t = frag.split('-')

            interval = 1

            if '%' in t:
                t, _interval = t.split('%')
                interval = int(_interval)

            if t == '':
                t = maximum

            f, t = int(f), int(t)

            t = min(t, maximum)

            parsed_fragment = range(f, t + 1, interval)
        else:
            parsed_fragment = [int(frag)]

        if not_values:
            remove += parsed_fragment
        else:
            ranges += parsed_fragment

    return list(sorted(set(ranges) - set(remove)))


def prettify_numpy_array(arr, space_or_prefix):
    """
    Returns a properly indented string representation of a numpy array.

    :param arr:
    :param space_or_prefix:
    :return:
    """
    six_spaces = ' ' * 6
    prepared = repr(np.array(arr)).replace(')', '').replace('array(', six_spaces)
    if isinstance(space_or_prefix, int):
        return prepared.replace(six_spaces, ' ' * space_or_prefix)
    else:
        return space_or_prefix + prepared.replace(six_spaces, ' ' * len(space_or_prefix)).lstrip()


def bits_to_numpy_type(bits):
    """
    Returns a numpy.dtype for one of the common image bit-depths:
    8 for unsigned int, 16 for unsigned short, 32 for float

    :param bits:
    :return:
    """
    # this is supposed to throw an error
    return {
        8: np.uint8,
        16: np.uint16,
        32: np.float32
    }[int(bits)]


class BaseCache(object):
    """
    A caching class
    """

    @staticmethod
    def prepare_key(key):
        """

        :param key:
        :return:
        """
        if isinstance(key, type('')):
            return key
        else:
            return repr(key)

    @staticmethod
    def serialize(data):
        """

        :param data:
        :return:
        """
        try:
            bio = BytesIO()
            pickle.dump(data, bio, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                # noinspection PyUnresolvedReferences
                pickled_data = bio.getbuffer()
            except AttributeError:
                pickled_data = bio.getvalue()
        except ImportError:
            pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        return pickled_data

    @staticmethod
    def deserialize(data):
        """

        :param data:
        :return:
        """
        assert data is not None
        bio = BytesIO(data)
        return pickle.load(bio)

    def __init__(self, filename_to_be_hashed, ignore_cache='nothing', cache_token=None):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.filename_hash_source = filename_to_be_hashed

        if cache_token is None:
            self.cache_token = "%s.%s" % (
                os.path.basename(filename_to_be_hashed).replace('.', '_').replace('?', '_').replace(',', '_'),
                hashlib.sha1(str(os.path.abspath(filename_to_be_hashed).lower()).encode()).hexdigest()[:8])
        else:
            self.cache_token = cache_token

        if ignore_cache == 'everything':
            self.ignore_cache = True
        elif ignore_cache == 'nothing':
            self.ignore_cache = []
        else:
            self.ignore_cache = ignore_cache.split(',')

    def contains(self, key):
        """

        :param key:
        :return:
        """
        return False

    def get(self, key):
        """

        :param key:
        :return:
        """
        return ''

    def set(self, key, value):
        """

        :param key:
        :param value:
        :return:
        """
        return

    def __contains__(self, key):
        if self.ignore_cache is True or key in self.ignore_cache:
            return False
        else:
            try:
                self.logger.debug("Checking whether '%s' exists", key)
                return self.contains(self.prepare_key(key))
            except Exception as e:
                self.logger.exception(
                    "While %s an Exception occurred (but continuing): %",
                    repr(self.__contains__), repr(e)
                )
                return False

    def __getitem__(self, key):
        try:
            self.logger.debug("Getting data for '%s'", key)
            return self.deserialize(self.get(self.prepare_key(key)))
        except Exception as e:
            self.logger.exception(
                "While %s an Exception occurred (but continuing): %s. Note that this will yield undefined behavior.",
                repr(self.__getitem__), repr(e)
            )
            # this is technically wrong ...
            return None

    def __setitem__(self, key, value):
        if self.ignore_cache is True or key in self.ignore_cache:
            return
        else:
            try:
                self.logger.debug("Setting data for '%s'", key)
                self.set(self.prepare_key(key), self.serialize(value))
            except Exception as e:
                self.logger.exception(
                    "While %s an Exception occurred (but continuing): %s",
                    repr(self.__setitem__), repr(e)
                )


class FileCache(BaseCache):
    """
    A caching class which stores the data in flat files.
    """
    def build_cache_filename(self, suffix):
        """

        :param suffix:
        :return:
        """
        return "%s.%s.cache" % (self.cache_token, suffix,)

    def contains(self, key):
        """

        :param key:
        :return:
        """
        return os.path.isfile(self.build_cache_filename(key))

    def get(self, key):
        """

        :param key:
        :return:
        """
        with open(self.build_cache_filename(key), 'rb') as fp:
            return fp.read(os.path.getsize(self.build_cache_filename(key)))

    def set(self, key, value):
        """

        :param key:
        :param value:
        """
        with open(self.build_cache_filename(key), 'wb+') as fp:
            fp.write(value)


Cache = FileCache


class Sqlite3Cache(BaseCache):
    """
    A caching class which stores the data in a sqlite3 database.
    """
    def contains(self, key):
        """

        :param key:
        :return:
        """
        result = self.conn.execute('SELECT COUNT(*) FROM entries WHERE name = ?', (key,))
        for row in result:
            return row[0] == 1
        return False

    def get(self, key):
        """

        :param key:
        :return:
        """
        result = self.conn.execute('SELECT value FROM entries WHERE name = ?', (key,))
        for row in result:
            return row[0]

    def keys(self):
        """


        :return:
        """
        result = self.conn.execute('SELECT name FROM entries')
        return [row[0] for row in result]

    def set(self, key, value):
        """

        :param key:
        :param value:
        """
        self.conn.execute('DELETE FROM entries WHERE name = ?', (key,))

        self.conn.execute(
            'INSERT INTO entries (name, value) VALUES (?, ?)',
            (key, sqlite3.Binary(value),)
        )

        self.conn.commit()

    def __init__(self, *args, **kwargs):
        super(Sqlite3Cache, self).__init__(*args, **kwargs)

        self.conn = None

        if self.ignore_cache is not True:
            self.conn = sqlite3.connect('%s.sq3.cache' % (self.cache_token, ))
            self.conn.isolation_level = None
            self.conn.execute('PRAGMA journal_mode = WAL')
            self.conn.execute('PRAGMA synchronous = NORMAL')
            self.conn.isolation_level = 'DEFERRED'
            self.conn.execute('CREATE TABLE IF NOT EXISTS entries (name TEXT, value BLOB)')
            self.conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS entries_name ON entries (name)')

    def __del__(self):
        if self.conn:
            self.conn.close()


class NotReallyATree(list):
    """
        The class is a some-what duck-type compatible (it has a ``query`` method) dumb replacement
        for (c)KDTrees. It can be used to find the nearest matching point to a query point.
        (And does that by exhaustive search...)
    """
    def __init__(self, iterable):
        """
        :param iterable: input data
        :type iterable: iterable
        :return: the queryable object
        :rtype: NotReallyAtree
        """
        super(NotReallyATree, self).__init__(self)
        for i in iterable:
            self.append(i)
        self.na = np.array(iterable)

    def query(self, q):  # w_numpy
        """
        Finds the point which is nearest to ``q``.
        Uses the Euclidean distance.

        :param q: query point
        :return: distance, index
        :rtype: float, int

        >>> t = NotReallyATree([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        >>> t.query([1.25, 1.25])
        (0.3535533905932738, 0)
        >>> t = NotReallyATree([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        >>> t.query([2.3535533905932737622, 2.3535533905932737622])
        (0.5000000000000002, 1)
        """
        distances = np.sqrt(np.sum(np.power(self.na - q, 2.0), 1))
        pos = np.argmin(distances, 0)
        return distances[pos], pos

