#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""table utils"""
from __future__ import division, print_function, absolute_import
from builtins import str  # , unicode  # noqa
# from future.utils import viewitems  # noqa
from past.builtins import basestring
# try:  # python 3.5+
#    from io import StringIO
#    from ConfigParser import ConfigParser
#    from itertools import izip as zip
# except:
#     from StringIO import StringIO
#     from configparser import ConfigParser

from traceback import print_exc
import datetime
import time

import pandas as pd
from dateutil.parser import parse as parse_date

np = pd.np


def make_date(dt, date_parser=parse_date):
    """Coerce a datetime or string into datetime.date object

    Arguments:
      dt (str or datetime.datetime or atetime.time or numpy.Timestamp): time or date
        to be coerced into a `datetime.date` object

    Returns:
      datetime.time: Time of day portion of a `datetime` string or object

    >>> make_date('')
    datetime.date(1970, 1, 1)
    >>> make_date(None)
    datetime.date(1970, 1, 1)
    >>> make_date("11:59 PM") == datetime.date.today()
    True
    >>> make_date(datetime.datetime(1999, 12, 31, 23, 59, 59))
    datetime.date(1999, 12, 31)
    """
    if not dt:
        return datetime.date(1970, 1, 1)
    if isinstance(dt, basestring):
        dt = date_parser(dt)
    try:
        dt = dt.timetuple()[:3]
    except:
        dt = tuple(dt)[:3]
    return datetime.date(*dt)


def make_datetime(dt, date_parser=parse_date):
    """Coerce a datetime or string into datetime.datetime object

    Arguments:
      dt (str or datetime.datetime or atetime.time or numpy.Timestamp): time or date
        to be coerced into a `datetime.date` object

    Returns:
      datetime.time: Time of day portion of a `datetime` string or object

    >>> make_date('')
    datetime.date(1970, 1, 1)
    >>> make_date(None)
    datetime.date(1970, 1, 1)
    >>> make_date("11:59 PM") == datetime.date.today()
    True
    >>> make_date(datetime.datetime(1999, 12, 31, 23, 59, 59))
    datetime.date(1999, 12, 31)
    >>> make_datetime(['1970-10-31', '1970-12-25'])  # doctest: +NORMALIZE_WHITESPACE
    [datetime.datetime(1970, 10, 31, 0, 0), datetime.datetime(1970, 12, 25, 0, 0)]
    """
    if (isinstance(dt, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp, np.datetime64)) or
            dt in (float('nan'), float('inf'), float('-inf'), None, '')):
        return dt
    if isinstance(dt, (float, int)):
        return datetime_from_ordinal_float(dt)
    if isinstance(dt, datetime.date):
        return datetime.datetime(dt.year, dt.month, dt.day)
    if isinstance(dt, datetime.time):
        return datetime.datetime(1, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond)
    if not dt:
        return datetime.datetime(1970, 1, 1)
    if isinstance(dt, basestring):
        try:
            return date_parser(dt)
        except:
            print('Unable to make_datetime({})'.format(dt))
            raise
    try:
        return datetime.datetime(*dt.timetuple()[:7])
    except:
        try:
            dt = list(dt)
            if 0 < len(dt) < 7:
                try:
                    return datetime.datetime(*dt[:7])
                except:
                    pass
        except:  # TypeError
            # dt is not iterable
            return dt

    return [make_datetime(val, date_parser=date_parser) for val in dt]


def make_time(dt, date_parser=parse_date):
    """Ignore date information in a datetime string or object

    Arguments:
      dt (str or datetime.datetime or atetime.time or numpy.Timestamp): time or date
        to be coerced into a `datetime.time` object

    Returns:
      datetime.time: Time of day portion of a `datetime` string or object

    >>> make_time(None)
    datetime.time(0, 0)
    >>> make_time("")
    datetime.time(0, 0)
    >>> make_time("11:59 PM")
    datetime.time(23, 59)
    >>> make_time(datetime.datetime(1999, 12, 31, 23, 59, 59))
    datetime.time(23, 59, 59)
    """
    if not dt:
        return datetime.time(0, 0)
    if isinstance(dt, basestring):
        try:
            dt = date_parser(dt)
        except:
            print_exc()
            print('Unable to parse {}'.format(repr(dt)))
    try:
        dt = dt.timetuple()[3:6]
    except:
        dt = tuple(dt)[3:6]
    return datetime.time(*dt)


def quantize_datetime(dt, resolution=None):
    """Quantize a datetime to integer years, months, days, hours, minutes, seconds or microseconds

    Also works with a `datetime.timetuple` or `time.struct_time` or a 1to9-tuple of ints or floats.
    Also works with a sequenece of struct_times, tuples, or datetimes

    >>> quantize_datetime(datetime.datetime(1970,1,2,3,4,5,6), resolution=3)
    datetime.datetime(1970, 1, 2, 0, 0)

    Notice that 6 is the highest resolution value with any utility
    >>> quantize_datetime(datetime.datetime(1970,1,2,3,4,5,6), resolution=7)
    datetime.datetime(1970, 1, 2, 3, 4, 5)
    >>> quantize_datetime(datetime.datetime(1971,2,3,4,5,6,7), 1)
    datetime.datetime(1971, 1, 1, 0, 0)
    """
    # FIXME: this automatically truncates off microseconds just because timtuple() only goes out to sec
    resolution = int(resolution or 6)
    if hasattr(dt, 'timetuple'):
        dt = dt.timetuple()  # strips timezone info

    if isinstance(dt, time.struct_time):
        # strip last 3 fields (tm_wday, tm_yday, tm_isdst)
        dt = list(dt)[:6]
        # struct_time has no microsecond, but accepts float seconds
        dt += [int((dt[5] - int(dt[5])) * 1000000)]
        dt[5] = int(dt[5])
        return datetime.datetime(*(dt[:resolution] + [1] * max(3 - resolution, 0)))

    if isinstance(dt, tuple) and len(dt) <= 9 and all(isinstance(val, (float, int)) for val in dt):
        dt = list(dt) + [0] * (max(6 - len(dt), 0))
        # if the 6th element of the tuple looks like a float set of seconds need to add microseconds
        if len(dt) == 6 and isinstance(dt[5], float):
            dt = list(dt) + [1000000 * (dt[5] - int(dt[5]))]
            dt[5] = int(dt[5])
        dt = tuple(int(val) for val in dt)
        return datetime.datetime(*(dt[:resolution] + [1] * max(resolution - 3, 0)))

    return [quantize_datetime(value) for value in dt]


def is_valid_american_date_string(s, require_year=True):
    if not isinstance(s, basestring):
        return False
    if require_year and len(s.split('/')) != 3:
        return False
    return bool(1 <= int(s.split('/')[0]) <= 12 and 1 <= int(s.split('/')[1]) <= 31)


def ordinal_float(dt):
    """Like datetime.ordinal, but rather than integer allows fractional days (so float not ordinal at all)

    Similar to the Microsoft Excel numerical representation of a datetime object

    >>> ordinal_float(datetime.datetime(1970, 1, 1))
    719163.0
    >>> ordinal_float(datetime.datetime(1, 2, 3, 4, 5, 6, 7))  # doctest: +ELLIPSIS
    34.1702083334143...
    """
    try:
        return dt.toordinal() + ((((dt.microsecond / 1000000.) + dt.second) / 60. + dt.minute) / 60 + dt.hour) / 24.
    except:
        try:
            return ordinal_float(make_datetime(dt))
        except:
            pass
    dt = list(make_datetime(val) for val in dt)
    assert(all(isinstance(val, datetime.datetime) for val in dt))
    return [ordinal_float(val) for val in dt]


def datetime_from_ordinal_float(days):
    """Inverse of `ordinal_float()`, converts a float number of days back to a `datetime` object

    >>> dt = datetime.datetime(1970, 1, 1)
    >>> datetime_from_ordinal_float(ordinal_float(dt)) == dt
    True
    >>> dt = datetime.datetime(1, 2, 3, 4, 5, 6, 7)
    >>> datetime_from_ordinal_float(ordinal_float(dt)) == dt
    True
    """
    if isinstance(days, (float, int)):
        if np.isnan(days) or days in set((float('nan'), float('inf'), float('-inf'))):
            return days
        dt = datetime.datetime.fromordinal(int(days))
        seconds = (days - int(days)) * 3600. * 24.
        microseconds = (seconds - int(seconds)) * 1000000
        return dt + datetime.timedelta(days=0, seconds=int(seconds), microseconds=int(round(microseconds)))
    return [datetime_from_ordinal_float(d) for d in days]


def timetag_str(dt=None, sep='-', filler='0', resolution=6):
    """Generate a date-time tag suitable for appending to a file name.

    >>> timetag_str(resolution=3) == '-'.join('{0:02d}'.format(i) for i in tuple(datetime.datetime.now().timetuple()[:3]))
    True
    >>> timetag_str(datetime.datetime(2004,12,8,1,2,3,400000))
    '2004-12-08-01-02-03'
    >>> timetag_str(datetime.datetime(2004,12,8))
    '2004-12-08-00-00-00'
    >>> timetag_str(datetime.datetime(2003,6,19), filler='')
    '2003-6-19-0-0-0'
    """
    resolution = int(resolution or 6)
    if sep in (None, False):
        sep = ''
    sep = str(sep)
    dt = datetime.datetime.now() if dt is None else dt
    # FIXME: don't use timetuple which truncates microseconds
    return sep.join(('{0:' + filler + ('2' if filler else '') + 'd}').format(i) for i in tuple(dt.timetuple()[:resolution]))
timestamp_str = make_timestamp = make_timetag = timetag_str


def days_since(dt, dt0=datetime.datetime(1970, 1, 1, 0, 0, 0)):
    return ordinal_float(dt) - ordinal_float(dt0)
