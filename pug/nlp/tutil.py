#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""table utils"""
from __future__ import division, print_function, absolute_import
from builtins import str, zip, int, list  # , unicode  # noqa
from future.utils import viewitems  # noqa
from past.builtins import basestring
# from builtins import (
#          bytes, dict, int, list, object, range, str,
#          ascii, chr, hex, input, next, oct, open,
#          pow, round, super,
#          filter, map, zip)
import re
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

import pytz

from .constant import DEFAULT_TZ
from .constant import MAX_DATETIME, MIN_DATETIME, MAX_TIMESTAMP, MIN_TIMESTAMP, NAT
import pug.nlp.regex as rex


# pytz.timezone offsets for str abbreviation.
# WARNING: many abbreviations are ambiguous!
# munged table from @NasBanov: http://stackoverflow.com/a/4766400/623735
TZ_OFFSET_ABBREV = [
    [-12, 'Y'],
    [-11, 'X', 'NUT', 'SST'],
    [-10, 'W', 'CKT', 'HAST', 'HST', 'TAHT', 'TKT'],
    [-9, 'V', 'AKST', 'GAMT', 'GIT', 'HADT', 'HNY'],
    [-8, 'U', 'AKDT', 'CIST', 'HAY', 'HNP', 'PST', 'PT'],
    [-7, 'T', 'HAP', 'HNR', 'MST', 'PDT'],
    [-6, 'S', 'CST', 'EAST', 'GALT', 'HAR', 'HNC', 'MDT'],
    [-5, 'R', 'CDT', 'COT', 'EASST', 'ECT', 'EST', 'ET', 'HAC', 'HNE', 'PET'],
    [-4, 'Q', 'AST', 'BOT', 'CLT', 'COST', 'EDT', 'FKT', 'GYT', 'HAE', 'HNA', 'PYT'],
    [-3, 'P', 'ADT', 'ART', 'BRT', 'CLST', 'FKST', 'GFT', 'HAA', 'PMST', 'PYST', 'SRT', 'UYT', 'WGT'],
    [-2, 'O', 'BRST', 'FNT', 'PMDT', 'UYST', 'WGST'],
    [-1, 'N', 'AZOT', 'CVT', 'EGT'],
    [0, 'Z', 'EGST', 'GMT', 'UTC', 'WET', 'WT'],
    [1, 'A', 'CET', 'DFT', 'WAT', 'WEDT', 'WEST'],
    [2, 'B', 'CAT', 'CEDT', 'CEST', 'EET', 'SAST', 'WAST'],
    [3, 'C', 'EAT', 'EEDT', 'EEST', 'IDT', 'MSK'],
    [4, 'D', 'AMT', 'AZT', 'GET', 'GST', 'KUYT', 'MSD', 'MUT', 'RET', 'SAMT', 'SCT'],
    [5, 'E', 'AMST', 'AQTT', 'AZST', 'HMT', 'MAWT', 'MVT', 'PKT', 'TFT', 'TJT', 'TMT', 'UZT', 'YEKT'],
    [6, 'F', 'ALMT', 'BIOT', 'BTT', 'IOT', 'KGT', 'NOVT', 'OMST', 'YEKST'],
    [7, 'G', 'CXT', 'DAVT', 'HOVT', 'ICT', 'KRAT', 'NOVST', 'OMSST', 'THA', 'WIB'],
    [8, 'H', 'ACT', 'AWST', 'BDT', 'BNT', 'CAST', 'HKT', 'IRKT', 'KRAST', 'MYT', 'PHT', 'SGT', 'ULAT', 'WITA', 'WST'],
    [9, 'I', 'AWDT', 'IRKST', 'JST', 'KST', 'PWT', 'TLT', 'WDT', 'WIT', 'YAKT'],
    [10, 'K', 'AEST', 'ChST', 'PGT', 'VLAT', 'YAKST', 'YAPT'],
    [11, 'L', 'AEDT', 'LHDT', 'MAGT', 'NCT', 'PONT', 'SBT', 'VLAST', 'VUT'],
    [12, 'M', 'ANAST', 'ANAT', 'FJT', 'GILT', 'MAGST', 'MHT', 'NZST', 'PETST', 'PETT', 'TVT', 'WFT'],
    [13, 'FJST', 'NZDT'],
    [11.5, 'NFT'],
    [10.5, 'ACDT', 'LHST'],
    [9.5, 'ACST'],
    [6.5, 'CCT', 'MMT'],
    [5.75, 'NPT'],
    [5.5, 'SLT'],
    [4.5, 'AFT', 'IRDT'],
    [3.5, 'IRST'],
    [-2.5, 'HAT', 'NDT'],
    [-3.5, 'HNT', 'NST', 'NT'],
    [-4.5, 'HLV', 'VET'],
    [-9.5, 'MART', 'MIT'],
]
TZ_ABBREV_OFFSET = {}
for row in TZ_OFFSET_ABBREV:
    for abbrev in row[1:]:
        TZ_ABBREV_OFFSET[abbrev.strip().upper()] = float(row[0])
# FIXME: autogenerate this from pytz.timezone(iso_tz_name).tzname(datetime.datetime())
#         or [pytz.timezone(tz)._tzinfos.keys() for tz in pytz.all_timezones if hasattr(pytz.timezone(tz), '_tzinfos')]
TZ_ABBREV_INFO = {
    'AKST': ('US/Alaska',  -10), 'AKDT': ('US/Alaska',   -9),  'AKT': ('US/Alaska',  -10),
    'HAST': ('US/Hawaii',   -9), 'HADT': ('US/Hawaii',   -8),  'HAT': ('US/Hawaii',   -9),
    'PST':  ('US/Pacific',  -8),  'PDT': ('US/Pacific',  -7),   'PT': ('US/Pacific',  -8),
    'MST':  ('US/Mountain', -7),  'MDT': ('US/Mountain', -6),   'MT': ('US/Mountain', -7),
    'CST':  ('US/Central',  -6),  'CDT': ('US/Central',  -5),   'CT': ('US/Central',  -6),
    'EST':  ('US/Eastern',  -5),  'EDT': ('US/Eastern',  -4),   'ET': ('US/Eastern',  -5),
    'AST':  ('US/Atlantic', -4),  'ADT': ('US/Atlantic', -3),   'AT': ('US/Atlantic', -4),
    'GMT':  ('UTC', 0),
}
TZ_ABBREV_OFFSET = dict(((abbrev, info[1]) for abbrev, info in viewitems(TZ_ABBREV_INFO)))
TZ_ABBREV_NAME = dict(((abbrev, info[0]) for abbrev, info in viewitems(TZ_ABBREV_INFO)))


np = pd.np


def parse_time(timestr):
    dt = parse_date(timestr)
    if dt.date() == datetime.datetime.today().date() and re.match('^\s*\d+\:\d+.*', timestr):
        return dt.time()
    raise ValueError('Unknown string format.')


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


def make_tz_aware(dt, tz='UTC', is_dst=None):
    """Add timezone information to a datetime object, only if it is naive.

    >>> make_tz_aware(datetime.datetime(2001, 9, 8, 7, 6))
    datetime.datetime(2001, 9, 8, 7, 6, tzinfo=<UTC>)
    >>> make_tz_aware(['2010-01-01'], 'PST')
    [datetime.datetime(2010, 1, 1, 0, 0, tzinfo=<DstTzInfo 'US/Pacific' PST-1 day, 16:00:00 STD>)]
    >>> make_tz_aware(['1970-10-31', '1970-12-25', '1971-07-04'], 'CDT')  # doctest: +NORMALIZE_WHITESPACE
    [datetime.datetime(1970, 10, 31, 0, 0, tzinfo=<DstTzInfo 'US/Central' CST-1 day, 18:00:00 STD>),
     datetime.datetime(1970, 12, 25, 0, 0, tzinfo=<DstTzInfo 'US/Central' CST-1 day, 18:00:00 STD>),
     datetime.datetime(1971,  7,  4, 0, 0, tzinfo=<DstTzInfo 'US/Central' CDT-1 day, 19:00:00 DST>)]
    >>> make_tz_aware([None, float('nan'), float('inf'), 1980, 1979.25*365.25, '1970-10-31', '1970-12-25', '1971-07-04'],
    ...               'CDT')  # doctest: +NORMALIZE_WHITESPACE
    [None, nan, inf,
     datetime.datetime(6, 6, 3, 0, 0, tzinfo=<DstTzInfo 'US/Central' LMT-1 day, 18:09:00 STD>),
     datetime.datetime(1980, 4, 16, 1, 30, tzinfo=<DstTzInfo 'US/Central' CST-1 day, 18:00:00 STD>),
     datetime.datetime(1970, 10, 31, 0, 0, tzinfo=<DstTzInfo 'US/Central' CST-1 day, 18:00:00 STD>),
     datetime.datetime(1970, 12, 25, 0, 0, tzinfo=<DstTzInfo 'US/Central' CST-1 day, 18:00:00 STD>),
     datetime.datetime(1971, 7, 4, 0, 0, tzinfo=<DstTzInfo 'US/Central' CDT-1 day, 19:00:00 DST>)]
    >>> make_tz_aware(datetime.time(22, 23, 59, 123456))
    datetime.time(22, 23, 59, 123456, tzinfo=<UTC>)
    >>> make_tz_aware(datetime.time(22, 23, 59, 123456), 'PDT', is_dst=True)
    datetime.time(22, 23, 59, 123456, tzinfo=<DstTzInfo 'US/Pacific' LMT-1 day, 16:07:00 STD>)

    """
    # make sure dt is a datetime, time, or list of datetime/times
    dt = make_datetime(dt)
    if not isinstance(dt, (list, datetime.datetime, datetime.date, datetime.time, pd.Timestamp)):
        return dt
    # TODO: deal with sequence of timezones
    try:
        tz = dt.tzinfo or tz
    except AttributeError:
        pass
    try:
        tzstr = str(tz).strip().upper()
        if tzstr in TZ_ABBREV_NAME:
            is_dst = is_dst or tzstr.endswith('DT')
            tz = TZ_ABBREV_NAME.get(tzstr, tz)
    except:
        pass
    try:
        tz = pytz.timezone(tz)
    except AttributeError:
        # from traceback import print_exc
        # print_exc()
        pass
    try:
        return tz.localize(dt, is_dst=is_dst)
    except:
        # from traceback import print_exc
        # print_exc()  # TypeError: unsupported operand type(s) for +: 'datetime.time' and 'datetime.timedelta'
        pass
    # could be datetime.time, which can't be localized. Insted `replace` the TZ
    # don't try/except in case dt is not a datetime or time type -- should raise an exception
    if not isinstance(dt, list):
        return dt.replace(tzinfo=tz)

    return [make_tz_aware(dt0, tz=tz, is_dst=is_dst) for dt0 in dt]


def normalize_datetime(t, time=datetime.timedelta(hours=16)):
    if isinstance(t, datetime.datetime):
        if not t.hours + t.seconds:
            if time:
                t += time
        return t
    if isinstance(t, datetime.date):
        return normalize_datetime(datetime.datetime(t), time=time)
    if isinstance(t, basestring):
        return normalize_datetime(parse_date(t))
    return normalize_datetime(datetime.datetime(*[int(i) for i in t]))


def normalize_date(d):
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, datetime.datetime):
        return datetime.date(d.year, d.month, d.day)
    if isinstance(d, basestring):
        return normalize_date(parse_date(d))
    return normalize_date(datetime.datetime(*[int(i) for i in d]))


def clean_wiki_datetime(dt, squelch=True):
    if isinstance(dt, datetime.datetime):
        return dt
    elif not isinstance(dt, basestring):
        dt = ' '.join(dt)
    try:
        return make_tz_aware(parse_date(dt))
    except:
        if not squelch:
            print("Failed to parse %r as a date" % dt)
    dt = [s.strip() for s in dt.split(' ')]
    # get rid of any " at " or empty strings
    dt = [s for s in dt if s and s.lower() != 'at']

    # deal with the absence of :'s in wikipedia datetime strings

    if rex.month_name.match(dt[0]) or rex.month_name.match(dt[1]):
        if len(dt) >= 5:
            dt = dt[:-2] + [dt[-2].strip(':') + ':' + dt[-1].strip(':')]
            return clean_wiki_datetime(' '.join(dt))
        elif len(dt) == 4 and (len(dt[3]) == 4 or len(dt[0]) == 4):
            dt[:-1] + ['00']
            return clean_wiki_datetime(' '.join(dt))
    elif rex.month_name.match(dt[-2]) or rex.month_name.match(dt[-3]):
        if len(dt) >= 5:
            dt = [dt[0].strip(':') + ':' + dt[1].strip(':')] + dt[2:]
            return clean_wiki_datetime(' '.join(dt))
        elif len(dt) == 4 and (len(dt[-1]) == 4 or len(dt[-3]) == 4):
            dt = [dt[0], '00'] + dt[1:]
            return clean_wiki_datetime(' '.join(dt))

    try:
        return make_tz_aware(parse_date(' '.join(dt)))
    except Exception as e:
        if squelch:
            from traceback import format_exc
            print(format_exc(e) + '\n^^^ Exception caught ^^^\nWARN: Failed to parse datetime string %r\n      from list of strings %r' %
                  (' '.join(dt), dt))
            return dt
        raise(e)


def clip_datetime(dt, tz=DEFAULT_TZ, is_dst=None):
    """Limit a datetime to a valid range for datetime, datetime64, and Timestamp objects
    >>> from datetime import timedelta
    >>> from clayton.constant import MAX_DATETIME64, MAX_DATETIME, MAX_TIMESTAMP
    >>> clip_datetime(MAX_DATETIME + timedelta(100)) == pd.Timestamp(MAX_DATETIME64, tz='utc') == MAX_TIMESTAMP
    True
    >>> MAX_TIMESTAMP
    Timestamp('2262-04-11 23:47:16.854775807+0000', tz='UTC')
    """
    if isinstance(dt, datetime.datetime):
        # TODO: this gives up a day of datetime range due to assumptions about timezone
        #       make MIN/MAX naive and replace dt.replace(tz=None) before comparison
        #       set it back when done
        dt = make_tz_aware(dt, tz=tz, is_dst=is_dst)
        try:
            return pd.tslib.Timestamp(dt)
        except:
            pass
        if dt > MAX_DATETIME:
            return MAX_TIMESTAMP
        elif dt < MIN_DATETIME:
            return MIN_TIMESTAMP
        return NAT
    return dt
