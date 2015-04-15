#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Utilities for Natural Language Processing (NLP):

* Vocabulary and dimension reduction
* Word statistics calculation
* Add a timezone to a datetime
* Slice a django queryset
* Genereate batches from a long list or sequence
* Inverse dict/hashtable lookup
* Generate a valid python variable or class name from a string
* Generate a slidedeck-compatible markdown from an text or markdown outline or list
* Convert a sequence of sequences to a dictionary of sequences
* Pierson correlation coefficient calculation
* Parse a string into sentences or tokens
* Table (list of list) manipulation
* make_time, make_date, quantize_datetime -- ignore portions of a datetime struct
* ordinal_float, datetime_from_ordinal_float -- conversion between datetimes and float days
* days_since    -- subract two date or datetime objects and return difference in days (float)

'''

import os
import stat
import collections
import itertools
import datetime
import time
import pytz
import dateutil
import types
import re
import string
import csv
import warnings
from collections import OrderedDict
from traceback import print_exc
from decimal import Decimal, InvalidOperation, InvalidContext
import math
from types import NoneType
from StringIO import StringIO
import copy
import codecs

import pandas as pd
np = pd.np
from dateutil.parser import parse as parse_date
import progressbar
from pytz import timezone
from fuzzywuzzy import process as fuzzy
from slugify import slugify

import charlist
import regex_patterns as RE

import xlrd

import logging
logger = logging.getLogger('pug.nlp.util')


def parse_time(timestr):
    dt = parse_date(timestr)
    if dt.date() == datetime.datetime.today().date() and re.match('^\s*\d+\:\d+.*', timestr):
        return dt.time()
    raise ValueError('Unknown string format.')


ROUNDABLE_NUMERIC_TYPES = (float, long, int, Decimal, bool)
FLOATABLE_NUMERIC_TYPES = (float, long, int, Decimal, bool)
BASIC_NUMERIC_TYPES = (float, long, int) 
NUMERIC_TYPES = (float, long, int, Decimal, complex, str)  # datetime.datetime, datetime.date
NUMBERS_AND_DATETIMES = (float, long, int, Decimal, complex, parse_time, parse_date, str)
SCALAR_TYPES = (float, long, int, Decimal, bool, complex, basestring, str, unicode)  # datetime.datetime, datetime.date
# numpy types are derived from these so no need to include numpy.float64, numpy.int64 etc
DICTABLE_TYPES = (collections.Mapping, tuple, list)  # convertable to a dictionary (inherits collections.Mapping or is a list of key/value pairs)
VECTOR_TYPES = (list, tuple)
PUNC = unicode(string.punctuation)


# 4 types of "histograms" and their canonical name/label
HIST_NAME = {
                'hist': 'hist', 'ff':  'hist',  'fd': 'hist', 'dff':  'hist', 'dfd': 'hist', 'gfd': 'hist', 'gff': 'hist', 'bfd': 'hist', 'bff': 'hist',
                'pmf':  'pmf',  'pdf': 'pmf',   'pd': 'pmf',
                'cmf':  'cmf',  'cdf': 'cmf',
                'cfd':  'cfd',  'cff': 'cfd',   'cdf': 'cfd',
            }
HIST_CONFIG = {
    'hist': { 
        'name': 'Histogram',  # frequency distribution, frequency function, discrete ff/fd, grouped ff/fd, binned ff/fd
        'kwargs': { 'normalize': False, 'cumulative': False, },
        'index': 0,
        'xlabel': 'Bin',
        'ylabel': 'Count',
        },
    'pmf': {
        # PMFs have discrete, exact values as bins rather than ranges (finite bin widths)
        #   but this histogram configuration doesn't distinguish between PMFs and PDFs, 
        #   since mathematically they have all the same properties. 
        #    PDFs just have a range associated with each discrete value (which should be when integrating a PDF but not when summing a PMF where the "width" is uniformly 1)
        'name': 'Probability Mass Function',   # probability density function, probability distribution [function]
        'kwargs': { 'normalize': True, 'cumulative': False, },
        'index': 1,
        'xlabel': 'Bin',
        'ylabel': 'Probability',
        },
    'cmf': {
        'name': 'Cumulative Probability',
        'kwargs': { 'normalize': True, 'cumulative': True, },
        'index': 2,
        'xlabel': 'Bin',
        'ylabel': 'Cumulative Probability',
        },
    'cfd': {
        'name': 'Cumulative Frequency Distribution',
        'kwargs': { 'normalize': False, 'cumulative': True, },
        'index': 3,
        'xlabel': 'Bin',
        'ylabel': 'Cumulative Count',
        },
    }

# MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
# MONTH_PREFIXES = [m[:3] for m in MONTHS]
# MONTH_SUFFIXES = [m[3:] for m in MONTHS]
# SUFFIX_LETTERS = ''.join(set(''.join(MONTH_SUFFIXES)))

# TZ constants
try:
    from django.conf import settings
    DEFAULT_TZ = timezone(settings.TIME_ZONE)
except:
    DEFAULT_TZ = timezone('UTC')


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
[-9.5, 'MART', 'MIT']]
TZ_ABBREV_OFFSET = {}
for row in TZ_OFFSET_ABBREV:
    for abbrev in row[1:]:
        TZ_ABBREV_OFFSET[abbrev.strip().upper()] = float(row[0])
# FIXME: autogenerate this from pytz.timezone(iso_tz_name).tzname(datetime.datetime()) 
#         or [pytz.timezone(tz)._tzinfos.keys() for tz in pytz.all_timezones if hasattr(pytz.timezone(tz), '_tzinfos')]
TZ_ABBREV_INFO = {
    'AKST': ('US/Alaska',  -10), 'AKDT': ('US/Alaska',   -9),  'AKT': ('US/Alaska' , -10),
    'HAST': ('US/Hawaii',   -9), 'HADT': ('US/Hawaii',   -8),  'HAT': ('US/Hawaii',   -9),
     'PST': ('US/Pacific',  -8),  'PDT': ('US/Pacific',  -7),   'PT': ('US/Pacific',  -8),
     'MST': ('US/Mountain', -7),  'MDT': ('US/Mountain', -6),   'MT': ('US/Mountain', -7),
     'CST': ('US/Central',  -6),  'CDT': ('US/Central',  -5),   'CT': ('US/Central',  -6),
     'EST': ('US/Eastern',  -5),  'EDT': ('US/Eastern',  -4),   'ET': ('US/Eastern',  -5),
     'AST': ('US/Atlantic', -4),  'ADT': ('US/Atlantic', -3),   'AT': ('US/Atlantic', -4),
     'GMT': ('UTC', 0),
    }
TZ_ABBREV_OFFSET = dict(((abbrev, info[1]) for abbrev, info in TZ_ABBREV_INFO.iteritems()))
TZ_ABBREV_NAME = dict(((abbrev, info[0]) for abbrev, info in TZ_ABBREV_INFO.iteritems()))


def qs_to_table(qs, excluded_fields=['id']):
    rows, rowl = [], []
    qs = qs.all()
    fields = sorted(qs[0]._meta.get_all_field_names())
    for row in qs:
        for f in fields:
            if f in excluded_fields:
                continue
            rowl += [getattr(row,f)]
        rows, rowl = rows + [rowl], []
    return rows


def force_hashable(obj, recursive=True):
    """Force frozenset() command to freeze the order and contents of mutables and iterables like lists, dicts, generators

    Useful for memoization and constructing dicts or hashtables where keys must be immutable.

    FIXME: Rename function because "hashable" is misleading. 
           A better name might be `force_immutable`.
           because some hashable objects (generators) are tuplized  by this function
           `tuplized` is probably a better name, but strings are left alone, so not quite right

    >>> force_hashable([1,2.,['3','four'],'five', {'s': 'ix'}])
    (1, 2.0, ('3', 'four'), 'five', (('s', 'ix'),))
    >>> force_hashable(i for i in range(4))
    (0, 1, 2, 3)
    >>> from collections import Counter
    >>> force_hashable(Counter('abbccc')) ==  (('a', 1), ('c', 3), ('b', 2))
    True
    """
    # if it's already hashable, and isn't a generator (which are also hashable, but not mutable)
    if hasattr(obj, '__hash__') and not hasattr(obj, 'next'):
        try:
            hash(obj)
            return obj
        except:
            pass
    if hasattr(obj, '__iter__'):
        # looks like a Mapping if it has .get() and .items(), so should treat it like one
        if hasattr(obj, 'get') and hasattr(obj, 'items'):
            # FIXME: prevent infinite recursion:
            #        tuples don't have 'items' method so this will recurse forever if elements within new tuple aren't hashable and recurse has not been set!
            return force_hashable(tuple(obj.items()))
        if recursive:
            return tuple(force_hashable(item) for item in obj)
        return tuple(obj)
    # strings are hashable so this ends the recursion for any object without an __iter__ method (strings do not)
    return unicode(obj)


def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in dict(d).iteritems())


def inverted_dict_of_lists(d):
    """Return a dict where the keys are all the values listed in the values of the original dict

    >>> inverted_dict_of_lists({0: ['a', 'b'], 1: 'cd'}) == {'a': 0, 'b': 0, 'cd': 1}
    True
    """
    new_dict = {}
    for (old_key, old_value_list) in dict(d).iteritems():
        for new_key in listify(old_value_list):
            new_dict[new_key] = old_key
    return new_dict


def sort_strings(strings, sort_order=None, reverse=False, case_sensitive=False, sort_order_first=True):
    """Sort a list of strings according to the provided sorted list of string prefixes

    TODO:
        - Provide an option to use `.startswith()` rather than a fixed prefix length (will be much slower)

    Arguments:
        sort_order_first (bool): Whether strings in sort_order should always preceed "unknown" strings
        sort_order (sequence of str): Desired ordering as a list of prefixes to the strings
            If sort_order strings have varying length, the max length will determine the prefix length compared
        reverse (bool): whether to reverse the sort orded. Passed through to `sorted(strings, reverse=reverse)`
        case_senstive (bool): Whether to sort in lexographic rather than alphabetic order
         and whether the prefixes  in sort_order are checked in a case-sensitive way 

    Examples:
        >>> sort_strings(['morn32', 'morning', 'unknown', 'date', 'dow', 'doy', 'moy'], 
        ...              ('dat', 'dow', 'moy', 'dom', 'doy', 'mor'))  # doctest: +NORMALIZE_WHITESPACE
        ['date', 'dow', 'moy', 'doy', 'morn32', 'morning', 'unknown']
        >>> sort_strings(['morn32', 'morning', 'unknown', 'less unknown', 'lucy', 'date', 'dow', 'doy', 'moy'], 
        ...              ('dat', 'dow', 'moy', 'dom', 'doy', 'mor'), reverse=True)  # doctest: +NORMALIZE_WHITESPACE
        ['unknown', 'lucy', 'less unknown', 'morning', 'morn32', 'doy', 'moy', 'dow', 'date']

        Strings whose prefixes don't exist in `sort_order` sequence can be interleaved into the
        sorted list in lexical order by setting `sort_order_first=False`
        >>> sort_strings(['morn32', 'morning', 'unknown', 'lucy', 'less unknown', 'date', 'dow', 'doy', 'moy'],
        ...              ('dat', 'dow', 'moy', 'dom', 'moy', 'mor'),
        ...              sort_order_first=False)  # doctest: +NORMALIZE_WHITESPACE
        ['date', 'dow', 'doy', 'less unknown', 'lucy', 'moy', 'morn32', 'morning', 'unknown']
    """
    if not case_sensitive:
        sort_order = tuple(s.lower() for s in sort_order)
        strings = tuple(s.lower() for s in strings)
    prefix_len = max(len(s) for s in sort_order)

    def compare(a, b, prefix_len=prefix_len):
        if prefix_len:
            if a[:prefix_len] in sort_order:
                if b[:prefix_len] in sort_order:
                    comparison = sort_order.index(a[:prefix_len]) - sort_order.index(b[:prefix_len])
                    comparison /= abs(comparison or 1)
                    if comparison:
                        return comparison * (-2 * reverse + 1)
                elif sort_order_first:
                    return -1 * (-2 * reverse + 1)
            # b may be in sort_order list, so it should be first
            elif sort_order_first and b[:prefix_len] in sort_order:
                return -2 * reverse + 1
        return (-1 * (a < b) + 1 * (a > b)) * (-2 * reverse + 1)

    return sorted(strings, cmp=compare)


def clean_field_dict(field_dict, cleaner=unicode.strip, time_zone=None):
    r"""Normalize field values by stripping whitespace from strings, localizing datetimes to a timezone, etc

    >>> sorted(clean_field_dict({'_state': object(), 'x': 1, 'y': u"\t  Wash Me! \n" }).items())
    [('x', 1), ('y', u'Wash Me!')]
    """
    d = {}
    if time_zone is None:
        tz = DEFAULT_TZ
    for k, v in field_dict.iteritems():
        if k == '_state':
            continue
        if isinstance(v, basestring):
            d[k] = cleaner(unicode(v))
        elif isinstance(v, (datetime.datetime, datetime.date)):
            d[k] = tz.localize(v)
        else:
            d[k] = v
    return d


# def reduce_vocab(tokens, similarity=.85, limit=20):
#     """Find spelling variations of similar words within a list of tokens to reduce token set size

#     Arguments:
#       tokens (list or set or tuple of str): token strings from which to eliminate similar spellings

#     Examples:
#       >>> reduce_vocab(('on', 'hon', 'honey', 'ones', 'one', 'two', 'three'))  # doctest: +NORMALIZE_WHITESPACE


#     """
#     tokens = set(tokens)
#     thesaurus = {}
#     while tokens:
#         tok = tokens.pop()
#         matches = fuzzy.extractBests(tok, tokens, score_cutoff=int(similarity * 100), limit=20)
#         if matches:
#             thesaurus[tok] = zip(*matches)[0]
#         else:
#             thesaurus[tok] = (tok,)
#         for syn in thesaurus[tok][1:]:
#             tokens.discard(syn)
#     return thesaurus


def reduce_vocab(tokens, similarity=.85, limit=20, sort_order=-1):
    """Find spelling variations of similar words within a list of tokens to reduce token set size

    Lexically sorted in reverse order (unless `reverse=False`), before running through fuzzy-wuzzy
    which results in the longer of identical spellings to be prefered (e.g. "ones" prefered to "one")
    as the key token. Usually you wantThis is usually what you want.

    Arguments:
      tokens (list or set or tuple of str): token strings from which to eliminate similar spellings
      similarity (float): portion of characters that should be unchanged in order to be considered a synonym
        as a fraction of the key token length.
        e.g. `0.85` (which means 85%) allows "hon" to match "on" and "honey", but not "one"

    Returns:
      dict: { 'token': ('similar_token', 'similar_token2', ...), ...}

    Examples:
      >>> tokens = ('on', 'hon', 'honey', 'ones', 'one', 'two', 'three')
      >>> answer = {'hon': ('on', 'honey'),
      ...           'one': ('ones',),
      ...           'three': (),
      ...           'two': ()}
      >>> reduce_vocab(tokens, sort_order=1) == answer
      True
      >>> answer = {'honey': ('hon',),
      ...           'ones': ('on', 'one'),
      ...           'three': (),
      ...           'two': ()}
      >>> reduce_vocab(tokens, sort_order=-1) == answer
      True
      >>> reduce_vocab(tokens, similarity=0.3, limit=2, sort_order=-1) ==  {'ones': ('one',), 'three': ('honey',), 'two': ('on', 'hon')}
      True
      >>> reduce_vocab(tokens, similarity=0.3, limit=3, sort_order=-1) ==  {'ones': (), 'three': ('honey',), 'two': ('on', 'hon', 'one')}
      True

    """
    if 0 <= similarity <= 1:
        similarity *= 100
    if sort_order:
        tokens = set(tokens)
        tokens_sorted = sorted(list(tokens), reverse=bool(sort_order < 0))
    else:
        tokens_sorted = list(tokens)
        tokens = set(tokens)
    # print(tokens)
    thesaurus = {}
    for tok in tokens_sorted:
        try:
            tokens.remove(tok)
        except (KeyError, ValueError):
            continue
        # FIXME: this is slow because the tokens list must be regenerated and reinstantiated with each iteration
        matches = fuzzy.extractBests(tok, list(tokens), score_cutoff=int(similarity), limit=limit)
        if matches:
            thesaurus[tok] = zip(*matches)[0]
        else:
            thesaurus[tok] = ()
        for syn in thesaurus[tok]:
            tokens.discard(syn)
    return thesaurus


def reduce_vocab_by_len(tokens, similarity=.87, limit=20, reverse=True):
    """Find spelling variations of similar words within a list of tokens to reduce token set size

    Sorted by length (longest first unless reverse=False) before running through fuzzy-wuzzy
    which results in longer key tokens.

    Arguments:
      tokens (list or set or tuple of str): token strings from which to eliminate similar spellings

    Returns:
      dict: { 'token': ('similar_token', 'similar_token2', ...), ...}

    Examples:
      >>> tokens = ('on', 'hon', 'honey', 'ones', 'one', 'two', 'three')
      >>> reduce_vocab_by_len(tokens) ==  {'honey': ('on', 'hon', 'one'), 'ones': (), 'three': (), 'two': ()}
      True
    """
    tokens = set(tokens)
    tokens_sorted = zip(*sorted([(len(tok), tok) for tok in tokens], reverse=reverse))[1]
    return reduce_vocab(tokens=tokens_sorted, similarity=similarity, limit=limit, sort_order=0)


def quantify_field_dict(field_dict, precision=None, date_precision=None, cleaner=unicode.strip):
    r"""Convert strings and datetime objects in the values of a dict into float/int/long, if possible

    Arguments:
      field_dict (dict): The dict to have any values (not keys) that are strings "quantified"
      precision (int): Number of digits of precision to enforce
      cleaner: A string cleaner to apply to all string before


    FIXME: define a time zone for the datetime object
    >>> sorted(quantify_field_dict({'_state': object(), 'x': 12345678911131517L, 'y': "\t  Wash Me! \n", 'z': datetime.datetime(1970, 10, 23, 23, 59, 59, 123456)}).iteritems())  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    [('x', 12345678911131517L), ('y', u'Wash Me!'), ('z', 25...99.123456)]
    """
    if cleaner:
        d = clean_field_dict(field_dict, cleaner=cleaner)
    for k, v in d.iteritems():
        if isinstance(d[k], datetime.datetime):
            # seconds since epoch = datetime.datetime(1969,12,31,18,0,0)
            try:
                # around the year 2250, a float conversion of this string will lose 1 microsecond of precision, and around 22500 the loss of precision will be 10 microseconds
                d[k] = float(d[k].strftime('%s.%f'))  # seconds since Jan 1, 1970
                if date_precision is not None and isinstance(d[k], ROUNDABLE_NUMERIC_TYPES):
                    d[k] = round(d[k], date_precision)
                    continue
            except:
                pass
        if not isinstance(d[k], (int, float, long)):
            try:
                d[k] = float(d[k])
            except:
                pass
        if precision is not None and isinstance(d[k], ROUNDABLE_NUMERIC_TYPES):
            d[k] = round(d[k], precision)
        if isinstance(d[k], float) and d[k].is_integer():
            # `int()` will convert to a long, if value overflows an integer type
            # use the original value, `v`, in case it was a long and d[k] is has been truncated by the conversion to float!
            d[k] = int(v)
    return d


def generate_batches(sequence, batch_len=1, allow_partial=True, ignore_errors=True, verbosity=1):
    """Iterate through a sequence (or generator) in batches of length `batch_len`

    http://stackoverflow.com/a/761125/623735
    >>> [batch for batch in generate_batches(range(7), 3)]
    [[0, 1, 2], [3, 4, 5], [6]]
    """
    it = iter(sequence)
    last_value = False
    # An exception will be thrown by `.next()` here and caught in the loop that called this iterator/generator 
    while not last_value:
        batch = []
        for n in xrange(batch_len):
            try:
                batch += (it.next(),)
            except StopIteration:
                last_value = True
                if batch:
                    break
                else:
                    raise StopIteration
            except Exception:
                # 'Error: new-line character seen in unquoted field - do you need to open the file in universal-newline mode?'       
                if verbosity > 0:
                    print_exc()
                if not ignore_errors:
                    raise
        yield batch


def generate_tuple_batches(qs, batch_len=1):
    """Iterate through a queryset in batches of length `batch_len`

    >>> [batch for batch in generate_tuple_batches(range(7), 3)]
    [(0, 1, 2), (3, 4, 5), (6,)]
    """
    num_items, batch = 0, []
    for item in qs:
        if num_items >= batch_len:
            yield tuple(batch)
            num_items = 0
            batch = []
        num_items += 1
        batch += [item]
    if num_items:
        yield tuple(batch)


def sliding_window(seq, n=2):
    """Generate overlapping sliding/rolling windows (of width n) over an iterable
    
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   

    References:
      http://stackoverflow.com/a/6822773/623735

    Examples:

    >>> list(sliding_window(range(6), 3))  # doctest: +NORMALIZE_WHITESPACE
    [(0, 1, 2),
     (1, 2, 3),
     (2, 3, 4),
     (3, 4, 5)]
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def generate_slices(sliceable_set, batch_len=1, length=None, start_batch=0):
    """Iterate through a sequence (or generator) in batches of length `batch_len`

    See Also:
      pug.dj.db.generate_queryset_batches

    References:
      http://stackoverflow.com/a/761125/623735

    Examples:
      >>  [batch for batch in generate_slices(range(7), 3)]
      [(0, 1, 2), (3, 4, 5), (6,)]
      >>  from django.contrib.auth.models import User, Permission
      >>  len(list(generate_slices(User.objects.all(), 2)))       == max(math.ceil(User.objects.count() / 2.), 1)
      True
      >>  len(list(generate_slices(Permission.objects.all(), 2))) == max(math.ceil(Permission.objects.count() / 2.), 1)
      True
    """
    if length is None:
        try:
            length = sliceable_set.count()
        except:
            length = len(sliceable_set)
    length = int(length)

    for i in range(length / batch_len + 1):
        if i < start_batch:
            continue
        start = i * batch_len
        end = min((i + 1) * batch_len, length)
        if start != end:
            yield tuple(sliceable_set[start:end])
    raise StopIteration


COUNT_NAMES = ['count', 'cnt', 'number', 'num', '#', 'frequency', 'probability', 'prob', 'occurences']
def find_count_label(d):
    """Find the member of a set that means "count" or "frequency" or "probability" or "number of occurrences".

    """
    for name in COUNT_NAMES:
        if name in d:
            return name
    for name in COUNT_NAMES:
        if str(name).lower() in d:
            return name


def first_in_seq(seq):
    # lists/sequences
    return next(iter(seq))


def get_key_for_value(dict_obj, value, default=None):
    """
    >>> get_key_for_value({0: 'what', 'k': 'ever', 'you': 'want', 'to find': None}, 'you')
    >>> get_key_for_value({0: 'what', 'k': 'ever', 'you': 'want', 'to find': None}, 'you', default='Not Found')
    'Not Found'
    >>> get_key_for_value({0: 'what', 'k': 'ever', 'you': 'want', 'to find': None}, 'other', default='Not Found')
    'Not Found'
    >>> get_key_for_value({0: 'what', 'k': 'ever', 'you': 'want', 'to find': None}, 'want')
    'you'
    >>> get_key_for_value({0: 'what', '': 'ever', 'you': 'want', 'to find': None, 'you': 'too'}, 'what')
    0
    >>> get_key_for_value({0: 'what', '': 'ever', 'you': 'want', 'to find': None, 'you': 'too', ' ': 'want'}, 'want')
    ' '
    """
    for k, v in dict_obj.iteritems():
        if v == value:
            return k
    return default


def list_set(seq):
    """Similar to `list(set(seq))`, but maintains the order of `seq` while eliminating duplicates

    In general list(set(L)) will not have the same order as the original list.
    This is a list(set(L)) function that will preserve the order of L.

    Arguments:
      seq (iterable): list, tuple, or other iterable to be used to produce an ordered `set()`

    Returns:
      iterable: A copy of `seq` but with duplicates removed, and distinct elements in the same order as in `seq`

    Examples:
      >>> list_set([2.7,3,2,2,2,1,1,2,3,4,3,2,42,1])
      [2.7, 3, 2, 1, 4, 42]
      >>> list_set(['Zzz','abc', ('what.', 'ever.'), 0, 0.0, 'Zzz', 0.00, 'ABC'])
      ['Zzz', 'abc', ('what.', 'ever.'), 0, 'ABC']
    """
    new_list = []
    for i in seq:
        if i not in new_list:
            new_list += [i]
    return type(seq)(new_list)


def fuzzy_get(possible_keys, approximate_key, default=None, similarity=0.6, tuple_joiner='|', key_and_value=False, dict_keys=None, ):
    r"""Find the closest matching key in a dictionary (or element in a list)

    For a dict, optionally retrieve the associated value associated with the closest key

    Notes:
      `possible_keys` must have all string elements or keys!
      Argument order is in reverse order relative to `fuzzywuzzy.process.extractOne()`
        but in the same order as get(self, key) method on dicts

    Arguments:
      possible_keys (dict): object to run the get method on using the key that is most similar to one within the dict
      approximate_key (str): key to look for a fuzzy match within the dict keys
      default (obj): the value to return if a similar key cannote be found in the `possible_keys`
      similarity (str): fractional similiarity between the approximate_key and the dict key (0.9 means 90% of characters must be identical)
      tuple_joiner (str): Character to use as delimitter/joiner between tuple elements.
        Used to create keys of any tuples to be able to use fuzzywuzzy string matching on it.
      key_and_value (bool): Whether to return both the key and its value (True) or just the value (False).
        Default is the same behavior as dict.get (i.e. key_and_value=False)
      dict_keys (list of str): if you already have a set of keys to search, this will save this funciton a little time and RAM

    See Also:
      get_similar: Allows nonstring keys and searches object attributes in addition to keys

    Examples:
      >>> fuzzy_get({'seller': 2.7, 'sailor': set('e')}, 'sail')
      set(['e'])
      >>> fuzzy_get({'seller': 2.7, 'sailor': set('e'), 'camera': object()}, 'SLR')
      2.7
      >>> fuzzy_get({'seller': 2.7, 'sailor': set('e'), 'camera': object()}, 'I')
      set(['e'])
      >>> fuzzy_get({'word': tuple('word'), 'noun': tuple('noun')}, 'woh!', similarity=.3, key_and_value=True)
      ('word', ('w', 'o', 'r', 'd'))
      >>> fuzzy_get({'word': tuple('word'), 'noun': tuple('noun')}, 'woh!', similarity=.9, key_and_value=True)
      (None, None)
      >>> fuzzy_get({'word': tuple('word'), 'noun': tuple('noun')}, 'woh!', similarity=.9, default='darn :-()', key_and_value=True)
      (None, 'darn :-()')
      >>> possible_keys = 'alerts astronomy conditions currenthurricane forecast forecast10day geolookup history hourly hourly10day planner rawtide satellite tide webcams yesterday'.split(' ')
      >>> fuzzy_get(possible_keys, "cond")
      'conditions'
      >>> fuzzy_get(possible_keys, "Tron")
      'astronomy'
      >>> df = pd.DataFrame(np.arange(6*2).reshape(2,6), columns=('alpha','beta','omega','begin','life','end'))
      >>> fuzzy_get(df, 'beg')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
      0    3
      1    9
      Name: begin, dtype: int...
      >>> fuzzy_get(df, 'get')
      >>> fuzzy_get(df, 'et')[1]
      7
      >>> fuzzy_get(df, 'get')
    """
    dict_obj = copy.copy(possible_keys)
    if not isinstance(dict_obj, (collections.Mapping, pd.DataFrame, pd.Series)):
        dict_obj = dict((x, x) for x in dict_obj)

    fuzzy_key, value = None, default
    if approximate_key in dict_obj:
        fuzzy_key, value = approximate_key, dict_obj[approximate_key]
    else:
        strkey = unicode(approximate_key)
        if approximate_key and strkey and strkey.strip():
            # print 'no exact match was found for {0} in {1} so preprocessing keys'.format(approximate_key, dict_obj.keys())
            if any(isinstance(k, (tuple, list)) for k in dict_obj):
                dict_obj = dict((tuple_joiner.join(str(k2) for k2 in k), v) for (k, v) in dict_obj.iteritems())
                if isinstance(approximate_key, (tuple, list)):
                    strkey = tuple_joiner.join(approximate_key)
            # WARN: fuzzywuzzy requires that the second argument be a list (sets and tuples fail!)
            dict_keys = list(set(dict_keys if dict_keys else dict_obj))
            if strkey in dict_keys:
                fuzzy_key, value = strkey, dict_obj[strkey]
            else:
                strkey = strkey.strip()
                if strkey in dict_keys:
                    fuzzy_key, value = strkey, dict_obj[strkey]
                else:
                    #print 'no exact match was found for {0} in {1} so checking with similarity cutoff of {2}'.format(strkey, dict_keys, similarity)
                    # WARN: extractBests will return [] if dict_keys is anything other than a list (even sets and tuples fail!)
                    fuzzy_key_scores = fuzzy.extractBests(strkey, dict_keys, score_cutoff=min(max(similarity*100.0 - 1, 0), 100), limit=6)
                    #print strkey, fuzzy_key_scores
                    if fuzzy_key_scores:
                        # print fuzzy_key_scores
                        fuzzy_score_keys = []
                        # add length similarity as part of score
                        for (i, (k, score)) in enumerate(fuzzy_key_scores):
                            fuzzy_score_keys += [(score * math.sqrt(len(strkey)**2 / float((len(k)**2 + len(strkey)**2) or 1)), k)]
                        # print fuzzy_score_keys
                        fuzzy_score, fuzzy_key = sorted(fuzzy_score_keys)[-1]
                        value = dict_obj[fuzzy_key]
    if key_and_value:
        if key_and_value in ('v', 'V', 'value', 'VALUE', 'Value'):
            return value
        return fuzzy_key, value
    else:
        return value


def fuzzy_get_value(obj, approximate_key, default=None, **kwargs):
    """ Like fuzzy_get, but assume the obj is dict-like and return the value without the key

    Notes:
      Argument order is in reverse order relative to `fuzzywuzzy.process.extractOne()` 
        but in the same order as get(self, key) method on dicts

    Arguments:
      obj (dict-like): object to run the get method on using the key that is most similar to one within the dict
      approximate_key (str): key to look for a fuzzy match within the dict keys
      default (obj): the value to return if a similar key cannote be found in the `possible_keys`
      similarity (str): fractional similiarity between the approximate_key and the dict key (0.9 means 90% of characters must be identical)
      tuple_joiner (str): Character to use as delimitter/joiner between tuple elements.
        Used to create keys of any tuples to be able to use fuzzywuzzy string matching on it.
      key_and_value (bool): Whether to return both the key and its value (True) or just the value (False). 
        Default is the same behavior as dict.get (i.e. key_and_value=False)
      dict_keys (list of str): if you already have a set of keys to search, this will save this funciton a little time and RAM

    Examples:
      >>> fuzzy_get_value({'seller': 2.7, 'sailor': set('e')}, 'sail') == set(['e'])
      True
      >>> fuzzy_get_value({'seller': 2.7, 'sailor': set('e'), 'camera': object()}, 'SLR')
      2.7
      >>> fuzzy_get_value({'seller': 2.7, 'sailor': set('e'), 'camera': object()}, 'I') == set(['e'])
      True
      >>> fuzzy_get_value({'word': tuple('word'), 'noun': tuple('noun')}, 'woh!', similarity=.3)
      ('w', 'o', 'r', 'd')
      >>> df = pd.DataFrame(np.arange(6*2).reshape(2,6), columns=('alpha','beta','omega','begin','life','end'))
      >>> fuzzy_get_value(df, 'life')[0], fuzzy_get(df, 'omega')[0]
      (4, 2)
    """
    dict_obj = OrderedDict(obj)
    try:
        return dict_obj[dict_obj.keys()[int(approximate_key)]]
    except (ValueError, IndexError):
        pass
    return fuzzy_get(dict_obj, approximate_key, key_and_value=False, **kwargs)


def fuzzy_get_tuple(dict_obj, approximate_key, dict_keys=None, key_and_value=False, similarity=0.6, default=None):
    """Find the closest matching key and/or value in a dictionary (must have all string keys!)"""
    return fuzzy_get(dict(('|'.join(str(k2) for k2 in k), v) for (k, v) in dict_obj.iteritems()), 
                     '|'.join(str(k) for k in approximate_key), dict_keys=dict_keys, key_and_value=key_and_value, similarity=similarity, default=default)



def sod_transposed(seq_of_dicts, align=True, fill=True, filler=None):
    """Return sequence (list) of dictionaries, transposed into a dictionary of sequences (lists)
    
    >>> sorted(sod_transposed([{'c': 1, 'cm': u'P'}, {'c': 1, 'ct': 2, 'cm': 6, 'cn': u'MUS'}, {'c': 1, 'cm': u'Q', 'cn': u'ROM'}], filler=0).items())
    [('c', [1, 1, 1]), ('cm', [u'P', 6, u'Q']), ('cn', [0, u'MUS', u'ROM']), ('ct', [0, 2, 0])]
    >>> sorted(sod_transposed(({'c': 1, 'cm': u'P'}, {'c': 1, 'ct': 2, 'cm': 6, 'cn': u'MUS'}, {'c': 1, 'cm': u'Q', 'cn': u'ROM'}), fill=0, align=0).items())
    [('c', [1, 1, 1]), ('cm', [u'P', 6, u'Q']), ('cn', [u'MUS', u'ROM']), ('ct', [2])]
    """
    result = {}
    if isinstance(seq_of_dicts, collections.Mapping):
        seq_of_dicts = [seq_of_dicts]
    it = iter(seq_of_dicts)
    # if you don't need to align and/or fill, then just loop through and return
    if not (align and fill):
        for d in it:
            for k in d:
                result[k] = result.get(k, []) + [d[k]]
        return result
    # need to align and/or fill, so pad as necessary
    for i, d in enumerate(it):
        for k in d:
            result[k] = result.get(k, [filler] * (i * int(align))) + [d[k]]
        for k in result:
            if k not in d:
                result[k] += [filler]
    return result


def joined_seq(seq, sep=None):
    """Join a sequence into a tuple or a concatenated string

    >>> joined_seq(range(3), ', ')
    '0, 1, 2'
    >>> joined_seq([1, 2, 3])
    (1, 2, 3)
    """
    joined_seq = tuple(seq)
    if isinstance(sep, basestring):
        joined_seq = sep.join(str(item) for item in joined_seq)
    return joined_seq


def consolidate_stats(dict_of_seqs, stats_key=None, sep=','):
    """Join (stringify and concatenate) keys (table fields) in a dict (table) of sequences (columns)

    >>> consolidate_stats(dict([('c', [1, 1, 1]), ('cm', [u'P', 6, u'Q']), ('cn', [0, u'MUS', u'ROM']), ('ct', [0, 2, 0])]), stats_key='c')
    [{'P,0,0': 1}, {'6,MUS,2': 1}, {'Q,ROM,0': 1}]
    >>> consolidate_stats([{'c': 1, 'cm': 'P', 'cn': 0, 'ct': 0}, {'c': 1, 'cm': 6, 'cn': 'MUS', 'ct': 2}, {'c': 1, 'cm': 'Q', 'cn': 'ROM', 'ct': 0}], stats_key='c')
    [{'P,0,0': 1}, {'6,MUS,2': 1}, {'Q,ROM,0': 1}]
    """
    if isinstance(dict_of_seqs, dict):
        stats = dict_of_seqs[stats_key]
        keys = joined_seq(sorted([k for k in dict_of_seqs if k is not stats_key]), sep=None)
        joined_key = joined_seq(keys, sep=sep)
        result = {stats_key: [], joined_key: []}
        for i, statistic in enumerate(stats):
            result[stats_key] += [statistic]
            result[joined_key] += [joined_seq((dict_of_seqs[k][i] for k in keys if k is not stats_key), sep)]
        return list({k: result[stats_key][i]} for i, k in enumerate(result[joined_key]))
    return [{joined_seq((d[k] for k in sorted(d) if k is not stats_key), sep): d[stats_key]} for d in dict_of_seqs]


def dos_from_table(table, header=None):
    """Produce dictionary of sequences from sequence of sequences, optionally with a header "row".

    >>> dos_from_table([['hello', 'world'], [1, 2], [3,4]]) == {'hello': [1, 3], 'world': [2, 4]}
    True
    """
    start_row = 0
    if not table:
        return table
    if not header:
        header = table[0]
        start_row = 1
    header_list = header
    if header and isinstance(header, basestring):
        header_list = header.split('\t')
        if len(header_list)!=len(table[0]):
            header_list = header.split(',')
        if len(header_list)!=len(table[0]):
            header_list = header.split(' ')
    ans = {}
    for i, k in enumerate(header):
        ans[k] = [row[i] for row in table[start_row:]]
    return ans


def transposed_lists(list_of_lists, default=None):
    """Like `numpy.transposed`, but allows uneven row lengths

    Uneven lengths will affect the order of the elements in the rows of the transposed lists

    >>> transposed_lists([[1, 2], [3, 4, 5], [6]])
    [[1, 3, 6], [2, 4], [5]]
    >>> transposed_lists(transposed_lists([[], [1, 2, 3], [4]]))
    [[1, 2, 3], [4]]
    >>> l = transposed_lists([range(4),[4,5]])
    >>> l
    [[0, 4], [1, 5], [2], [3]]
    >>> transposed_lists(l)
    [[0, 1, 2, 3], [4, 5]]
    """
    if default is None or default is [] or default is tuple():
        default = []
    elif default is 'None':
        default = [None]
    else:
        default = [default]
    
    N = len(list_of_lists)
    Ms = [len(row) for row in list_of_lists]
    M = max(Ms)
    ans = []
    for j in range(M):
        ans += [[]]
        for i in range(N):
            if j < Ms[i]:
                ans[-1] += [list_of_lists[i][j]]
            else:
                ans[-1] += list(default)
    return ans


def transposed_matrix(matrix, filler=None, row_type=list, matrix_type=list, value_type=None):
    """Like numpy.transposed, evens up row (list) lengths that aren't uniform, filling with None.

    Also, makes all elements a uniform type (default=type(matrix[0][0])), 
    except for filler elements.

    TODO: add feature to delete None's at the end of rows so that transpose(transpose(LOL)) = LOL

    >>> transposed_matrix([[1, 2], [3, 4, 5], [6]])
    [[1, 3, 6], [2, 4, None], [None, 5, None]]
    >>> transposed_matrix(transposed_matrix([[1, 2], [3, 4, 5], [6]]))
    [[1, 2, None], [3, 4, 5], [6, None, None]]
    >>> transposed_matrix([[], [1, 2, 3], [4]])  # empty first row forces default value type (float)
    [[None, 1.0, 4.0], [None, 2.0, None], [None, 3.0, None]]
    >>> transposed_matrix(transposed_matrix([[], [1, 2, 3], [4]]))
    [[None, None, None], [1.0, 2.0, 3.0], [4.0, None, None]]
    >>> l = transposed_matrix([range(4),[4,5]])
    >>> l
    [[0, 4], [1, 5], [2, None], [3, None]]
    >>> transposed_matrix(l)
    [[0, 1, 2, 3], [4, 5, None, None]]
    >>> transposed_matrix([[1,2],[1],[1,2,3]])
    [[1, 1, 1], [2, None, 2], [None, None, 3]]
    """

    matrix_type = matrix_type or type(matrix)
    # matrix = matrix_type(matrix)

    try:
        row_type = row_type or type(matrix[0])
    except:
        pass
    if not row_type or row_type == type(None):
        row_type = list

    try:
        value_type = value_type or type(matrix[0][0]) or float
    except:
        pass
    if not value_type or value_type == type(None):
        value_type = float

    #print matrix_type, row_type, value_type

    # original matrix is NxM, new matrix will be MxN
    N = len(matrix)
    Ms = [len(row) for row in matrix]
    M = 0 if not Ms else max(Ms)

    ans = []
    # for each row in the new matrix (column in old matrix)
    for j in range(M):
        # add a row full of copies the `fill` value up to the maximum width required
        ans += [row_type([filler] * N)]
        for i in range(N):
            try:
                ans[j][i] = value_type(matrix[i][j])
            except IndexError:
                ans[j][i] = filler
            except TypeError:
                ans[j][i] = filler

    try:
        if isinstance(ans[0], row_type):
            return matrix_type(ans)
    except:
        pass
    return matrix_type([row_type(row) for row in ans])


def hist_from_counts(counts, normalize=False, cumulative=False, to_str=False, sep=',', min_bin=None, max_bin=None):
    """Compute an emprical histogram, PMF or CDF in a list of lists

    TESTME: compare results to hist_from_values_list and hist_from_float_values_list
    """
    counters = [dict((i, c)for i, c in enumerate(counts))]


    intkeys_list = [[c for c in counts_dict if (isinstance(c, int) or (isinstance(c, float) and int(c) == c))] for counts_dict in counters]
    min_bin, max_bin = min_bin or 0, max_bin or len(counts) - 1 

    histograms = []
    for intkeys, counts in zip(intkeys_list, counters):
        histograms += [OrderedDict()]
        if not intkeys:
            continue
        if normalize:
            N = sum(counts[c] for c in intkeys)
            for c in intkeys:
                counts[c] = float(counts[c]) / N
        if cumulative:
            for i in xrange(min_bin, max_bin + 1):
                histograms[-1][i] = counts.get(i, 0) + histograms[-1].get(i-1, 0)
        else:
            for i in xrange(min_bin, max_bin + 1):
                histograms[-1][i] = counts.get(i, 0)
    if not histograms:
        histograms = [OrderedDict()]

    # fill in the zero counts between the integer bins of the histogram
    aligned_histograms = []

    for i in range(min_bin, max_bin + 1):
        aligned_histograms += [tuple([i] + [hist.get(i, 0) for hist in histograms])]

    if to_str:
        # FIXME: add header row
        return str_from_table(aligned_histograms, sep=sep, max_rows=365*2+1)

    return aligned_histograms


def hist_from_values_list(values_list, fillers=(None,), normalize=False, cumulative=False, to_str=False, sep=',', min_bin=None, max_bin=None):
    """Compute an emprical histogram, PMF or CDF in a list of lists or a csv string

    Only works for discrete (integer) values (doesn't bin real values).
    `fillers`: list or tuple of values to ignore in computing the histogram

    >>> hist_from_values_list([1,1,2,1,1,1,2,3,2,4,4,5,7,7,9])  # doctest: +NORMALIZE_WHITESPACE
    [(1, 5), (2, 3), (3, 1), (4, 2), (5, 1), (6, 0), (7, 2), (8, 0), (9, 1)]
    >>> hist_from_values_list([(1,9),(1,8),(2,),(1,),(1,4),(2,5),(3,3),(5,0),(2,2)])  # doctest: +NORMALIZE_WHITESPACE
    [[(1, 4), (2, 3), (3, 1), (4, 0), (5, 1)], [(0, 1), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 0), (7, 0), (8, 1), (9, 1)]]
    >>> hist_from_values_list(transposed_matrix([(8,),(1,3,5),(2,),(3,4,5,8)]))  # doctest: +NORMALIZE_WHITESPACE
    [[(8, 1)], [(1, 1), (2, 0), (3, 1), (4, 0), (5, 1)], [(2, 1)], [(3, 1), (4, 1), (5, 1), (6, 0), (7, 0), (8, 1)]]
    """
    value_types = tuple([int, float] + [type(filler) for filler in fillers])

    if all(isinstance(value, value_types) for value in values_list):
        # ignore all fillers and convert all floats to ints when doing counting
        counters = [collections.Counter(int(value) for value in values_list if isinstance(value, (int, float)))]
    elif all(len(row)==1 for row in values_list) and all(isinstance(row[0], value_types) for row in values_list):
        return hist_from_values_list([values[0] for values in values_list], fillers=fillers, normalize=normalize, cumulative=cumulative, to_str=to_str, sep=sep, min_bin=min_bin, max_bin=max_bin)
    else:  # assume it's a row-wise table (list of rows)
        return [
            hist_from_values_list(col, fillers=fillers, normalize=normalize, cumulative=cumulative, to_str=to_str, sep=sep, min_bin=min_bin, max_bin=max_bin)
            for col in transposed_matrix(values_list)
            ]

    if not values_list:
        return []

    intkeys_list = [[c for c in counts if (isinstance(c, int) or (isinstance(c, float) and int(c) == c))] for counts in counters]
    try:
        min_bin = int(min_bin)
    except:
        min_bin = min(min(intkeys) for intkeys in intkeys_list)
    try:
        max_bin = int(max_bin)
    except:
        max_bin = max(max(intkeys) for intkeys in intkeys_list)

    # FIXME: this looks slow and hazardous (like it's ignore min/max bin):
    min_bin = max(min_bin, min((min(intkeys) if intkeys else 0) for intkeys in intkeys_list))  # TODO: reuse min(intkeys)
    max_bin = min(max_bin, max((max(intkeys) if intkeys else 0) for intkeys in intkeys_list))  # TODO: reuse max(intkeys)

    histograms = []
    for intkeys, counts in zip(intkeys_list, counters):
        histograms += [OrderedDict()]
        if not intkeys:
            continue
        if normalize:
            N = sum(counts[c] for c in intkeys)
            for c in intkeys:
                counts[c] = float(counts[c]) / N
        if cumulative:
            for i in xrange(min_bin, max_bin + 1):
                histograms[-1][i] = counts.get(i, 0) + histograms[-1].get(i-1, 0)
        else:
            for i in xrange(min_bin, max_bin + 1):
                histograms[-1][i] = counts.get(i, 0)
    if not histograms:
        histograms = [OrderedDict()]

    # fill in the zero counts between the integer bins of the histogram
    aligned_histograms = []

    for i in range(min_bin, max_bin + 1):
        aligned_histograms += [tuple([i] + [hist.get(i, 0) for hist in histograms])]

    if to_str:
        # FIXME: add header row
        return str_from_table(aligned_histograms, sep=sep, max_rows=365*2+1)

    return aligned_histograms


def flatten_csv(path='.', ext='csv', date_parser=parse_date, verbosity=0, output_ext=None):
    """Load all CSV files in the given path, write .flat.csv files, return `DataFrame`s

    Arguments:
      path (str): file or folder to retrieve CSV files and `pandas.DataFrame`s from
      ext (str): file name extension (to filter files by)
      date_parser (function): if the MultiIndex can be interpretted as a datetime, this parser will be used

    Returns:
      dict of DataFrame: { file_path: flattened_data_frame }
    """
    date_parser = date_parser or (lambda x: x)
    dotted_ext, dotted_output_ext = None, None
    if ext != None and output_ext != None:
        dotted_ext = ('' if ext.startswith('.') else '.') + ext
        dotted_output_ext = ('' if output_ext.startswith('.') else '.') + output_ext
    table = {}
    for file_properties in find_files(path, ext=ext or '', verbosity=verbosity):
        file_path = file_properties['path']
        if output_ext and (dotted_output_ext + '.') in file_path:
            continue
        df = pd.DataFrame.from_csv(file_path, parse_dates=False)
        df = flatten_dataframe(df)
        if dotted_ext != None and dotted_output_ext != None:
            df.to_csv(file_path[:-len(dotted_ext)] + dotted_output_ext + dotted_ext)
        table[file_path] = df
    return table


def get_similar(obj, labels, default=None, min_similarity=0.5):
    """Similar to fuzzy_get, but allows non-string keys and a list of possible keys

    Searches attributes in addition to keys and indexes to find the closest match.

    See Also:
        `fuzzy_get`

    """
    raise NotImplementedError("Unfinished implementation, needs to be incorporated into fuzzy_get where a list of scores and keywords is sorted.")
    labels = listify(labels)
    not_found = lambda: 0
    min_score = int(min_similarity * 100)
    for similarity_score in [100, 95, 90, 80, 70, 50, 30, 10, 5, 0]:
        if similarity_score <= min_score:
            similarity_score = min_score
        for label in labels:
            try:
                result = obj.get(label, not_found)
            except AttributeError:
                try:
                    result = obj.__getitem__(label)
                except (IndexError, TypeError):
                    result = not_found
            if not result is not_found:
                return result
        if similarity_score == min_score:
            if not result is not_found:
                return result


def normalize_column_labels(obj, labels):
    """Like `get_similar` but returns the matched labels/keys rather than the values and 1 key for each label in labels"""


def update_dict(d, u=None, depth=-1, take_new=True, default_mapping_type=dict, prefer_update_type=False, copy=False):
    """
    Recursively merge (union or update) dict-like objects (collections.Mapping) to the specified depth.

    >>> update_dict({'k1': {'k2': 2}}, {'k1': {'k2': {'k3': 3}}, 'k4': 4})
    {'k1': {'k2': {'k3': 3}}, 'k4': 4}
    >>> update_dict(OrderedDict([('k1', OrderedDict([('k2', 2)]))]), {'k1': {'k2': {'k3': 3}}, 'k4': 4})
    OrderedDict([('k1', OrderedDict([('k2', {'k3': 3})])), ('k4', 4)])
    >>> update_dict(OrderedDict([('k1', dict([('k2', 2)]))]), {'k1': {'k2': {'k3': 3}}, 'k4': 4})
    OrderedDict([('k1', {'k2': {'k3': 3}}), ('k4', 4)])
    >>> orig = {'orig_key': 'orig_value'}
    >>> updated = update_dict(orig, {'new_key': 'new_value'}, copy=True)
    >>> updated == orig
    False
    >>> updated2 = update_dict(orig, {'new_key2': 'new_value2'})
    >>> updated2 == orig
    True
    >>> update_dict({'k1': {'k2': {'k3': 3}}, 'k4': 4}, {'k1': {'k2': 2}}, depth=1, take_new=False)
    {'k1': {'k2': 2}, 'k4': 4}
    >>> update_dict({'k1': {'k2': {'k3': 3}}, 'k4': 4}, None)
    {'k1': {'k2': {'k3': 3}}, 'k4': 4}
    >>> update_dict({'k1': {'k2': {'k3': 3}}, 'k4': 4}, {'k1': ()})
    {'k1': (), 'k4': 4}
    >>> # FIXME: this result is unexpected the same as for `take_new=False`
    >>> update_dict({'k1': {'k2': {'k3': 3}}, 'k4': 4}, {'k1': {'k2': 2}}, depth=1, take_new=True)
    {'k1': {'k2': 2}, 'k4': 4}
    """
    u = u or {}
    orig_mapping_type = type(d)
    if prefer_update_type and isinstance(u, collections.Mapping):
        dictish = type(u)
    elif isinstance(d, collections.Mapping):
        dictish = orig_mapping_type
    else:
        dictish = default_mapping_type
    if copy:
        d = dictish(d)
    for k, v in u.iteritems():
        if isinstance(d, collections.Mapping):
            if isinstance(v, collections.Mapping) and not depth == 0:
                r = update_dict(d.get(k, dictish()), v, depth=max(depth - 1, -1), copy=copy)
                d[k] = r
            elif take_new:
                d[k] = u[k]
        elif take_new:
            d = dictish([(k, u[k])])
    return d


def mapped_transposed_lists(lists, default=None):
    """
    Swap rows and columns in list of lists with different length rows/columns

    Pattern from
    http://code.activestate.com/recipes/410687-transposing-a-list-of-lists-with-different-lengths/
    Replaces any zeros or Nones with default value.

    Examples:
    >>> l = mapped_transposed_lists([range(4),[4,5]],None)
    >>> l
    [[0, 4], [1, 5], [2, None], [3, None]]
    >>> mapped_transposed_lists(l)
    [[0, 1, 2, 3], [4, 5, None, None]]
    """
    if not lists:
        return []
    return map(lambda *row: [el if isinstance(el, (float, int)) else default for el in row], *lists)


def make_name(s, camel=None, lower=None, space='_', remove_prefix=None, language='python', string_type=unicode):
    """Process a string to produce a valid python variable/class/type name

    Arguments:
      space (str): string to substitute for spaces ('' to delete all whitespace)
      camel (bool): whether to camel-case names, Django Model Name style (first letter capitalized)
      lower (bool): whether to lowercase all strings 
      language (str): case-insensitive language identifier (to deterimine allowable identifier characters)
        e.g. 'Python', 'Python2', 'Python3', 'Javascript', 'ECMA'

    Examples:
      Generate Django model names out of file names
      >>> make_name('women in IT.csv', camel=True)
      u'WomenInItCsv'
      
      Generate Django field names out of CSV header strings
      >>> make_name('ID Number (9-digits)')
      u'id_number_9_digits_'
      >>> make_name("PD / SZ")
      u'pd_sz'

      Generate Javscript object attribute names from CSV header strings
      >>> make_name(u'pi (\u03C0)', space = '', language='javascript')
      u'pi\u03c0'
      >>> make_name(u'pi (\u03C0)', space = '', language='javascript')
      u'pi\u03c0'
    """
    if camel is None and lower is None:
        lower = True
    if not s:
        return None
    ecma_languages = ['ecma', 'javasc']
    unicode_languages = ecma_languages
    language = language or 'python'
    language = language.lower().strip()[:6]
    string_type = string_type or str
    if language in unicode_languages:
        string_type = unicode
    s = string_type(s)  # TODO: encode in ASCII, UTF-8, or the charset used for this file!
    if remove_prefix and s.startswith(remove_prefix):
        s = s[len(remove_prefix):]
    if camel:
        if space and space == '_':
            space = ''
        if any(c in ' \t\n\r' + string.punctuation for c in s) or s.lower() == s:
            if lower:
                s = s.lower()
            s = s.title()
    elif lower:
        s = s.lower()
    # TODO: add language Regexes to filter characters appropriately for python or javascript
    space_escape = '\\' if space and space not in ' _' else ''
    if not language in ecma_languages:
        invalid_char_regex = re.compile('[^a-zA-Z0-9' + space_escape + space +']+')
    else:
        # FIXME: Unicode categories and properties only works in Perl Regexes!
        invalid_char_regex = re.compile('[\W' + space_escape + space +']+', re.UNICODE)
    if space is not None:
        # get rid of all invalid characters, substitting the space-filler for them all
        s = invalid_char_regex.sub(space, s)
        # get rid of duplicate space-filler characters
        if space:
            s = re.sub('[' + space_escape + space + ']{2,}', space, s)
    return s
make_name.DJANGO_FIELD = {'camel': False, 'lower': True, 'space': '_'}
make_name.DJANGO_MODEL = {'camel': True, 'lower': False, 'space': '', 'remove_prefix': 'models'}


def make_filename(s, space=None, language='msdos', strict=False, max_len=None, repeats=1024):
    r"""Process string to remove any characters not allowed by the language specified (default: MSDOS)

    In addition, optionally replace spaces with the indicated "space" character
    (to make the path useful in a copy-paste without quoting).

    Uses the following regular expression to substitute spaces for invalid characters:

        re.sub(r'[ :\\/?*&"<>|~`!]{1}', space, s)

    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', strict=False)
    'Whatever-crazy-s-$h-7-n-m3-ou-can-come-up.-with.-txt-'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', strict=False, repeats=1)
    'Whatever-crazy--s-$h-7-n-m3----ou--can-come-up.-with.-txt--'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', repeats=1)
    'Whatever-crazy--s-$h-7-n-m3----ou--can-come-up.-with.-txt--'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!')
    'Whatever-crazy-s-$h-7-n-m3-ou-can-come-up.-with.-txt-'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', strict=True, repeats=1)
    u'Whatever_crazy_s_h_7_n_m3_ou_can_come_up_with_txt_'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', strict=True, repeats=1, max_len=14)
    u'Whatever_crazy'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', max_len=14)
    'Whatever-crazy'
    """
    filename = None
    if strict or language.lower().strip() in ('strict', 'variable', 'expression', 'python'):
        if space == None:
            space = '_'
        elif not space:
            space = ''
        filename = make_name(s, space=space, lower=False)
    else:
        if space == None:
            space = '-'
        elif not space:
            space = ''
    if not filename:
        if language.lower().strip() in ('posix', 'unix', 'linux', 'centos', 'ubuntu', 'fedora', 'redhat', 'rhel', 'debian', 'deb'):
            filename = re.sub(r'[^0-9A-Za-z._-]' + '\{1,{0}\}'.format(repeats), space, s)
        else:
            filename = re.sub(r'[ :\\/?*&"<>|~`!]{' + ('1,{0}'.format(repeats)) + r'}', space, s)
    if max_len and int(max_len) > 0 and filename:
        return filename[:int(max_len)]
    else:
        return filename


def update_file_ext(filename, ext='txt', sep='.'):
    """Force the file or path str to end with the indicated extension

    Note: a dot (".") is assumed to delimit the extension

    >>> update_file_ext('/home/hobs/extremofile', 'bac')
    '/home/hobs/extremofile.bac'
    >>> update_file_ext('/home/hobs/piano.file/', 'music')
    '/home/hobs/piano.file/.music'
    >>> update_file_ext('/home/ninja.hobs/Anglofile', '.uk')
    '/home/ninja.hobs/Anglofile.uk'
    >>> update_file_ext('/home/ninja-corsi/audio', 'file', sep='-')
    '/home/ninja-corsi/audio-file'
    """ 
    path, filename = os.path.split(filename)

    if ext and ext[0] == sep:
        ext = ext[1:]
    return os.path.join(path, sep.join(filename.split(sep)[:-1 if filename.count(sep) > 1 else 1] + [ext]))


def tryconvert(value, desired_types=SCALAR_TYPES, default=None, empty='', strip=True):
    """
    Convert value to one of the desired_types specified (in order of preference) without raising an exception.

    If value is empty is a string and Falsey, then return the `empty` value specified.
    If value can't be converted to any of the desired_types requested, then return the `default` value specified.

    >>> tryconvert('MILLEN2000', desired_types=float, default='GENX')
    'GENX'
    >>> tryconvert('1.23', desired_types=[int,float], default='default')
    1.23
    >>> tryconvert('-1.0', desired_types=[int,float])  # assumes you want a float if you have a trailing .0 in a str
    -1.0
    >>> tryconvert(-1.0, desired_types=[int,float])  # assumes you want an int if int type listed first
    -1
    >>> repr(tryconvert('1+1', desired_types=[int,float]))
    'None'
    """
    if value in tryconvert.EMPTY:
        if isinstance(value, basestring):
            return type(value)(empty)
        return empty
    if isinstance(value, basestring):
        # there may not be any "empty" strings that won't be caught by the `is ''` check above, but just in case
        if not value:
            return type(value)(empty)
        if strip:
            value = value.strip()
    if isinstance(desired_types, type):
        desired_types = (desired_types,)
    if desired_types is not None and len(desired_types) == 0:
        desired_types = tryconvert.SCALAR
    if len(desired_types):
        if isinstance(desired_types, (list, tuple)) and len(desired_types) and isinstance(desired_types[0], (list, tuple)):
            desired_types = desired_types[0]
        elif isinstance(desired_types, type):
            desired_types = [desired_types]
    for t in desired_types:
        try:
            return t(value)
        except (ValueError, TypeError, InvalidOperation, InvalidContext):
            continue
        # if any other weird exception happens then need to get out of here
        return default
    # if no conversions happened successfully then return the default value requested
    return default
tryconvert.EMPTY = ('', None, float('nan'))
tryconvert.SCALAR = SCALAR_TYPES


def transcode(infile, outfile=None, incoding="shift-jis", outcoding="utf-8"):
    """Change encoding of text file"""
    if not outfile:
        outfile = os.path.basename(infile) + '.utf8'
    with codecs.open(infile, "rb", incoding) as fpin:
        with codecs.open(outfile, "wb", outcoding) as fpout:
            fpout.write(fpin.read())


def strip_br(s):
    r""" Strip the trailing html linebreak character (<BR />) from a string or sequence of strings 

    A sequence of strings is assumed to be a row in a CSV/TSV file or words from a line of text
    so only the last element in a sequence is "stripped"

    >>> strip_br(' Title <BR> ')
    ' Title'
    >>> strip_br(range(1,4))
    [1, 2, 3]
    >>> strip_br((' Column 1<br />', ' Last Column < br / >  '))
    (' Column 1<br />', ' Last Column')
    >>> strip_br(['name', 'rank', 'serial\nnumber', 'date <BR />'])
    ['name', 'rank', 'serial\nnumber', 'date']
    >>> strip_br(None)
    >>> strip_br([])
    []
    >>> strip_br(())
    ()
    >>> strip_br(('one element<br>',))
    ('one element',)
    """

    if isinstance(s, basestring):
        return re.sub(r'\s*<\s*[Bb][Rr]\s*[/]?\s*>\s*$','', s)
    elif isinstance(s, (tuple, list)):
        # strip just the last element in a list or tuple
        try:
            return type(s)(list(s)[:-1] + [strip_br(s[-1])])
        except:  # len(s) == 0
            return s
    else:
        try:
            return type(s)(strip_br(str(s)))
        except:  # s is None
            return s


def read_csv(csv_file, ext='.csv', format=None, delete_empty_keys=False,
             fieldnames=[], rowlimit=100000000, numbers=False, normalize_names=True, unique_names=True,
             verbosity=0):
    r"""
    Read a csv file from a path or file pointer, returning a dict of lists, or list of lists (according to `format`)

    filename: a directory or list of file paths
    numbers: whether to attempt to convert strings in csv to numbers

    TODO:
        merge with `nlp.util.make_dataframe` function

    Handles unquoted and quoted strings, quoted commas, quoted newlines (EOLs), complex numbers, times, dates, datetimes,
    >>> read_csv('"name\r\n",rank,"serial\nnumber",date <BR />\t\n"McCain, John","1","123456789",9/11/2001\nBob,big cheese,1-23,1/1/2001 12:00 GMT', format='header+values list', numbers=True)  # doctest: +NORMALIZE_WHITESPACE
    [['name', 'rank', 'serial\nnumber', 'date'],
     ['McCain, John', 1.0, 123456789.0, datetime.datetime(2001, 9, 11, 0, 0)],
     ['Bob',
      'big cheese',
      datetime.datetime(2015, 1, 23, 0, 0),
      datetime.datetime(2001, 1, 1, 12, 0, tzinfo=tzutc())]]
    """
    if not csv_file:
        return
    if isinstance(csv_file, basestring):
        # truncate `csv_file` in case it is a string buffer containing GBs of data
        path = csv_file[:1025]
        try:
            # see http://stackoverflow.com/a/4169762/623735 before trying 'rU'
            fpin = open(path, 'rUb')  # U = universal EOL reader, b = binary
        except:
            # truncate path more, in case path is used later as a file description:
            path = csv_file[:128]
            fpin = StringIO(csv_file)
    else:
        fpin = csv_file
        try:
            path = csv_file.name
        except:
            path = 'unknown file buffer path'

    format = format or 'h'
    format = format[0].lower()

    # if fieldnames not specified then assume that first row of csv contains headings
    csvr = csv.reader(fpin, dialect=csv.excel)
    if not fieldnames:
        while not fieldnames or not any(fieldnames):
            fieldnames = strip_br([str(s).strip() for s in csvr.next()])
        if verbosity > 0:
            logger.info('Column Labels: ' + repr(fieldnames))
    if unique_names:
        norm_names = OrderedDict([(fldnm, fldnm) for fldnm in fieldnames])
    else:
        norm_names = OrderedDict([(num, fldnm) for num, fldnm in enumerate(fieldnames)])
    if normalize_names:
        norm_names = OrderedDict([(num, make_name(fldnm, **make_name.DJANGO_FIELD)) for num, fldnm in enumerate(fieldnames)])
        # required for django-formatted json files
        model_name = make_name(path, **make_name.DJANGO_MODEL)
    if format in 'c':  # columnwise dict of lists
        recs = OrderedDict((norm_name, []) for norm_name in norm_names.values())
    elif format in 'vh':
        recs = [fieldnames]
    else:
        recs = []
    if verbosity > 0:
        logger.info('Field Names: ' + repr(norm_names if normalize_names else fieldnames))
    rownum = 0
    eof = False
    pbar = None
    start_seek_pos = fpin.tell() or 0
    if verbosity > 1:
        print('Starting at byte {} in file buffer.'.format(start_seek_pos))
    fpin.seek(0, os.SEEK_END)
    file_len = fpin.tell() - start_seek_pos  # os.fstat(fpin.fileno()).st_size
    fpin.seek(start_seek_pos)

    if verbosity > 1:
        print('There appear to be {} bytes remaining in the file buffer. Resetting (seek) to starting position in file.'.format(file_len))
    if verbosity > 0:
        pbar = progressbar.ProgressBar(maxval=file_len)
        pbar.start()
    while csvr and rownum < rowlimit and not eof:
        if pbar:
            pbar.update(fpin.tell() - start_seek_pos)
        rownum += 1
        row = []
        row_dict = OrderedDict()
        # skip rows with all empty strings as values,
        while not row or not any(len(x) for x in row):
            try:
                row = csvr.next()
                if verbosity > 1:
                    logger.info('  row content: ' + repr(row))
            except StopIteration:
                eof = True
                break
        if eof:
            break
        if len(row) and isinstance(row[-1], basestring) and len(row[-1]):
            row = strip_br(row)
        if numbers:
            # try to convert the type to a numerical scalar type (int, float etc)
            row = [tryconvert(v, desired_types=NUMBERS_AND_DATETIMES, empty=None, default=v) for v in row]
        if row:
            N = min(max(len(row), 0), len(norm_names))
            row_dict = OrderedDict(((field_name, field_value) 
                for field_name, field_value in zip(list(norm_names.values() 
                    if unique_names else norm_names)[:N],row[:N]) 
                        if (str(field_name).strip() or delete_empty_keys is False)))
            if format in 'dj':  # django json format
                recs += [{"pk": rownum, "model": model_name, "fields": row_dict}]
            elif format in 'vhl':  # list of lists of values, with header row (list of str)
                recs += [[value for field_name, value in row_dict.iteritems() if (field_name.strip() or delete_empty_keys is False)]]
            elif format in 'c':  # columnwise dict of lists
                for field_name in row_dict:
                    recs[field_name] += [row_dict[field_name]]
                if verbosity > 2:
                    print([recs[field_name][-1] for field_name in row_dict])
            else:
                recs += [row_dict]
            if verbosity > 2 and not format in 'c':
                print(recs[-1])

    if file_len > fpin.tell():
        logger.info("Only %d of %d bytes were read and processed." % (fpin.tell(), file_len))
    if pbar:
        pbar.finish()
    fpin.close()
    if not unique_names:
        return recs, norm_names
    return recs


# date and datetime separators
COLUMN_SEP = re.compile(r'[,/;]')


def make_dataframe(prices, num_prices=1, columns=('portfolio',)):
    """Convert a file, list of strings, or list of tuples into a Pandas DataFrame

    Arguments:
      num_prices (int): if not null, the number of columns (from right) that contain numeric values
    """
    if isinstance(prices, pd.Series):
        return pd.DataFrame(prices)
    if isinstance(prices, pd.DataFrame):
        return prices
    if isinstance(prices, basestring) and os.path.isfile(prices):
        prices = open(prices, 'rU')
    if isinstance(prices, file):
        values = []
        # FIXME: what if it's not a CSV but a TSV or PSV
        csvreader = csv.reader(prices, dialect='excel', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            # print row
            values += [row]
        prices.close()
        prices = values
    if any(isinstance(row, basestring) for row in prices):
        prices = [COLUMN_SEP.split(row) for row in prices]
    # print prices
    index = []
    if isinstance(prices[0][0], (datetime.date, datetime.datetime, datetime.time)):
        index = [prices[0] for row in prices]
        for i, row in prices:
            prices[i] = row[1:]
    # try to convert all strings to something numerical:
    elif any(any(isinstance(value, basestring) for value in row) for row in prices):
        #print '-'*80
        for i, row in enumerate(prices):
            #print i, row
            for j, value in enumerate(row):
                s = unicode(value).strip().strip('"').strip("'")
                #print i, j, s
                try:
                    prices[i][j] = int(s)
                    # print prices[i][j]
                except:
                    try:
                        prices[i][j] = float(s)
                    except:
                        # print 'FAIL'
                        try:
                            # this is a probably a bit too forceful
                            prices[i][j] = parse_date(s)
                        except:
                            pass
    # print prices
    width = max(len(row) for row in prices)
    datetime_width = width - num_prices
    if not index and isinstance(prices[0], (tuple, list)) and num_prices:
        # print '~'*80
        new_prices = []
        try:
            for i, row in enumerate(prices):
                # print i, row
                index += [datetime.datetime(*[int(i) for i in row[:datetime_width]])
                          + datetime.timedelta(hours=16)]
                new_prices += [row[datetime_width:]]
                # print prices[-1]
        except:
            for i, row in enumerate(prices):
                index += [row[0]]
                new_prices += [row[1:]]
        prices = new_prices or prices
    # print index
    # TODO: label the columns somehow (if first row is a bunch of strings/header)
    if len(index) == len(prices):
        df = pd.DataFrame(prices, index=index, columns=columns)
    else:
        df = pd.DataFrame(prices)
    return df


def column_name_to_date(name):
    """
    TODO: should probably assume a 2000 epoch for 2-digit dates

    >>> column_name_to_date('10-Apr')
    datetime.date(10, 4, 1)
    >>> column_name_to_date('10_2011')
    datetime.date(2011, 10, 1)
    >>> column_name_to_date('apr_10')
    datetime.date(10, 4, 1)
    """
    month_nums = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    year_month = re.split(r'[^0-9a-zA-Z]{1}', name)
    try:
        year = int(year_month[0])
        month = year_month[1]
    except:
        year = int(year_month[1])
        month = year_month[0]
    month = month_nums.get(str(month).lower().title(), None)
    if 0 <= year <= 2100 and 1 <= month <= 12:
        return datetime.date(year, month, 1)
    try:
        year = int(year_month[1])
        month = int(year_month[0])
    except:
        year. month = 0, 0
    if 0 <= year <= 2100 and 1 <= month <= 12:
        return datetime.date(year, month, 1)
    try:
        month = int(year_month[1])
        year = int(year_month[0])
    except:
        year. month = 0, 0
    if 0 <= year <= 2100 and 1 <= month <= 12:
        return datetime.date(year, month, 1)



def first_digits(s, default=0):
    """Return the fist (left-hand) digits in a string as a single integer, ignoring sign (+/-).
    >>> first_digits('+123.456')
    123
    """
    s = re.split(r'[^0-9]+', str(s).strip().lstrip('+-' + charlist.whitespace))
    if len(s) and len(s[0]):
        return int(s[0])
    return default


def int_pair(s, default=(0, None)):
    """Return the digits to either side of a single non-digit character as a 2-tuple of integers

    >>> int_pair('90210-007')
    (90210, 7)
    >>> int_pair('04321.0123')
    (4321, 123)
    """
    s = re.split(r'[^0-9]+', str(s).strip())
    if len(s) and len(s[0]):
        if len(s) > 1 and len(s[1]):
            return (int(s[0]), int(s[1]))
        return (int(s[0]), default[1])
    return default


def make_us_postal_code(s, allowed_lengths=(), allowed_digits=()):
    """
    >>> make_us_postal_code(1234)
    '01234'
    >>> make_us_postal_code(507.6009)
    '507'
    >>> make_us_postal_code(90210.0)
    '90210'
    >>> make_us_postal_code(39567.7226)
    '39567-7226'
    >>> make_us_postal_code(39567.7226)
    '39567-7226'
    """
    allowed_lengths = allowed_lengths or tuple(N if N < 6 else N + 1 for N in allowed_digits)
    allowed_lengths = allowed_lengths or (2, 3, 5, 10)
    ints = int_pair(s)
    z = str(ints[0]) if ints[0] else ''
    z4 = '-' + str(ints[1]) if ints[1] else ''
    if len(z) == 4:
        z = '0' + z
    if len(z + z4) in allowed_lengths:
        return z + z4
    elif len(z) in (min(l, 5) for l in allowed_lengths):
        return z
    return ''

# TODO: create and check MYSQL_MAX_FLOAT constant
def make_float(s, default='', ignore_commas=True):
    r"""Coerce a string into a float

    >>> make_float('12,345')
    12345.0
    >>> make_float('12.345')
    12.345
    >>> make_float('1+2')
    3.0
    >>> make_float('+42.0')
    42.0
    >>> make_float('\r\n-42?\r\n')
    -42.0
    >>> make_float('$42.42')
    42.42
    >>> make_float('B-52')
    -52.0
    >>> make_float('1.2 x 10^34')
    1.2e+34
    >>> make_float(float('nan'))
    nan
    >>> make_float(float('-INF'))
    -inf
    """
    if ignore_commas and isinstance(s, basestring):
        s = s.replace(',', '')
    try:
        return float(s)
    except:
        try:
            return float(str(s))
        except ValueError:
            try:
                return float(normalize_scientific_notation(str(s), ignore_commas))
            except ValueError:
                try:
                    return float(first_digits(s))
                except ValueError:
                    return default


# TODO: create and check MYSQL_MAX_FLOAT constant
def make_int(s, default='', ignore_commas=True):
    r"""Coerce a string into an integer (long ints will fail)

    TODO:
    - Ignore dashes and other punctuation within a long string of digits,
       like a telephone number, partnumber, datecode or serial number.
    - Use the Decimal type to allow infinite precision
    - Use regexes to be more robust

    >>> make_int('12345')
    12345
    >>> make_int('0000012345000       ')
    12345000
    >>> make_int(' \t\n123,450,00\n')
    12345000
    """
    if ignore_commas and isinstance(s, basestring):
        s = s.replace(',', '')
    try:
        return int(s)
    except:
        pass
    try:
        return int(re.split(str(s), '[^-0-9,.Ee]')[0])
    except ValueError:
        try:
            return int(float(normalize_scientific_notation(str(s), ignore_commas)))
        except (ValueError, TypeError):
            try:
                return int(first_digits(s))
            except (ValueError, TypeError):
                return default


# FIXME: use locale and/or check that they occur ever 3 digits (1000's places) to decide whether to ignore commas
def normalize_scientific_notation(s, ignore_commas=True, verbosity=1):
    """Produce a string convertable with float(s), if possible, fixing some common scientific notations

    Deletes commas and allows addition.
    >>> normalize_scientific_notation(' -123 x 10^-45 ')
    '-123e-45'
    >>> normalize_scientific_notation(' -1+1,234 x 10^-5,678 ')
    '1233e-5678'
    >>> normalize_scientific_notation('$42.42')
    '42.42'
    """
    s = s.lstrip(charlist.not_digits_nor_sign)
    s = s.rstrip(charlist.not_digits)
    #print s
    # TODO: substitute ** for ^ and just eval the expression rather than insisting on a base-10 representation
    num_strings = RE.scientific_notation_exponent.split(s, maxsplit=2)
    #print num_strings
    # get rid of commas
    s = RE.re.sub(r"[^.0-9-+" + "," * int(not ignore_commas) + r"]+", '', num_strings[0])
    #print s
    # if this value gets so large that it requires an exponential notation, this will break the conversion
    if not s:
        return None
    try:
        s = str(eval(s.strip().lstrip('0')))
    except:
        if verbosity > 1:
            print 'Unable to evaluate %s' % repr(s)
        try:
            s = str(float(s))
        except:
            print 'Unable to float %s' % repr(s)
            s = ''
    #print s
    if len(num_strings) > 1:
        if not s:
            s = '1'
        s += 'e' + RE.re.sub(r'[^.0-9-+]+', '', num_strings[1])
    if s:
        return s
    return None


def normalize_names(names):
    """Coerce a string or nested list of strings into a flat list of strings."""
    if isinstance(names, basestring):
        names = names.split(',')
    names = listify(names)
    return [str(name).strip() for name in names]


def string_stats(strs, valid_chars='012346789', left_pad='0', right_pad='', strip=True):
    """Count the occurrence of a category of valid characters within an iterable of serial numbers, model numbers, or other strings"""
    if left_pad == None:
        left_pad = ''.join(c for c in RE.ASCII_CHARACTERS if c not in valid_chars)
    if right_pad == None:
        right_pad = ''.join(c for c in RE.ASCII_CHARACTERS if c not in valid_chars)

    def normalize(s):
        if strip:
            s = s.strip()
        s = s.lstrip(left_pad)
        s = s.rstrip(right_pad)
        return s

    # should probably check to make sure memory not exceeded
    strs = [normalize(s) for s in strs]
    lengths = collections.Counter(len(s) for s in strs)
    counts = {}
    max_length = max(lengths.keys())

    for i in range(max_length):
        # print i
        for s in strs:
            if i < len(s):
                counts[ i]   = counts.get( i  , 0) + int(s[ i  ] in valid_chars)
                counts[-i-1] = counts.get(-i-1, 0) + int(s[-i-1] in valid_chars)
        long_enough_strings = float(sum(c for l, c in lengths.items() if l >= i))
        counts[i] = counts[i] / long_enough_strings
        counts[-i-1] = counts[-i-1] / long_enough_strings

    return counts


def normalize_serial_number(sn, 
                            max_length=None, left_fill='0', right_fill='', blank='', 
                            valid_chars=' -0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', 
                            invalid_chars=None, 
                            strip_whitespace=True, join=False, na=RE.nones):
    r"""Make a string compatible with typical serial number requirements

    # Default configuration strips internal and external whitespaces and retains only the last 10 characters

    >>> normalize_serial_number('1C 234567890             ')
    '0234567890'

    >>> normalize_serial_number('1C 234567890             ', max_length=20)
    '000000001C 234567890'
    >>> normalize_serial_number('Unknown', blank=None, left_fill='')
    ''
    >>> normalize_serial_number('N/A', blank='', left_fill='')
    'A'
    
    >>> normalize_serial_number('1C 234567890             ', max_length=20, left_fill='')
    '1C 234567890'

    Notice how the max_length setting (20) carries over from the previous test!
    >>> len(normalize_serial_number('Unknown', blank=False))
    20
    >>> normalize_serial_number('Unknown', blank=False)
    '00000000000000000000'
    >>> normalize_serial_number(' \t1C\t-\t234567890 \x00\x7f', max_length=14, left_fill='0', valid_chars='0123456789ABC', invalid_chars=None, join=True)
    '0001C234567890'

    Notice how the max_length setting carries over from the previous test!
    >>> len(normalize_serial_number('Unknown', blank=False))
    14

    Restore the default max_length setting
    >>> len(normalize_serial_number('Unknown', blank=False, max_length=10))
    10
    >>> normalize_serial_number('NO SERIAL', blank='--=--', left_fill='')  # doctest: +NORMALIZE_WHITESPACE
    'NO SERIAL'
    >>> normalize_serial_number('NO SERIAL', blank='', left_fill='')  # doctest: +NORMALIZE_WHITESPACE
    'NO SERIAL'

    >>> normalize_serial_number('1C 234567890             ', valid_chars='0123456789')
    '0234567890'
    """
    # All 9 kwargs have persistent default values stored as attributes of the funcion instance
    if max_length is None:
        max_length = normalize_serial_number.max_length
    else:
        normalize_serial_number.max_length = max_length
    if left_fill is None:
        left_fill = normalize_serial_number.left_fill
    else:
        normalize_serial_number.left_fill = left_fill
    if right_fill is None:
        right_fill = normalize_serial_number.right_fill
    else:
        normalize_serial_number.right_fill = right_fill
    if blank is None:
        blank = normalize_serial_number.blank
    else:
        normalize_serial_number.blank = blank
    if valid_chars is None:
        valid_chars = normalize_serial_number.valid_chars
    else:
        normalize_serial_number.valid_chars = valid_chars
    if invalid_chars is None:
        invalid_chars = normalize_serial_number.invalid_chars
    else:
        normalize_serial_number.invalid_chars = invalid_chars
    if strip_whitespace is None:
        strip_whitespace = normalize_serial_number.strip_whitespace
    else:
        normalize_serial_number.strip_whitespace = strip_whitespace
    if join is None:
        join = normalize_serial_number.join
    else:
        normalize_serial_number.join = join
    if na is None:
        na = normalize_serial_number.na
    else:
        normalize_serial_number.na = na

    if invalid_chars is None:
        invalid_chars = (c for c in charlist.ascii if c not in valid_chars)
    invalid_chars = ''.join(invalid_chars)
    sn = str(sn).strip(invalid_chars)
    if strip_whitespace:
        sn = sn.strip()
    if invalid_chars:
        if join:
            sn = sn.translate(None, invalid_chars)
        else:
            sn = multisplit(sn, invalid_chars)[-1]
    sn = sn[-max_length:]
    if strip_whitespace:
        sn = sn.strip()
    if na:
        if isinstance(na, (tuple, set, dict, list)) and sn in na:
            sn = ''
        elif na.match(sn):
            sn = ''
    if not sn and not (blank is False):
        return blank
    if left_fill:
        sn = left_fill * (max_length - len(sn)/len(left_fill)) + sn
    if right_fill:
        sn = sn + right_fill * (max_length - len(sn)/len(right_fill))
    return sn
normalize_serial_number.max_length=10
normalize_serial_number.left_fill='0'
normalize_serial_number.right_fill=''
normalize_serial_number.blank=''
normalize_serial_number.valid_chars=' -0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' 
normalize_serial_number.invalid_chars=None
normalize_serial_number.strip_whitespace=True
normalize_serial_number.join=False
normalize_serial_number.na=RE.nones
invalid_chars=None
strip_whitespace=True
join=False
na=RE.nones


normalize_account_number = normalize_serial_number


def multisplit(s, seps=list(string.punctuation) + list(string.whitespace), blank=True):
    r"""Just like str.split(), except that a variety (list) of seperators is allowed.
    
    >>> multisplit(r'1-2?3,;.4+-', string.punctuation)
    ['1', '2', '3', '', '', '4', '', '']
    >>> multisplit(r'1-2?3,;.4+-', string.punctuation, blank=False)
    ['1', '2', '3', '4']
    >>> multisplit(r'1C 234567890', '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n' + string.punctuation)
    ['1C 234567890']
    """
    seps = ''.join(seps)
    return [s2 for s2 in s.translate(''.join([(chr(i) if chr(i) not in seps else seps[0]) for i in range(256)])).split(seps[0]) if (blank or s2)]


def make_real(list_of_lists):
    for i, l in enumerate(list_of_lists):
        for j, val in enumerate(l):
            list_of_lists[i][j] = float(normalize_scientific_notation(str(val), ignore_commas=True))
    return list_of_lists


# def linear_correlation(x, y=None, ddof=0):
#     """Pierson linear correlation coefficient (-1 <= plcf <= +1)
#     >>> abs(linear_correlation(range(5), [1.2 * x + 3.4 for x in range(5)]) - 1.0) < 0.000001
#     True
#     # >>> abs(linear_correlation(sci.rand(2, 1000)))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
#     # 0.0...
#     """
#     if y is None:
#         if len(x) == 2:
#             y = x[1]
#             x = x[0]
#         elif len(x[0]) ==2:
#             y = [yi for xi, yi in x] 
#             x = [xi for xi, yi in x]
#         else:
#             mat = np.cov(x, ddof=ddof)
#             R = []
#             N = len(mat)
#             for i in range(N):
#                 R += [[1.] * N]
#                 for j in range(i+1,N):
#                     R[i][j] = mat[i,j]
#                     for k in range(N):
#                         R[i][j] /= (mat[k,k] ** 0.5)
#             return R
#     return np.cov(x, y, ddof=ddof)[1,0] / np.std(x, ddof=ddof) / np.std(y, ddof=ddof)


# def best_correlation_offset(x, y, ddof=0):
#     """Find the delay between x and y that maximizes the correlation between them
#     A negative delay means a negative-correlation between x and y was maximized
#     """
#     def offset_correlation(offset, x=x, y=y):
#         N = len(x)
#         if offset < 0:
#             y = [-1 * yi for yi in y]
#             offset = -1 * offset 
#         # TODO use interpolation to allow noninteger offsets
#         return linear_correlation([x[(i - int(offset)) % N] for i in range(N)], y)
#     return sci.minimize(offset_correlation, 0)


def imported_modules():
    for name, val in globals().iteritems():
        if isinstance(val, types.ModuleType):
            yield val


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
    >>> make_tz_aware([None, float('nan'), float('inf'), 1980, 1979.25*365.25, '1970-10-31', '1970-12-25', '1971-07-04'], 'CDT')  # doctest: +NORMALIZE_WHITESPACE
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
        return make_tz_aware(dateutil.parser.parse(dt))
    except:
        if not squelch:
            print("Failed to parse %r as a date" % dt)
    dt = [s.strip() for s in dt.split(' ')]
    # get rid of any " at " or empty strings
    dt = [s for s in dt if s and s.lower() != 'at']

    # deal with the absence of :'s in wikipedia datetime strings

    if RE.month_name.match(dt[0]) or RE.month_name.match(dt[1]):
        if len(dt) >= 5:
            dt = dt[:-2] + [dt[-2].strip(':') + ':' + dt[-1].strip(':')]
            return clean_wiki_datetime(' '.join(dt))
        elif len(dt) == 4 and (len(dt[3]) == 4 or len(dt[0]) == 4):
            dt[:-1] + ['00']
            return clean_wiki_datetime(' '.join(dt))
    elif RE.month_name.match(dt[-2]) or RE.month_name.match(dt[-3]):
        if len(dt) >= 5:
            dt = [dt[0].strip(':') + ':' + dt[1].strip(':')] + dt[2:]
            return clean_wiki_datetime(' '.join(dt))
        elif len(dt) == 4 and (len(dt[-1]) == 4 or len(dt[-3]) == 4):
            dt = [dt[0], '00'] + dt[1:]
            return clean_wiki_datetime(' '.join(dt))

    try:
        return make_tz_aware(dateutil.parser.parse(' '.join(dt)))
    except Exception as e:
        if squelch:
            from traceback import format_exc
            print format_exc(e) +  '\n^^^ Exception caught ^^^\nWARN: Failed to parse datetime string %r\n      from list of strings %r' % (' '.join(dt), dt)
            return dt
        raise(e)


def minmax_len_and_blackwhite_list(s, min_len=1, max_len=256, blacklist=None, whitelist=None, lower=False):
    if min_len > len(s) or len(s) > max_len:
        return False
    if lower:
        s = s.lower()
    if blacklist and s in blacklist:
        return False
    if whitelist and s not in whitelist:
        return False
    return True


def strip_HTML(s):
    """Simple, clumsy, slow HTML tag stripper"""
    result = ''
    total = 0
    for c in s:
        if c == '<':
            total = 1
        elif c == '>':
            total = 0
            result += ' '
        elif total == 0:
            result += c
    return result


def strip_edge_punc(s, punc=None, lower=None, str_type=str):
    if lower is None:
        lower = strip_edge_punc.lower
    if punc is None:
        punc = strip_edge_punc.punc
    if lower:
        s = s.lower()
    if not isinstance(s, basestring):
        return [strip_edge_punc(str_type(s0), punc) for s0 in s]
    return s.strip(punc)
strip_edge_punc.lower = False
strip_edge_punc.punc = PUNC


def get_sentences(s, regex=RE.sentence_sep):
    if isinstance(regex, basestring):
        regex = re.compile(regex)
    return [sent for sent in regex.split(s) if sent]


# this regex assumes "s' " is the end of a possessive word and not the end of an inner quotation, e.g. He said, "She called me 'Hoss'!"
def get_words(s, splitter_regex=RE.word_sep_except_external_appostrophe, 
              preprocessor=strip_HTML, postprocessor=strip_edge_punc, min_len=None, max_len=None, blacklist=None, whitelist=None, lower=False, filter_fun=None, str_type=str):
    r"""Segment words (tokens), returning a list of all tokens 

    Does not return any separating whitespace or punctuation marks.
    Attempts to return external apostrophes at the end of words.
    Comparable to `nltk.word_toeknize`.

    Arguments:
      splitter_regex (str or re): compiled or uncompiled regular expression
        Applied to the input string using `re.split()`
      preprocessor (function): defaults to a function that strips out all HTML tags
      postprocessor (function): a function to apply to each token before return it as an element in the word list
        Applied using the `map()` builtin
      min_len (int): delete all words shorter than this number of characters
      max_len (int): delete all words longer than this number of characters
      blacklist and whitelist (list of str): words to delete or preserve
      lower (bool): whether to convert all words to lowercase
      str_type (type): typically `str` or `unicode`, any type constructor that should can be applied to all words before returning the list

    Returns:
      list of str: list of tokens

    >>> get_words('He said, "She called me \'Hoss\'!". I didn\'t hear.')
    ['He', 'said', 'She', 'called', 'me', 'Hoss', 'I', "didn't", 'hear']
    >>> get_words('The foxes\' oh-so-tiny den was 2empty!')
    ['The', 'foxes', 'oh-so-tiny', 'den', 'was', '2empty']
    """
    # TODO: Get rid of `lower` kwarg (and make sure code that uses it doesn't break) 
    #       That and other simple postprocessors can be done outside of get_words
    postprocessor = postprocessor or str_type
    preprocessor = preprocessor or str_type
    if min_len is None:
        min_len = get_words.min_len
    if max_len is None:
        max_len = get_words.max_len
    blacklist = blacklist or get_words.blacklist
    whitelist = whitelist or get_words.whitelist
    filter_fun = filter_fun or get_words.filter_fun
    lower = lower or get_words.lower
    try:
        s = open(s, 'r')
    except:
        pass
    try:
        s = s.read()
    except:
        pass
    if not isinstance(s, basestring):
        try:
            # flatten the list of lists of words from each obj (file or string)
            return [word for obj in s for word in get_words(obj)]
        except:
            pass
    try:
        s = preprocessor(s)
    except:
        pass
    if isinstance(splitter_regex, basestring):
        splitter_regex = re.compile(splitter_regex)
    s = map(postprocessor, splitter_regex.split(s))
    s = map(str_type, s)
    if not filter_fun:
        return s
    return [word for word in s if filter_fun(word, min_len=min_len, max_len=max_len, blacklist=blacklist, whitelist=whitelist, lower=lower)]
get_words.blacklist = ('', None, '\'', '.', '_', '-')
get_words.whitelist = None
get_words.min_len = 1
get_words.max_len = 256
get_words.lower = False
get_words.filter_fun = minmax_len_and_blackwhite_list


def pluralize_field_name(names=None, retain_prefix=False):
    if not names:
        return ''
    elif isinstance(names, basestring):
        if retain_prefix:
            split_name = names
        else:
            split_name = names.split('__')[-1]
        if not split_name:
            return names
        elif 0 < len(split_name) < 4 or split_name.lower()[-4:] not in ('call', 'sale', 'turn'):
            return split_name
        else:
            return split_name + 's'
    else:
        return [pluralize_field_name(name) for name in names]
pluralize_field_names = pluralize_field_name


def tabulate(lol, headers, eol='\n'):
    """Use the pypi tabulate package instead!"""
    yield '| %s |' % ' | '.join(headers) + eol
    yield '| %s:|' % ':| '.join(['-'*len(w) for w in headers]) + eol
    for row in lol:
        yield '| %s |' % '  |  '.join(str(c) for c in row) + eol


def intify(obj, str_fun=str, use_ord=True, use_hash=True, use_len=True):
    """FIXME: this is unpythonic and does things you don't expect!

    FIXME: rename to "integer_from_category"

    Returns an integer representative of a categorical object (string, dict, etc)

    >>> intify('1.2345e10')
    12345000000
    >>> intify([12]), intify('[99]'), intify('(12,)')
    (91, 91, 40)
    >>> intify('A'), intify('a'), intify('AAA'), intify('B'), intify('BB')
    (97, 97, 97, 98, 98)
    >>> intify(272)
    272
    >>> intify(float('nan'), use_ord=False, use_hash=False, str_fun=None)
    >>> intify(float('nan'), use_ord=False, use_hash=False, use_len=False)
    >>> intify(float('nan')), intify('n'), intify(None)
    (110, 110, 110)
    >>> intify(None, use_ord=False, use_hash=False, use_len=False)
    >>> intify(None, use_ord=False, use_hash=False, str_fun=False)
    >>> intify(None, use_hash=False, str_fun=False) 
    """
    try:
        return int(obj)
    except:
        pass
    try:
        float_obj = float(obj)
        if float('-inf') < float_obj < float('inf'):
            # WARN: This will increment sys.maxint by +1 and decrement sys.maxint by -1!!!!
            #       But hopefully these cases will be dealt with as expected, above
            return int(float_obj)
    except:
        pass
    if not str_fun:
        str_fun = lambda x:x
    if use_ord:    
        try:
            return ord(str_fun(obj)[0].lower())
        except:
            pass
    if use_hash:  
        try:
            return hash(str_fun(obj))
        except:
            pass
    if use_len:
        try:
            return len(obj)
        except:
            pass
        try:
            return len(str_fun(obj))
        except:
            pass
    return None



def listify(values, N=1, delim=None):
    """Return an N-length list, with elements values, extrapolating as necessary.

    >>> listify("don't split into characters")
    ["don't split into characters"]
    >>> listify("len = 3", 3)
    ['len = 3', 'len = 3', 'len = 3']
    >>> listify("But split on a delimeter, if requested.", delim=',')
    ['But split on a delimeter', ' if requested.']
    >>> listify(["obj 1", "obj 2", "len = 4"], N=4)
    ['obj 1', 'obj 2', 'len = 4', 'len = 4']
    >>> listify(iter("len=7"), N=7)
    ['l', 'e', 'n', '=', '7', '7', '7']
    >>> listify(iter("len=5"))
    ['l', 'e', 'n', '=', '5']
    >>> listify(None, 3)
    [[], [], []]
    >>> listify([None],3)
    [None, None, None]
    >>> listify([], 3)
    [[], [], []]
    >>> listify('', 2)
    ['', '']
    >>> listify(0)
    [0]
    >>> listify(False, 2)
    [False, False]
    """
    ans = [] if values is None else values

    # convert non-string non-list iterables into a list
    if hasattr(ans, '__iter__') and not isinstance(ans, basestring):
        ans = list(ans)
    else:
        # split the string (if possible)
        if isinstance(delim, basestring) and isinstance(ans, basestring):
            try:
                ans = ans.split(delim)
            except:
                ans = [ans]
        else:
            ans = [ans]

    # pad the end of the list if a length has been specified
    if len(ans):
        if len(ans) < N and N > 1:
            ans += [ans[-1]] * (N - len(ans))
    else:
        if N > 1:
            ans = [[]] * N

    return ans


def tuplify(values, N=1, delim=None):
    return tuple(listify(values, N=N, delim=delim))


def unlistify(l, depth=1, typ=list, get=None):
    """Return the desired element in a list ignoring the rest.

    >>> unlistify([1,2,3])
    1
    >>> unlistify([1,[4, 5, 6],3], get=1)
    [4, 5, 6]
    >>> unlistify([1,[4, 5, 6],3], depth=2, get=1)
    5
    >>> unlistify([1,(4, 5, 6),3], depth=2, get=1)
    (4, 5, 6)
    >>> unlistify([1,2,(4, 5, 6)], depth=2, get=2)
    (4, 5, 6)
    >>> unlistify([1,2,(4, 5, 6)], depth=2, typ=(list, tuple), get=2)
    6
    """
    i = 0
    if depth is None:
        depth = 1
    index_desired = get or 0
    while i < depth and isinstance(l, typ):
        if len(l):
            if len(l) > index_desired:
                l = l[index_desired]
                i += 1
        else:
            return l
    return l


def is_ignorable_str(s, ignorable_strings=(), lower=True, filename=True, startswith=True):
    ignorable_strings = listify(ignorable_strings)
    if not (lower or filename or startswith):
        return s in ignorable_strings
    for ignorable in ignorable_strings:
        if lower:
            ignorable = ignorable.lower()
            s = s.lower()
        if filename:
            s = s.split(os.path.sep)[-1]
        if startswith and s.startswith(ignorable):
            return True
        elif s == ignorable:
            return True


def strip_keys(d, nones=False, depth=0):
    r"""Strip whitespace from all dictionary keys, to the depth indicated

    >>> strip_keys({' a': ' a', ' b\t c ': {'d e  ': 'd e  '}}) == {'a': ' a', 'b\t c': {'d e  ': 'd e  '}}
    True
    >>> strip_keys({' a': ' a', ' b\t c ': {'d e  ': 'd e  '}}, depth=100) == {'a': ' a', 'b\t c': {'d e': 'd e  '}}
    True
    """
    ans = type(d)((str(k).strip(), v) for (k, v) in OrderedDict(d).iteritems() if (not nones or (str(k).strip() and str(k).strip() != 'None')))
    if int(depth) < 1:
        return ans
    if int(depth) > strip_keys.MAX_DEPTH:
        warnings.warn(RuntimeWarning("Maximum recursion depth allowance (%r) exceeded." % strip_keys.MAX_DEPTH))
    for k, v in ans.iteritems():
        if isinstance(v, collections.Mapping):
            ans[k] = strip_keys(v, nones=nones, depth=int(depth)-1)
    return ans
strip_keys.MAX_DEPTH = 1e6


def str_from_table(table, sep='\t', eol='\n', max_rows=100000000, max_cols=1000000):
    max_rows = min(max_rows, len(table))
    return eol.join([sep.join(list(str(field) for field in row[:max_cols])) for row in table[:max_rows]])


def get_table_from_csv(filename='ssg_report_aarons_returns.csv', delimiter=',', dos=False):
    """Dictionary of sequences from CSV file"""
    table = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f, dialect='excel', delimiter=delimiter)
        for row in reader:
            table += [row]
    if not dos:
        return table
    return dos_from_table(table)


def save_sheet(table, filename, ext='tsv', verbosity=0):
    if ext.lower() == 'tsv':
        sep = '\t'
    else:
        sep = ','
    s = str_from_table(table, sep=sep)
    if verbosity > 2:
        print s
    if verbosity > 0:
        print 'Saving ' + filename + '.' + ext
    with open(filename + '.' + ext, 'w') as fpout:
        fpout.write(s)


def save_sheets(tables, filename, ext='.tsv', verbosity=0):
    for i, table in enumerate(tables):
        save_sheet(table, filename + '_Sheet%d' % i, ext=ext, verbosity=verbosity)


def shorten(s, max_len=16):
    """Attempt to shorten a phrase by deleting words at the end of the phrase

    >>> shorten('Hello World!')
    'Hello World'
    >>> shorten("Hello World! I'll talk your ear off!", 15)
    'Hello World'
    """
    short = s
    words = [abbreviate(word) for word in get_words(s)]
    for i in xrange(len(words), 0, -1):
        short = ' '.join(words[:i])
        if len(short) <= max_len:
            break
    return short[:max_len]


def abbreviate(s):
    """Some basic abbreviations

    TODO: load a large dictionary of abbreviations from NLTK, etc
    """
    return abbreviate.words.get(s, s)
abbreviate.words = {'account': 'acct', 'number': 'num', 'customer': 'cust', 'member': 'membr', 'building': 'bldg', 'serial number': 'SN', 'social security number': 'SSN'}


def remove_internal_vowels(s, space=''):
    # because this pattern overlaps for vowels separated by a single or no consonant, it must be run several times
    internal_vowel = re.compile(r'([A-Za-z])[aeiou]([A-Za-z])')
    strlen = len(s)
    while True:
        s = internal_vowel.sub(r'\1\2', s)
        if len(s) < strlen:
            strlen = len(s)
        else:
            break
    return re.sub(r'\s', space, s)


def normalize_year(y):
    y = RE.not_digit_list.sub('', str(y))
    try:
        y = int(y)
    except:
        y = None
    if 0 <= y < 70:
        y += 2000
    elif 70 <= y < 100:
        y += 1900
    return y


def generate_kmers(seq, k=4):
    """Return a generator of all the unique substrings (k-mer or q-gram strings) within a sequence/string

    Not effiicent for large k and long strings.
    Doesn't form substrings that are shorter than k, only exactly k-mers

    Used for algorithms like UniqTag for genome unique identifier locality sensitive hashing.

    jellyfish is a C implementation of k-mer counting

    If seq is a string generate a sequence of k-mer string
    If seq is a sequence of strings then generate a sequence of generators or sequences of k-mer strings
    If seq is a sequence of sequences of strings generate a sequence of sequence of generators ...

    Default k = 4 because that's the length of a gene base-pair?

    >>> ' '.join(generate_kmers('AGATAGATAGACACAGAAATGGGACCACAC'))
    'AGAT GATA ATAG TAGA AGAT GATA ATAG TAGA AGAC GACA ACAC CACA ACAG CAGA AGAA GAAA AAAT AATG ATGG TGGG GGGA GGAC GACC ACCA CCAC CACA ACAC'
    """
    if isinstance(seq, basestring):
        for i in range(len(seq) - k + 1):
           yield seq[i:i+k]
    elif isinstance(seq, (int, float, Decimal)):
        for s in generate_kmers(str(seq)):
            yield s
    else:
        for s in seq:
            yield generate_kmers(s, k)


def datetime_histogram(seq):
    """Plot a histogram of datetimes from a sequence (list, tuple, iterator) of date or datetimes"""
    raise NotImplementedError()


def kmer_tuple(seq, k=4):
    """Return a tuple all the unique substrings (k-mer or q-gram strings) within a sequence/string

    Not effiicent for large k and long strings.
    Doesn't form substrings that are shorter than k, only exactly k-mers

    Used for algorithms like UniqTag for genome unique identifier locality sensitive hashing.

    jellyfish is a C implementation of k-mer counting

    If seq is a string generate a sequence of k-mer string
    If seq is a sequence of strings then generate a sequence of generators or sequences of k-mer strings
    If seq is a sequence of sequences of strings generate a sequence of sequence of generators ...

    Default k = 4 because that's the length of a gene base-pair?

    Examples:
        # >>> kmer_tuple(['AGATAGATAG', 'ACACAGAAAT', 'GGGACCACAC'], k=4)
        # (('AGAT', 'GATA', 'ATAG', 'TAGA', 'AGAT', 'GATA', 'ATAG'),
        #  ('ACAC', 'CACA', 'ACAG', 'CAGA', 'AGAA', 'GAAA', 'AAAT'),
        #  ('GGGA', 'GGAC', 'GACC', 'ACCA', 'CCAC', 'CACA', 'ACAC'))
        >>> ' '.join(kmer_tuple('AGATAGATAGACACAGAAATGGGACCACAC'))
        'AAAT AATG ACAC ACAC ACAG ACCA AGAA AGAC AGAT AGAT ATAG ATAG ATGG CACA CACA CAGA CCAC GAAA GACA GACC GATA GATA GGAC GGGA TAGA TAGA TGGG'
    """
    return tuple(sorted(generate_kmers(seq, k=k)))


def kmer_counter(seq, k=4):
    """Return a sequence of all the unique substrings (k-mer or q-gram) within a short (<128 symbol) string

    Used for algorithms like UniqTag for genome unique identifier locality sensitive hashing.

    jellyfish is a C implementation of k-mer counting

    If seq is a string generate a sequence of k-mer string
    If seq is a sequence of strings then generate a sequence of generators or sequences of k-mer strings
    If seq is a sequence of sequences of strings generate a sequence of sequence of generators ...

    Default k = 4 because that's the length of a gene base-pair?

    >>> kmer_counter('AGATAGATAGACACAGAAATGGGACCACAC') == collections.Counter({'ACAC': 2, 'ATAG': 2, 'CACA': 2, 'TAGA': 2, 'AGAT': 2, 'GATA': 2, 'AGAC': 1, 'ACAG': 1, 'AGAA': 1, 'AAAT': 1, 'TGGG': 1, 'ATGG': 1, 'ACCA': 1, 'GGAC': 1, 'CCAC': 1, 'CAGA': 1, 'GAAA': 1, 'GGGA': 1, 'GACA': 1, 'GACC': 1, 'AATG': 1})
    True
    """
    if isinstance(seq, basestring):
        return collections.Counter(generate_kmers(seq, k))


def kmer_set(seq, k=4):
    """Return the set of unique k-length substrings within a the sequence/string `seq`

    Implements formula:
    C_k(s) = C(s) ∩ Σ^k 
    from http://biorxiv.org/content/early/2014/08/01/007583

    >>> sorted(kmer_set('AGATAGATAGACACAGAAATGGGACCACAC'))
    ['AAAT', 'AATG', 'ACAC', 'ACAG', 'ACCA', 'AGAA', 'AGAC', 'AGAT', 'ATAG', 'ATGG', 'CACA', 'CAGA', 'CCAC', 'GAAA', 'GACA', 'GACC', 'GATA', 'GGAC', 'GGGA', 'TAGA', 'TGGG']
    """
    if isinstance(seq, basestring):
        return set(generate_kmers(seq, k))


# def kmer_frequency(seq_of_seq, km=None):
#     """Count the number of sequences in seq_of_seq that contain a given kmer `km`

#     From http://biorxiv.org/content/early/2014/08/01/007583, implements the formula:
#     f(t, S) = |{s | t ∈ C^k(s) ∧ s ∈ S}|
#     where:
#     t = km
#     S = seq_of_seq
#     >>> kmer_frequency(['AGATAGATAG', 'ACACAGAAAT', 'GGGACCACAC'], km=4)
    
#     """
#     if km and isinstance(km, basestring):
#         return sum(km in counter for counter in kmer_counter(seq_of_seq, len(km)))
#     km = int(km)
#     counter = collections.Counter()
#     counter += collections.Counter(tuple(sorted(set(kmer_counter(seq, km)))) for seq in seq_of_seq)
#     return counter


# def uniq_tag(seq, k=4, other_strings=None):
#     """Hash that is the same for similar strings and can serve as an abbreviation for a string

#     Based on UniqTag:
#     http://biorxiv.org/content/early/2014/08/01/007583
#     Which was inspired by MinHasH:
#     http://en.wikipedia.org/wiki/MinHash

#     t_u = min arg min t ∈ C k(s) f(t, S)
#     uk(s, S) = min (arg_min((t ∈ C^k(s)), f(t, S))

#     uk(s, S) = "the UniqTag, the lexicographically minimal k-mer of those k-mers of s that are least frequent in S."

#     the "k-mers of s" can be found with kmer_set()
#     the frequencies of those k-mers in other_stirngs, S, should be provided by kmer_frequency(other_strings, km) for km in kmer_set(s)

#     >>> uniq_tag('Hello World')

#     """
#     # FIXME: UNTESTED!
#     if not other_strings:
#         if isinstance(seq, basestring):
#             other_strings = (seq,)
#         else:
#             other_strings = tuple(seq)
#         return uniq_tag(other_strings[0], other_strings)
#     other_strings = set(other_strings)
#     if isinstance(seq, basestring):
#         kms = kmer_set(seq)
#         km_frequencies = ((sum(km in kmer_set(s, k), s) for s in other_strings) for km in kms)
#         print min(km_frequencies)
#         return min(km_frequencies)[1]
#     return tuple(uniq_tag(s, other_strings) for s in seq)


def count_duplicates(items):
    """Return a dict of objects and thier counts (like a Counter), but only count > 1"""
    c = collections.Counter(items)
    return dict((k, v) for (k,v) in c.iteritems() if v > 1)



# def markdown_stats(doc):
#     """Compute statistics about the string or document provided.

#     Returns:
#         dict: e.g. {'pages': 24, 'words': 1234, 'vocabulary': 123, 'reaading level': 3, ...}
#     """
#     sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
#     sentences = sentence_detector.tokenize(doc)
#     tokens = nltk.tokenize.punkt.PunktWordTokenizer().tokenize(doc)
#     vocabulary = collections.Counter(tokens)
    
#     return collections.OrderedDict([
#         ('lines', sum([bool(l.strip().strip('-').strip()) for l in doc.split('\n')])),
#         ('pages', sum([bool(l.strip().startswith('---')) for l in doc.split('\n')]) + 1),
#         ('tokens', len(tokens)),
#         ('sentences', len(sentences)),
#         ('vocabulary', len(vocabulary.keys())),
#         ])


def slug_from_dict(d, max_len=128, delim='-'):
    """Produce a slug (short URI-friendly string) from an iterable Mapping (dict, OrderedDict)

    >>> slug_from_dict({'a': 1, 'b': 'beta', ' ': 'alpha'})
    '1-alpha-beta'
    """
    return slug_from_iter(d.values(), max_len=max_len, delim=delim)


def slug_from_iter(it, max_len=128, delim='-'):
    """Produce a slug (short URI-friendly string) from an iterable (list, tuple, dict)

    >>> slug_from_iter(['.a.', '=b=', '--alpha--'])
    'a-b-alpha'
    """

    nonnull_values = [str(v) for v in it if v or ((isinstance(v, (long, int, float, Decimal)) and str(v)))]
    return slugify(delim.join(shorten(v, max_len=int(float(max_len) / len(nonnull_values))) for v in nonnull_values), word_boundary=True)


def tfidf(corpus):
    """Compute a TFIDF matrix (Term Frequency and Inverse Document Freuqency matrix)"""
    raise NotImplementedError("Google TFIDF for canonical implementations")


def shakeness(doc):
    """Determine how similar a document's vocabulary is to Shakespeare's"""
    raise NotImplementedError("Import a Shakespear corpus and compare the distribution of words there to the ones in the sample doc (vocabulary similarity)")


def slash_product(string_or_seq, slash='/', space=' '):
    """Return a list of all possible meanings of a phrase containing slashes

    TODO:
        - Code is not in standard Sedgewick recursion form
        - Simplify by removing one of the recursive calls?
        - Simplify by using a list comprehension?

    >>> slash_product("The challenging/confusing interview didn't end with success/offer")  # doctest: +NORMALIZE_WHITESPACE
    ["The challenging interview didn't end with success",
     "The challenging interview didn't end with offer",
     "The confusing interview didn't end with success",
     "The confusing interview didn't end with offer"]
    >>> slash_product('I say goodbye/hello cruel/fun world.')  # doctest: +NORMALIZE_WHITESPACE
    ['I say goodbye cruel world.',
     'I say goodbye fun world.',
     'I say hello cruel world.',
     'I say hello fun world.']
    >>> slash_product('I say goodbye/hello/bonjour cruelness/fun/world')  # doctest: +NORMALIZE_WHITESPACE
    ['I say goodbye cruelness',
     'I say goodbye fun',
     'I say goodbye world',
     'I say hello cruelness',
     'I say hello fun',
     'I say hello world',
     'I say bonjour cruelness',
     'I say bonjour fun',
     'I say bonjour world']
    """
    # Terminating case is a sequence of strings without any slashes
    if not isinstance(string_or_seq, basestring):
        # If it's not a string and has no slashes, we're done
        if not any(slash in s for s in string_or_seq):
            return list(string_or_seq)
        ans = []
        for s in string_or_seq:
            # slash_product of a string will always return a flat list
            ans += slash_product(s)
        return slash_product(ans)
    # Another terminating case is a single string without any slashes
    if not slash in string_or_seq:
        return [string_or_seq]
    # The third case is a string with some slashes in it
    i = string_or_seq.index(slash)
    head, tail = string_or_seq[:i].split(space), string_or_seq[i+1:].split(space)
    alternatives = head[-1], tail[0]
    head, tail = space.join(head[:-1]), space.join(tail[1:])
    return slash_product([space.join([head, word, tail]).strip(space) for word in alternatives])


def is_valid_american_date_string(s, require_year=True):
    if not isinstance(s, basestring):
        return False
    if require_year and len(s.split('/')) != 3:
        return False
    return bool(1 <= int(s.split('/')[0]) <= 12 and 1 <= int(s.split('/')[1]) <= 31)


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
    if (isinstance(dt, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp, np.datetime64))
            or dt in (float('nan'), float('inf'), float('-inf'), None, '')):
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
            print 'Unable to parse {}'.format(repr(dt))
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
        return datetime.datetime(*(dt[:resolution] + [1] * max(3 - resolution , 0)))

    if isinstance(dt, tuple) and len(dt) <= 9 and all(isinstance(val, (float, int)) for val in dt):
        dt = list(dt) + [0] * (max(6 - len(dt), 0))
        # if the 6th element of the tuple looks like a float set of seconds need to add microseconds
        if len(dt) == 6 and isinstance(dt[5], float):
                dt = list(dt) + [1000000 * (dt[5] - int(dt[5]))]
                dt[5] = int(dt[5])
        dt = tuple(int(val) for val in dt)
        return datetime.datetime(*(dt[:resolution] + [1] * max(resolution - 3, 0)))

    return [quantize_datetime(value) for value in dt]


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

def flatten_dataframe(df, date_parser=parse_date, verbosity=0):
    """Creates 1-D timeseries (pandas.Series) coercing column labels into datetime.time objects

    Assumes that the columns are strings representing times of day (or datetime.time objects)
    Assumes that the index should be a datetime object. If it isn't already, the first column
    with "date" (case insenstive) in its label will be used as the FataFrame index.
    """

    # extract rows with nonull, nonnan index values
    df = df[pd.notnull(df.index)]

    # Make sure columns and row labels are all times and dates respectively
    # Ignores/clears any timezone information 
    if all(isinstance(i, int) for i in df.index):
        for label in df.columns:
            if 'date' in str(label).lower():
                df.index = [make_date(d) for d in df[label]]
                del df[label]
                break
    if not all(isinstance(i, pd.Timestamp) for i in df.index):
        date_index = []
        for i in df.index:
            try:
                date_index += [make_date(str(i))]
            except:
                date_index += [i]
        df.index = date_index
    df.columns = [make_time(str(c)) if (c and str(c) and str(c)[0] in '0123456789') else str(c) for c in df.columns]
    if verbosity > 2:
        print 'Columns: {0}'.format(df.columns)

    # flatten it
    df = df.transpose().unstack()

    df = df.drop(df.index[[(isinstance(d[1], (basestring, NoneType))) for d in df.index]])

    # df.index is now a compound key (tuple) of the column labels (df.columns) and the row labels (df.index) 
    # so lets combine them to be datetime values (pandas.Timestamp)
    dt = None
    t0 = df.index[0][1]
    t1 = df.index[1][1]
    try:
        dt_stepsize = datetime.timedelta(hours=t1.hour - t0.hour, minutes=t1.minute - t0.minute, seconds=t1.second - t0.second)
    except:
        dt_stepsize = datetime.timedelta(hours=0, minutes=15)
    parse_date_exception = False
    index = []
    for i, d in enumerate(df.index.values):
        dt = i
        if verbosity > 2:
            print d
        # # TODO: assert(not parser_date_exception)
        # if isinstance(d[0], basestring):
        #     d[0] = d[0]
        try:
            datetimeargs = list(d[0].timetuple()[:3]) + [d[1].hour, d[1].minute, d[1].second, d[1].microsecond]
            dt = datetime.datetime(*datetimeargs)
            if verbosity > 2:
                print '{0} -> {1}'.format(d, dt)
        except TypeError:
            if verbosity > 1:
                print_exc()
                # print 'file with error: {0}\ndate-time tuple that caused the problem: {1}'.format(file_properties, d)
            if isinstance(dt, datetime.datetime):
                if dt:
                    dt += dt_stepsize
                else:
                    dt = i
                    parse_date_exception = True
                    # dt = str(d[0]) + ' ' + str(d[1])
                    # parse_date_exception = True
            else:
                dt = i
                parse_date_exception = True
        except:
            if verbosity > 0:
                print_exc()
                # print 'file with error: {0}\ndate-time tuple that caused the problem: {1}'.format(file_properties, d)
            dt = i
        index += [dt]

    if index and not parse_date_exception:
        df.index = index
    else:
        df.index = list(pd.Timestamp(d) for d in index)
    return df


def dataframe_from_excel(path, sheetname=0, header=0, skiprows=None):  # , parse_dates=False):
    """Thin wrapper for pandas.io.excel.read_excel() that accepts a file path and sheet index/name

    Arguments:
      path (str): file or folder to retrieve CSV files and `pandas.DataFrame`s from
      ext (str): file name extension (to filter files by)
      date_parser (function): if the MultiIndex can be interpretted as a datetime, this parser will be used

    Returns:
      dict of DataFrame: { file_path: flattened_data_frame }
    """
    sheetname = sheetname or 0
    if isinstance(sheetname, (basestring, float)):
        try:
            sheetname = int(sheetname)
        except (TypeError, ValueError, OverflowError):
            sheetname = str(sheetname)
    wb = xlrd.open_workbook(path)
    # if isinstance(sheetname, int):
    #     sheet = wb.sheet_by_index(sheetname)
    # else:
    #     sheet = wb.sheet_by_name(sheetname)
    # assert(not parse_dates, "`parse_dates` argument and function not yet implemented!")
    # table = [sheet.row_values(i) for i in range(sheet.nrows)]
    return pd.io.excel.read_excel(wb, sheetname=sheetname, header=header, skiprows=skiprows, engine='xlrd')


def flatten_excel(path='.', ext='xlsx', sheetname=0, skiprows=None, header=0, date_parser=parse_date, verbosity=0, output_ext=None):
    """Load all Excel files in the given path, write .flat.csv files, return `DataFrame` dict

    Arguments:
      path (str): file or folder to retrieve CSV files and `pandas.DataFrame`s from
      ext (str): file name extension (to filter files by)
      date_parser (function): if the MultiIndex can be interpretted as a datetime, this parser will be used

    Returns:
      dict of DataFrame: { file_path: flattened_data_frame }
    """

    date_parser = date_parser or (lambda x: x)
    dotted_ext, dotted_output_ext = None, None
    if ext != None and output_ext != None:
        dotted_ext = ('' if ext.startswith('.') else '.') + ext
        dotted_output_ext = ('' if output_ext.startswith('.') else '.') + output_ext
    table = {}
    for file_properties in find_files(path, ext=ext or '', verbosity=verbosity):
        file_path = file_properties['path']
        if output_ext and (dotted_output_ext + '.') in file_path:
            continue
        df = dataframe_from_excel(file_path, sheetname=sheetname, header=header, skiprows=skiprows)
        df = flatten_dataframe(df, verbosity=verbosity)
        if dotted_ext != None and dotted_output_ext != None:
            df.to_csv(file_path[:-len(dotted_ext)] + dotted_output_ext + dotted_ext)
    return table


def walk_level(path, level=1):
    """Like os.walk, but takes `level` kwarg that indicates how deep the recursion will go.

    Notes:
      TODO: refactor `level`->`depth`

    References:
      http://stackoverflow.com/a/234329/623735

    Args:
     path (str):  Root path to begin file tree traversal (walk)
      level (int, optional): Depth of file tree to halt recursion at. 
        None = full recursion to as deep as it goes
        0 = nonrecursive, just provide a list of files at the root level of the tree
        1 = one level of depth deeper in the tree

    Examples:
      >>> root = os.path.dirname(__file__)
      >>> all((os.path.join(base,d).count('/')==(root.count('/')+1)) for (base, dirs, files) in walk_level(root, level=0) for d in dirs)
      True
    """
    if isinstance(level, NoneType):
        level = float('inf')
    path = path.rstrip(os.path.sep)
    if os.path.isdir(path):
        root_level = path.count(os.path.sep)
        for root, dirs, files in os.walk(path):
            yield root, dirs, files
            if root.count(os.path.sep) >= root_level + level:
                del dirs[:]
    elif os.path.isfile(path):
        yield os.path.dirname(path), [], [os.path.basename(path)]
    else:
        raise RuntimeError("Can't find a valid folder or file for path {0}".format(repr(path)))


def path_status(path, filename='', status=None, verbosity=0):
    """ Retrieve the access, modify, and create timetags for a path along with its size

    Arguments:
        path (str): full path to the file or directory to be statused
        status (dict): optional existing status to be updated/overwritten with new status values

    Returns:
        dict: {'size': bytes (int), 'accessed': (datetime), 'modified': (datetime), 'created': (datetime)}
    """
    status = status or {}
    if not filename:
        dir_path, filename = os.path.split()  # this will split off a dir and as `filename` if path doesn't end in a /
    else:
        dir_path = path
    full_path = os.path.join(dir_path, filename)
    if verbosity > 1:
        print(full_path)
    status['name'] = filename
    status['path'] = full_path
    status['dir']  = dir_path
    status['type'] = []
    try:
        status['size']     = os.path.getsize(full_path)
        status['accessed'] = datetime.datetime.fromtimestamp(os.path.getatime(full_path))
        status['modified'] = datetime.datetime.fromtimestamp(os.path.getmtime(full_path))
        status['created']  = datetime.datetime.fromtimestamp(os.path.getctime(full_path))
        status['mode'] = os.stat(full_path).st_mode   # first 3 digits are User, Group, Other permissions: 1=execute,2=write,4=read
        if os.path.ismount(full_path):
            status['type'] += ['mount-point']
        elif os.path.islink(full_path):
            status['type'] += ['symlink']
        if os.path.isfile(full_path):
            status['type'] += ['file']
        elif os.path.isdir(full_path):
            status['type'] += ['dir']
        if not status['type']:
            if stat.S_ISSOCK(status['mode']):
                status['type'] += ['socket']
            elif stat.S_ISCHR(status['mode']):
                status['type'] += ['special']
            elif stat.S_ISBLK(status['mode']):
                status['type'] += ['block-device']
            elif stat.S_ISFIFO(status['mode']):
                status['type'] += ['pipe']
        if not status['type']:
            status['type'] += ['unknown']
        elif status['type'] and status['type'][-1] == 'symlink':
            status['type'] += ['broken']
    except OSError:
        status['type'] = ['nonexistent'] + status['type']
        if verbosity > -1:
            warnings.warn("Unable to stat path '{}'".format(full_path))
    status['type'] = '->'.join(status['type'])

    return status


def find_files(path='', ext='', level=None, typ=list, dirs=False, files=True, verbosity=0):
    """ Recursively find all files in the indicated directory

    Filter by the indicated file name extension (ext)

    Args:
      path (str):  Root/base path to search.
      ext (str):   File name extension. Only file paths that ".endswith()" this string will be returned
      level (int, optional): Depth of file tree to halt recursion at. 
        None = full recursion to as deep as it goes
        0 = nonrecursive, just provide a list of files at the root level of the tree
        1 = one level of depth deeper in the tree
      typ (type):  output type (default: list). if a mapping type is provided the keys will be the full paths (unique)
      dirs (bool):  Whether to yield dir paths along with file paths (default: False)
      files (bool): Whether to yield file paths (default: True)
        `dirs=True`, `files=False` is equivalent to `ls -d`

    Returns: 
      list of dicts: dict keys are { 'path', 'name', 'bytes', 'created', 'modified', 'accessed', 'permissions' }
        path (str): Full, absolute paths to file beneath the indicated directory and ending with `ext`
        name (str): File name only (everythin after the last slash in the path)
        size (int): File size in bytes
        created (datetime): File creation timestamp from file system
        modified (datetime): File modification timestamp from file system
        accessed (datetime): File access timestamp from file system
        permissions (int): File permissions bytes as a chown-style integer with a maximum of 4 digits 
        type (str): One of 'file', 'dir', 'symlink->file', 'symlink->dir', 'symlink->broken'
          e.g.: 777 or 1755

    Examples:
      >>> 'util.py' in [d['name'] for d in find_files(os.path.dirname(__file__), ext='.py', level=0)]
      True
      >>> (d for d in find_files(os.path.dirname(__file__), ext='.py') if d['name'] == 'util.py').next()['size'] > 1000
      True

      There should be an __init__ file in the same directory as this script.
      And it should be at the top of the list.
      >>> sorted(d['name'] for d in generate_files(os.path.dirname(__file__), ext='.py', level=0))[0]
      '__init__.py'
      >>> sorted(generate_files().next().keys())
      ['accessed', 'created', 'dir', 'mode', 'modified', 'name', 'path', 'size', 'type']
      >>> all(d['type'] in ('file','dir','symlink->file','symlink->dir','mount-point->file','mount-point->dir','block-device','symlink->broken','pipe','special','socket','unknown') for d in generate_files(level=1, files=True, dirs=True))
      True
    """
    gen = generate_files(path, ext=ext, level=level, dirs=dirs, files=files, verbosity=verbosity)
    if isinstance(typ(), collections.Mapping):
        return typ((ff['path'], ff) for ff in gen)
    elif typ is not None:
        return typ(gen)
    else:
        return gen


def generate_files(path='', ext='', level=None, dirs=False, files=True, verbosity=0):
    """ Recursively generate files (and thier stats) in the indicated directory 
    
    Filter by the indicated file name extension (ext)

    Args:
      path (str):  Root/base path to search.
      ext (str):   File name extension. Only file paths that ".endswith()" this string will be returned
      level (int, optional): Depth of file tree to halt recursion at. 
        None = full recursion to as deep as it goes
        0 = nonrecursive, just provide a list of files at the root level of the tree
        1 = one level of depth deeper in the tree
      typ (type):  output type (default: list). if a mapping type is provided the keys will be the full paths (unique)
      dirs (bool):  Whether to yield dir paths along with file paths (default: False)
      files (bool): Whether to yield file paths (default: True)
        `dirs=True`, `files=False` is equivalent to `ls -d`

    Returns: 
      list of dicts: dict keys are { 'path', 'name', 'bytes', 'created', 'modified', 'accessed', 'permissions' }
        path (str): Full, absolute paths to file beneath the indicated directory and ending with `ext`
        name (str): File name only (everythin after the last slash in the path)
        size (int): File size in bytes
        created (datetime): File creation timestamp from file system
        modified (datetime): File modification timestamp from file system
        accessed (datetime): File access timestamp from file system
        permissions (int): File permissions bytes as a chown-style integer with a maximum of 4 digits 
        type (str): One of 'file', 'dir', 'symlink->file', 'symlink->dir', 'symlink->broken'
          e.g.: 777 or 1755

    Examples:
      >>> 'util.py' in [d['name'] for d in find_files(os.path.dirname(__file__), ext='.py', level=0)]
      True
      >>> (d for d in find_files(os.path.dirname(__file__), ext='.py') if d['name'] == 'util.py').next()['size'] > 1000
      True

      There should be an __init__ file in the same directory as this script.
      And it should be at the top of the list.
      >>> sorted(d['name'] for d in find_files(os.path.dirname(__file__), ext='.py', level=0))[0]
      '__init__.py'
      >>> os.path.join(os.path.dirname(__file__), '__init__.py') in find_files(
      ... os.path.dirname(__file__), ext='.py', level=0, typ=dict)
      True
      >>> sorted(find_files()[0].keys())
      ['accessed', 'created', 'dir', 'mode', 'modified', 'name', 'path', 'size', 'type']
      >>> all(d['type'] in ('file','dir','symlink->file','symlink->dir','mount-point->file','mount-point->dir','block-device','symlink->broken','pipe','special','socket','unknown')
      ... for d in find_files(level=1, files=True, dirs=True))
      True
    """
    path = path or './'
    ext = str(ext).lower()

    for dir_path, dir_names, filenames in walk_level(path, level=level):
        if verbosity > 0:
            print('Checking path "{}"'.format(dir_path))
        if files:
            for fn in filenames:  # itertools.chain(filenames, dir_names)
                if ext and not fn.lower().endswith(ext):
                    continue
                yield path_status(dir_path, fn, verbosity=verbosity)
        if dirs:
            # TODO: warn user if ext and dirs both set
            for fn in dir_names:
                if ext and not fn.lower().endswith(ext):
                    continue
                yield path_status(dir_path, fn, verbosity=verbosity)

    # if verbosity > 1:
    #     print files_found
    # return files_found


def find_dirs(*args, **kwargs):
    kwargs['files'] = kwargs.get('files', False)
    kwargs.update({'dirs': True})
    return find_files(*args, **kwargs)

