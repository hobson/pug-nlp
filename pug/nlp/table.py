#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""table utils"""
from __future__ import division, print_function, absolute_import
from builtins import str  # , unicode  # noqa
# from future.utils import viewitems  # noqa
from past.builtins import basestring
# try:  # python 3.5+
#     from io import StringIO
#     from ConfigParser import ConfigParser
#     from itertools import izip as zip
# except:
#     from StringIO import StringIO
#     from configparser import ConfigParser

from types import NoneType

import datetime
import xlrd
import pandas as pd
from dateutil.parser import parse as parse_date
from .futil import find_files


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
    if ext is not None and output_ext is not None:
        dotted_ext = ('' if ext.startswith('.') else '.') + ext
        dotted_output_ext = ('' if output_ext.startswith('.') else '.') + output_ext
    table = {}
    for file_properties in find_files(path, ext=ext or '', verbosity=verbosity):
        file_path = file_properties['path']
        if output_ext and (dotted_output_ext + '.') in file_path:
            continue
        df = dataframe_from_excel(file_path, sheetname=sheetname, header=header, skiprows=skiprows)
        df = flatten_dataframe(df, verbosity=verbosity)
        if dotted_ext is not None and dotted_output_ext is not None:
            df.to_csv(file_path[:-len(dotted_ext)] + dotted_output_ext + dotted_ext)
    return table


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
        print('Columns: {0}'.format(df.columns))

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
            print(d)
        # # TODO: assert(not parser_date_exception)
        # if isinstance(d[0], basestring):
        #     d[0] = d[0]
        try:
            datetimeargs = list(d[0].timetuple()[:3]) + [d[1].hour, d[1].minute, d[1].second, d[1].microsecond]
            dt = datetime.datetime(*datetimeargs)
            if verbosity > 2:
                print('{0} -> {1}'.format(d, dt))
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
    if ext is not None and output_ext is not None:
        dotted_ext = ('' if ext.startswith('.') else '.') + ext
        dotted_output_ext = ('' if output_ext.startswith('.') else '.') + output_ext
    table = {}
    for file_properties in find_files(path, ext=ext or '', verbosity=verbosity):
        file_path = file_properties['path']
        if output_ext and (dotted_output_ext + '.') in file_path:
            continue
        df = pd.DataFrame.from_csv(file_path, parse_dates=False)
        df = flatten_dataframe(df)
        if dotted_ext is not None and dotted_output_ext is not None:
            df.to_csv(file_path[:-len(dotted_ext)] + dotted_output_ext + dotted_ext)
        table[file_path] = df
    return table