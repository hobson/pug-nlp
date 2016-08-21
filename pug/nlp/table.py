#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""table utils"""
import xlrd
import pandas as pd


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