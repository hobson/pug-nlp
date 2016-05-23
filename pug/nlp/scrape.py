#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Crawlers and Scrapers for retrieving data/tables from URLs."""
from __future__ import division, print_function, absolute_import
from past.builtins import basestring

import os
from collections import OrderedDict
# import urllib2

# from bs4 import BeautifulSoup

from pug.nlp.regex_patterns import email_popular
from pug.nlp.util import make_name
from pug.nlp.constant import DATA_PATH
from pug.nlp.db import strip_nonascii
import pandas as pd


def find_emails(html=os.path.join(DATA_PATH, 'Locations.html')):
    """Extract email addresses from an html page or ASCII stream."""
    if isinstance(html, (str, bytes)):
        if os.path.isfile(html):
            html = open(html, 'r').read()
        html = email_popular.findall(html)
    return [x[0] for x in html]


uni_ascii = OrderedDict([
    (u'\xc2\xa0', ' '),      # nonbreaking? space:     " "
    (u'\xe2\x80\x91', '-'),  # smaller dash shifted left:     "‑"
    (u'\xe3\x81\xa3', '>'),  # backward skewed subscripted C: "っ"
    (u'\xc2\xa0', 'A'),     # Angstrom symbol
    (u'\u2011', '-'),      # smaller dash shifted left:     "‑"
    (u'\u3063', '>'),      # backward skewed subscripted C: "っ"
    (u'\xa0', ' '),         # nonbreaking? space:     " "
    ])


spaced_uni_emoticons = OrderedDict([
    # lenny
    (u'( ͡° ͜ʖ ͡°)', '(^-_-^)'),
    (u'( ͡°͜ ͡°)', '(^-_-^)'),
    (u'(͡° ͜ʖ ͡°)', '(^-_-^)'),
    (u'(͡°͜ ͡°)', '(^-_-^)'),
    # kiss
    (u"( '}{' )", "(_'}{'_)"),
    # lenny
    (u'( \xcd\xa1\xc2\xb0 \xcd\x9c\xca\x96 \xcd\xa1\xc2\xb0)', '(^-_-^)'),
    (u'( \xcd\xa1\xc2\xb0\xcd\x9c \xcd\xa1\xc2\xb0)', '(^-_-^)'),
    ])

spaced_ascii_emoticons = OrderedDict([
    (u"( '}{' )", "(_'}{'_)"),
    (u'( \xcd\xa1\xc2\xb0 \xcd\x9c\xca\x96 \xcd\xa1\xc2\xb0)', '(^-_-^)'),  # Lenny
    (u'( \xcd\xa1\xc2\xb0\xcd\x9c \xcd\xa1\xc2\xb0)', '(^-_-^)'),  # Lenny
    ])


def transcode_unicode(s):
    print(s)
    try:
        s = unicode(s).encode('utf-8')
    except:
        pass
    try:
        s = str(s).decode('utf-8')
    except:
        pass
    for c, equivalent in uni_ascii.iteritems():
        print(c)
        print(type(c))
        uni = unicode(s).replace(c, equivalent)
    return strip_nonascii(uni)


def clean_emoticon_wiki_table(html='https://en.wikipedia.org/wiki/List_of_emoticons',
                              save='list_of_emoticons-wikipedia-cleaned.csv',
                              data_dir=DATA_PATH,
                              table_num=1,
                              **kwargs):
    wikitables = pd.read_html(html, header=0)
    for i, wikidf in enumerate(wikitables):
        header = (' '.join(str(s).strip() for s in wikidf.columns)).lower()
        if table_num == i or (table_num is None and 'meaning' in header):
            break
    df = wikidf
    df.columns = [make_name(s, lower=True) for s in df.columns]
    table = []
    for icon, meaning in zip(df[df.columns[0]], df[df.columns[1]]):
        # kissing couple has space in it
        for ic, uni_ic in spaced_uni_emoticons.iteritems():
            icon = icon.replace(ic, uni_ic)
        for ic, asc_ic in spaced_ascii_emoticons.iteritems():
            icon = icon.replace(ic, asc_ic)
        icon = transcode_unicode(icon)
        icons = icon.split()
        for ic in icons:
            table += [[ic, meaning]]
    df = pd.DataFrame(table, columns=['emoticon', 'meaning'])
    if save:
        save = save if isinstance(save, basestring) else 'cleaned-emoticons-from-wikipedia.csv'
        df.to_csv(os.path.join(data_dir, save), encoding='utf-8', quoting=pd.io.common.csv.QUOTE_ALL)
    return df


def ascii_emoticon_table(html='http://git.emojione.com/demos/ascii-smileys.html',
                         save='ascii-smileys-from-emojione.csv',
                         data_dir=DATA_PATH,
                         table_num=0,
                         **kwargs):
    df = pd.read_html(html, header=0)[table_num]
    df = df[df.columns[:2]].copy()
    df.columns = ['emoticon', 'shortname']
    if save:
        save = save if isinstance(save, basestring) else 'ascii-smileys-from-emojione.csv'
        df.to_csv(os.path.join(data_dir, save), encoding='utf-8', quoting=pd.io.common.csv.QUOTE_ALL)


# # modified code downloaded from:
# # http://devwiki.beloblotskiy.com/index.php5/Generic_HTML_Table_parser_(python)
# # mods by: Aquil H. Abdullah
# from HTMLParser import HTMLParser
# # import pdb

# # Print debug info
# markup_debug_low = not True
# markup_debug_med = not True


# class NestedTableError(Exception):

#     """
#     Error raised when TableParser finds a nested table.
#     """

#     def __init__(self, msg):
#         self.msg = msg

#     def __str__(self):
#         return repr(self.msg)

# # Generic HTML table parser


# class TableParser(HTMLParser):

#     """
#     Class to handle extracting a table from an HTML Page.
#     NOTE: Does not handle Tables within
#     """

#     def __init__(self):
#         HTMLParser.__init__(self)
#         # Can't use super HTMLParser is an old-style class
#         # super(TableParser, self).__init__()
#         self._tables = list()  # Added to generic class
#         self._curr_table = list()  # Added to generic class
#         self._curr_row = list()  # Added to generic class
#         self._curr_cell = ''  # Added to generic class
#         self._in_table = False  # Added to generic class
#         self._td_cnt = 0
#         self._tr_cnt = 0
#         self._curr_tag = ''
#         self._colspan = 1

#     def get_tables(self):
#         """
#         Return the list of tables scraped from html page
#         """
#         return self._tables

#     def handle_starttag(self, tag, attrs):
#         self._curr_tag = tag
#         if tag.upper() == 'TABLE' and not self._in_table:
#             self._in_table = True
#         elif tag.upper() == 'TABLE' and self._in_table:
#             raise NestedTableError("Parsing Failed Nested Table Found.")

#         if tag == 'td':
#             self._td_cnt += 1
#             for attr in attrs:
#                 if attr[0].upper() == 'COLSPAN':
#                     self._colspan = int(attr[1])
#             self.col_start(self._td_cnt)
#             if markup_debug_low:
#                 print "<TD> --- %s ---" % self._td_cnt
#         elif tag == 'tr':
#             self._td_cnt = 0
#             self._tr_cnt += 1
#             self.row_start(self._tr_cnt)
#             if markup_debug_low:
#                 print "<TR> === %s ===" % self._tr_cnt
#         else:
#             if markup_debug_low:
#                 print "<%s>" % tag

#     def handle_endtag(self, tag):
#         if tag.upper() == 'TABLE':
#             self._in_table = False
#             self._tables.append(self._curr_table)
#             self._curr_table = list()
#         if markup_debug_low:
#             print "</%s>" % tag
#         # it's possible to check "start tag - end tag" pair here (see, tag and self._curr_tag)
#         if tag == 'tr':
#             self.row_finish(self._tr_cnt)
#         elif tag == 'td':
#             self.col_finish(self._td_cnt)
#             self._colspan = 1
#         else:
#             pass

#     def handle_data(self, data):
#         # if markup_debug_low: print u'[%s,%s] %s: "%s"' % (self._tr_cnt, self._td_cnt, self._curr_tag, unicode(data, 'mbcs'))
#         self.process_raw_data(self._tr_cnt, self._td_cnt, self._curr_tag, data)

#     # Overridable
#     def process_raw_data(self, row, col, tag, data):
#         if row > 0 and col > 0:
#             self.process_cell_data(row, col, tag, data)
#         else:
#             pass    # outside the table

#     # Overridable
#     def process_cell_data(self, row, col, tag, data):
#         # pass
#         self._curr_cell += data.strip() + ' '

#     # Overridable
#     def row_start(self, row):
#         # pass
#         self._curr_row = list()

#     # Overridable
#     def row_finish(self, row):
#         # pass
#         row = self._curr_row[:]
#         self._curr_table.append(row)

#     # Overridable
#     def col_start(self, col):
#         # pass
#         self._curr_cell = ''

#     # Overridable
#     def col_finish(self, col):
#         # pass
#         self._curr_row.append(self._curr_cell)
#         pad = self._colspan - 1
#         if pad > 0:
#             for i in range(pad):
#                 self._curr_row.append('')
