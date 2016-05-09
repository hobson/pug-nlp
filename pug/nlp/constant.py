#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Constants and discovered values, like path to current installation of pug-nlp."""
from __future__ import division, print_function, absolute_import

import os

import pug.nlp

BASE_PATH = os.path.dirname(pug.nlp.__file__)
DATA_PATH = os.path.join(BASE_PATH, 'data')
