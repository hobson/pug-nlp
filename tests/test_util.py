"""Run doctests in pug.nlp.util."""
from __future__ import print_function, absolute_import

import doctest

import pug.nlp.util

from unittest import TestCase


class DoNothingTest(TestCase):
    """A useless TestCase to encourage Django unittests to find this module and run `load_tests()`."""
    def test_example(self):
        self.assertTrue(True)


def load_tests(loader, tests, ignore):
    """Run doctests for the pug.nlp.util module"""
    tests.addTests(doctest.DocTestSuite(pug.nlp.util, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE))
    return tests
