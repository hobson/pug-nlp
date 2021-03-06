"""Run doctests in pug.nlp.tutil."""
from __future__ import print_function, absolute_import

import doctest

import pug.nlp.tutil

from unittest import TestCase


class DoNothingTest(TestCase):
    """A useless TestCase to encourage Django unittests to find this module and run `load_tests()`."""
    def test_example(self):
        self.assertTrue(True)


def load_tests(loader, tests, ignore):
    """Run doctests for the pug.nlp.tutil module"""
    tests.addTests(doctest.DocTestSuite(pug.nlp.tutil, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE))
    return tests
