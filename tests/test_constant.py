"""Run doctests in pug.nlp.constant"""

import unittest
import doctest

import pug.nlp.constant


class T(unittest.TestCase):
    """Do-Nothing Test to ensure unittest doesnt ignore this file"""

    def setUp(self):
        pass

    def test_doctests(self):
        self.assertEqual(doctest.testmod(pug.nlp.constant, verbose=True).failed, 0)


def load_tests(loader, tests, ignore):
    """Run doctests for the clayton.nlp module"""
    tests.addTests(doctest.DocTestSuite(pug.nlp.constant, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE))
    return tests
