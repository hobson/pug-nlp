"""Run doctests in pug.nlp.regex"""

import unittest
import doctest

import pug.nlp.regex


class T(unittest.TestCase):
    """Do-Nothing Test to ensure unittest doesnt ignore this file"""

    def setUp(self):
        pass

    def test_doctests(self):
        self.assertTrue(True)
        # self.assertEqual(doctest.testmod(pug.nlp.regex, verbose=True).failed, 0)


def load_tests(loader, tests, ignore):
    """Run doctests for the pug.nlp.regex module"""
    tests.addTests(doctest.DocTestSuite(pug.nlp.regex, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE))
    return tests
