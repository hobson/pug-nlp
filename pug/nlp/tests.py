#!/usr/bin/env python
"""
Uses the python unittest module to test this app with `python -m unittest pug.nlp`.
"""

# from django.test import TestCase
from unittest import TestCase, main
import doctest
from pug.nlp import util, http, regex, penn_treebank_tokenizer, detector_morse


class NLPDocTest(TestCase):

    def test_module(self, module=None):
        if module:
            failure_count, test_count = doctest.testmod(module, raise_on_error=False, verbose=True)
            msg = "Ran {0} tests in {3} and {1} passed ({2} failed)".format(test_count, test_count-failure_count, failure_count, module.__file__)
            print msg
            if failure_count:
                # print "Ignoring {0} doctest failures...".format(__file__)
                self.fail(msg)
            # return failure_count, test_count

    def test_util(self):
        self.test_module(util)

    def test_http(self):
        self.test_module(http)

    def test_regex_patterns(self):
        self.test_module(regex)

    def test_penn_treebank_tokenizer(self):
        self.test_module(penn_treebank_tokenizer)

    def test_detector_morse(self):
        self.test_module(detector_morse)


if __name__ == '__main__':
    main()
