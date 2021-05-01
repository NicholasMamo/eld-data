"""
Run unit tests on the search module.
"""

import os
import random
import string
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from wikinterface import search

class TestSearch(unittest.TestCase):
    """
    Test the search module.
    """

    def test_search_no_terms(self):
        """
        Test that when no terms are given, no articles are returned.
        """

        self.assertEqual([ ], search.collect([ ]))

    def test_search_one_term(self):
        """
        Test that when one term is given, relevant results are returned.
        """

        self.assertTrue(len(search.collect('Lyon')))

    def test_search_multiple_terms(self):
        """
        Test that when multiple terms are given, relevant results are given.
        """

        self.assertTrue(len(search.collect([ 'Lyon', 'Bordeaux' ])))

    def test_search_negative_limit(self):
        """
        Test that when searching with a negative limit, an exception is raised.
        """

        self.assertRaises(ValueError, search.collect, 'Lyon', -1)

    def test_search_zero_limit(self):
        """
        Test that when searching with a limit of zero, an exception is raised.
        """

        self.assertRaises(ValueError, search.collect, 'Lyon', 0)

    def test_search_limit(self):
        """
        Test that when searching with a limit, no more than that limit are returned.
        """

        self.assertEqual(10, len(search.collect('Lyon', 10)))

    def test_search_large_limit(self):
        """
        Test that when searching with a limit larger than 50, more than 50 articles are returned.
        """

        self.assertEqual(100, len(search.collect('Lyon', 100)))

    def test_search_unknown_term(self):
        """
        Test that when searching for an unknown term, no results are returned.
        """

        random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(32))
        self.assertFalse(len(search.collect(random_string)))

    def test_search_with_accent(self):
        """
        Test that when searching for a term with an accent, results are returned.
        """

        self.assertTrue(len(search.collect('Tătărușanu')))

    def test_search_with_ampersand(self):
        """
        Test that when performing a search for a term containing an ampersand, it is considered as a term.
        """

        self.assertTrue(len(search.collect('calvin & hobbes')))

    def test_search_with_space(self):
        """
        Test that when performing a search for a term containing a whitespace, it is considered as a term.
        """

        articles = search.collect('Ciprian Tătărușanu')
        self.assertTrue(len(articles))
        self.assertTrue('Ciprian Tătărușanu' in articles)
