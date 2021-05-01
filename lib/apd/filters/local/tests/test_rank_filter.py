"""
Test the functionality of the rank filter.
"""

import math
import os
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from apd.extractors.local.entity_extractor import EntityExtractor
from apd.scorers.local.tf_scorer import TFScorer
from apd.filters.local.rank_filter import RankFilter

from nlp.document import Document
from nlp.tokenizer import Tokenizer

class TestRankFilter(unittest.TestCase):
    """
    Test the implementation and results of the rank filter.
    """

    def test_k_float(self):
        """
        Test that when k is a float, the rank filter raises a ValueError.
        """

        self.assertRaises(ValueError, RankFilter, 1.1)

    def test_k_string(self):
        """
        Test that when k is a string, the rank filter raises a ValueError.
        """

        self.assertRaises(ValueError, RankFilter, "1")

    def test_k_int(self):
        """
        Test that when k is a int, the rank filter does not raise a ValueError.
        """

        self.assertTrue(RankFilter(1))

    def test_k_negative(self):
        """
        Test that when k is negative, the rank filter raises a ValueError.
        """

        self.assertRaises(ValueError, RankFilter, -1)

    def test_k_zero(self):
        """
        Test that when k is zero, the rank filter raises a ValueError.
        """

        self.assertRaises(ValueError, RankFilter, 0)

    def test_k_positive(self):
        """
        Test that when k is positive, the rank filter does not raise a ValueError.
        """

        self.assertTrue(RankFilter(1))

    def test_filter_empty(self):
        """
        Test that when filtering an empty set of candidates, nothing is returned.
        """

        filter = RankFilter(10)
        self.assertEqual({ }, filter.filter({ }))

    def test_filter_few(self):
        """
        Test that when filtering a number of candidates that is less than _k_, all candidates are returned.
        """

        filter = RankFilter(10)
        candidates = [ 'a', 'b', 'c', 'd', 'e' ]
        candidates = { candidate: 1 for candidate in candidates }
        self.assertEqual(candidates, filter.filter(candidates))

    def test_filter_all(self):
        """
        Test that when filtering a number of candidates that is equal to _k_, all candidates are returned.
        """

        filter = RankFilter(10)
        candidates = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j' ]
        candidates = { candidate: 1 for candidate in candidates }
        self.assertEqual(candidates, filter.filter(candidates))

    def test_filter_top(self):
        """
        Test that when filtering candidates, the top candidates are returned.
        """

        filter = RankFilter(5)
        vocabulary = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j' ]
        candidates = { candidate: i for i, candidate in enumerate(vocabulary) }
        self.assertEqual(set(vocabulary[-5:]), set(filter.filter(candidates).keys()))

    def test_filter_scores(self):
        """
        Test that when filtering candidates, the scores are retained the same.
        """

        filter = RankFilter(5)
        vocabulary = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j' ]
        candidates = { candidate: i for i, candidate in enumerate(vocabulary) }
        filtered = filter.filter(candidates)
        self.assertTrue(all( candidates[candidate] == score
                             for candidate, score in filtered.items() ))
