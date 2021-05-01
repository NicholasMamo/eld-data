"""
Run unit tests on the :class:`~nlp.weighting.global_schemes.filler.Filler` class.
"""

import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..')
if path not in sys.path:
    sys.path.append(path)

from filler import Filler

class TestFiller(unittest.TestCase):
    """
    Test the :class:`~nlp.weighting.global_schemes.filler.Filler` class.
    """

    def test_empty_list_score(self):
        """
        Test that weighting an empty list returns no weights.
        """

        tokens = []
        scheme = Filler()
        self.assertEqual({ }, scheme.score(tokens))

    def test_list_score(self):
        """
        Test that weighting a list returns the weights of the documents.
        """

        tokens = [ 'a', 'b' ]
        scheme = Filler()
        self.assertEqual({ 'a': 1, 'b': 1 }, scheme.score(tokens))

    def test_repeated_score(self):
        """
        Test that weighting a list with repeated features returns boolean weights.
        """

        tokens = [ 'a', 'b', 'a' ]
        scheme = Filler()
        self.assertEqual({ 'a': 1, 'b': 1 }, scheme.score(tokens))
