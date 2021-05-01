"""
Test the functionality of the threshold filter.
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
from apd.filters.local.threshold_filter import ThresholdFilter

from nlp.document import Document
from nlp.tokenizer import Tokenizer

class TestThresholdFilter(unittest.TestCase):
    """
    Test the implementation and results of the threshold filter.
    """

    def test_threshold_filter(self):
        """
        Test the basic functionality of the threshold filter.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Erdogan with threats to attack regime forces 'everywhere' in Syria",
            "Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        scorer = TFScorer()
        filter = ThresholdFilter(0.75)

        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)

        self.assertEqual(1, scores.get('erdogan', 0))
        self.assertEqual(0.5, scores.get('damascus', 0))

        scores = filter.filter(scores)
        self.assertTrue('erdogan' in scores)
        self.assertFalse('damascus' in scores)

    def test_negative_threshold(self):
        """
        Test that when a negative threshold is given, all candidate participants are retained.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Erdogan with threats to attack regime forces 'everywhere' in Syria",
            "Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        scorer = TFScorer()
        filter = ThresholdFilter(-1)

        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)

        self.assertEqual(1, scores.get('erdogan', 0))
        self.assertEqual(0.5, scores.get('damascus', 0))

        scores = filter.filter(scores)
        self.assertTrue('erdogan' in scores)
        self.assertTrue('damascus' in scores)

    def test_zero_threshold(self):
        """
        Test that when a threshold of zero is given, all candidate participants are retained.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Erdogan with threats to attack regime forces 'everywhere' in Syria",
            "Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        scorer = TFScorer()
        filter = ThresholdFilter(0)

        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)

        self.assertEqual(1, scores.get('erdogan', 0))
        self.assertEqual(0.5, scores.get('damascus', 0))

        scores = filter.filter(scores)
        self.assertTrue('erdogan' in scores)
        self.assertTrue('damascus' in scores)

    def test_equal_threshold(self):
        """
        Test that when a named entity has a score that is equal to the threshold, it is retained.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Erdogan with threats to attack regime forces 'everywhere' in Syria",
            "Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        scorer = TFScorer()
        filter = ThresholdFilter(0.5)

        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)

        self.assertEqual(1, scores.get('erdogan', 0))
        self.assertEqual(0.5, scores.get('damascus', 0))

        scores = filter.filter(scores)
        self.assertTrue('erdogan' in scores)
        self.assertTrue('damascus' in scores)

    def test_marginal_threshold(self):
        """
        Test that when a threshold exceeds a candidate score, the filter excludes them.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Erdogan with threats to attack regime forces 'everywhere' in Syria",
            "Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        scorer = TFScorer()
        filter = ThresholdFilter(0.51)

        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)

        self.assertEqual(1, scores.get('erdogan', 0))
        self.assertEqual(0.5, scores.get('damascus', 0))

        scores = filter.filter(scores)
        self.assertTrue('erdogan' in scores)
        self.assertFalse('damascus' in scores)

    def test_high_threshold(self):
        """
        Test that when a high threshold is given, no candidate participants are retained.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Erdogan with threats to attack regime forces 'everywhere' in Syria",
            "Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        scorer = TFScorer()
        filter = ThresholdFilter(1.1)

        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)

        self.assertEqual(1, scores.get('erdogan', 0))
        self.assertEqual(0.5, scores.get('damascus', 0))

        scores = filter.filter(scores)
        self.assertFalse('erdogan' in scores)
        self.assertFalse('damascus' in scores)
