"""
Test the functionality of the TF scorer.
"""

import math
import os
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from apd.extractors.local.token_extractor import TokenExtractor
from apd.scorers.local.tf_scorer import TFScorer

from nlp.document import Document
from nlp.tokenizer import Tokenizer

class TestTFScorer(unittest.TestCase):
    """
    Test the implementation and results of the TF scorer.
    """

    def test_tf_scorer(self):
        """
        Test the basic functionality of the TF scorer.
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

        extractor = TokenExtractor()
        scorer = TFScorer()
        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)
        self.assertEqual(1, scores.get('erdogan', 0))
        self.assertEqual(0.5, scores.get('damascus', 0))
        self.assertEqual(1, scores.get('threats', 0))

    def test_min_score(self):
        """
        Test that the minimum score is greater than 0.
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

        extractor = TokenExtractor()
        scorer = TFScorer()
        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)
        self.assertTrue(all( score > 0 for score in scores.values() ))

    def test_max_score(self):
        """
        Test that the maximum score is 1 when normalization is enabled.
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

        extractor = TokenExtractor()
        scorer = TFScorer()
        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)
        self.assertTrue(all( score <= 1 for score in scores.values() ))

    def test_score_of_unknown_token(self):
        """
        Test that the score of an unknown token is 0.
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

        extractor = TokenExtractor()
        scorer = TFScorer()
        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates)
        self.assertFalse(scores.get('unknown'))

    def test_score_across_multiple_documents(self):
        """
        Test that the score is based on term frequency.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Erdogan with threats to attack regime forces 'everywhere' in Syria",
            "After Erdogan's statement, Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TokenExtractor(tokenizer=tokenizer)
        scorer = TFScorer()
        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates, normalize_scores=False)
        self.assertEqual(3, scores.get('erdogan'))

    def test_normalization(self):
        """
        Test that when normalization is disabled, the returned scores are integers.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Erdogan with threats to attack regime forces 'everywhere' in Syria",
            "After Erdogan's statement, Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TokenExtractor()
        scorer = TFScorer()
        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates, normalize_scores=False)
        self.assertEqual(2, scores.get('erdogan'))

    def test_repeated_tokens(self):
        """
        Test that when tokens are repeated, the frequency that is returned is the term frequency.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "After Erdogan's statement, Damascus says Erdogan 'disconnected from reality' after threats",
        ]

        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TokenExtractor(tokenizer=tokenizer)
        scorer = TFScorer()
        candidates = extractor.extract(corpus)
        scores = scorer.score(candidates, normalize_scores=False)
        self.assertEqual(2, scores.get('erdogan'))
