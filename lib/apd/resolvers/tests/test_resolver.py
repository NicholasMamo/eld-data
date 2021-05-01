"""
Test the functionality of the base resolver.
"""

import os
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nltk.corpus import stopwords

from apd.extractors.local.entity_extractor import EntityExtractor
from apd.extractors.local.token_extractor import TokenExtractor
from apd.scorers.local.tf_scorer import TFScorer
from apd.filters.local.threshold_filter import ThresholdFilter
from apd.resolvers import Resolver

from nlp.document import Document
from nlp.tokenizer import Tokenizer

class TestResolver(unittest.TestCase):
    """
    Test the implementation and results of the base resolver.
    """

    def test_resolve_empty(self):
        """
        Test that when resolving an empty set of candidates, nothing is returned.
        """

        resolved, unresolved = Resolver().resolve({ })
        self.assertEqual([ ], resolved)
        self.assertEqual([ ], unresolved)

    def test_resolve_all(self):
        """
        Test that when resolving candidates, all of them are returned.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=3, stem=False, case_fold=True)
        posts = [
            "Manchester United falter against Tottenham Hotspur",
            "Manchester United unable to avoid defeat to Tottenham",
            "Tottenham lose again",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        """
        Ensure that all candidates are resolved.
        """
        candidates = TokenExtractor().extract(corpus)
        scores = TFScorer().score(candidates)
        scores = ThresholdFilter(0).filter(scores)
        self.assertTrue(scores)
        resolved, unresolved = Resolver().resolve(scores)
        self.assertEqual(set(scores.keys()), set(resolved))
        self.assertEqual([ ], unresolved)

    def test_sorting(self):
        """
        Test that the resolver sorts the tokens in descending order of score.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=3, stem=False, case_fold=True)
        posts = [
            "Manchester United falter against Tottenham Hotspur",
            "Manchester United unable to avoid defeat to Tottenham",
            "Tottenham lose again",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        """
        Ensure that the more common candidates are ranked towards the beginning.
        """
        candidates = TokenExtractor().extract(corpus)
        scores = TFScorer().score(candidates)
        scores = ThresholdFilter(0).filter(scores)
        self.assertTrue(scores)
        resolved, unresolved = Resolver().resolve(scores)
        self.assertEqual(set(scores.keys()), set(resolved))
        self.assertEqual([ ], unresolved)
        self.assertEqual('tottenham', resolved[0])
        self.assertEqual(set([ 'manchester', 'united' ]), set(resolved[1:3]))
