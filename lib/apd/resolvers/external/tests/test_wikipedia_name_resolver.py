"""
Test the functionality of the Wikipedia name resolver.
"""

import os
import random
import string
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nltk.corpus import stopwords

from apd.extractors.local.entity_extractor import EntityExtractor
from apd.scorers.local.tf_scorer import TFScorer
from apd.filters.local.threshold_filter import ThresholdFilter
from apd.resolvers.external.wikipedia_name_resolver import WikipediaNameResolver

from nlp.document import Document
from nlp.tokenizer import Tokenizer
from nlp.weighting.tf import TF

class TestWikipediaNameResolver(unittest.TestCase):
    """
    Test the implementation and results of the Wikipedia name resolver.
    """

    def test_wikipedia_name_resolver(self):
        """
        Test the Wikipedia name resolver.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=1, stem=False)
        posts = [
            "Manchester United falter against Burnley",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)
        scores = ThresholdFilter(0).filter(scores)

        resolver = WikipediaNameResolver(TF(), tokenizer, 0, corpus)
        resolved, unresolved = resolver.resolve(scores)

        self.assertTrue('manchester united' in resolved)
        self.assertTrue('burnley' in resolved)

    def test_all_resolved_or_unresolved(self):
        """
        Test that the resolver either resolves or does not resolve named entities.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=1, stem=False)
        posts = [
            "Manchester United falter against Burnley",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)
        scores = ThresholdFilter(0).filter(scores)

        resolver = WikipediaNameResolver(TF(), tokenizer, 0, corpus)
        resolved, unresolved = resolver.resolve(scores)
        self.assertEqual(len(scores), len(resolved + unresolved))

    def test_random_string_unresolved(self):
        """
        Test that a random string is unresolved.
        """

        random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(32))

        tokenizer = Tokenizer(min_length=1, stem=False)
        resolver = WikipediaNameResolver(TF(), tokenizer, 0, [ ])
        resolved, unresolved = resolver.resolve({ random_string: 1 })
        self.assertTrue(random_string in unresolved)

    def test_low_threshold(self):
        """
        Test that when the threshold is zero, it includes all ambiguous candidates.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=2, stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "Memphis mum about his future at Lyon after the Dutch footballer wins it for the Ligue 1 team",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)

        resolver = WikipediaNameResolver(TF(), tokenizer, 0.1, corpus)
        resolved, unresolved = resolver.resolve(scores)
        self.assertTrue('Memphis Depay' in resolved)
        self.assertFalse(unresolved)

    def test_low_threshold(self):
        """
        Test that when the threshold is not zero, it excludes some ambiguous candidates.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=2, stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "Memphis mum about his future at Lyon after the Dutch footballer wins it for the Ligue 1 team",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)

        resolver = WikipediaNameResolver(TF(), tokenizer, 0.4, corpus)
        resolved, unresolved = resolver.resolve(scores)
        self.assertTrue('Memphis' in unresolved)

    def test_resolve_empty(self):
        """
        Test that when resolving an empty set of candidates, the resolver returns empty lists.
        """

        resolver = WikipediaNameResolver(TF(), Tokenizer(), 0, [ ])
        resolved, unresolved = resolver.resolve({ })
        self.assertFalse(len(resolved))
        self.assertFalse(len(unresolved))

    def test_sorting(self):
        """
        Test that the resolver sorts the named entities in descending order of score.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=3, stem=False, case_fold=True)
        posts = [
            "Manchester United falter against Manchester City",
            "Manchester United unable to avoid defeat to Watford",
            "Watford lose again",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)
        scores = ThresholdFilter(0).filter(scores)
        resolved, unresolved = WikipediaNameResolver(TF(), tokenizer, 0, corpus).resolve(scores)
        self.assertEqual('manchester united', resolved[0])
        self.assertEqual('watford', resolved[1])
        self.assertEqual('manchester city', resolved[2])

    def test_sorting_ambiguous(self):
        """
        Test that the resolver sorts the named entities in descending order of score, but ambiguous candidates are at the end.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=3, stem=False, case_fold=True)
        posts = [
            "Manchester United falter against Manchester City",
            "Manchester United unable to avoid defeat to Tottenham",
            "Tottenham lose again",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)
        scores = ThresholdFilter(0).filter(scores)
        resolved, unresolved = WikipediaNameResolver(TF(), tokenizer, 0, corpus).resolve(scores)
        self.assertEqual('manchester united', resolved[0])
        self.assertEqual('manchester city', resolved[1])
        self.assertEqual('tottenham', resolved[2])
