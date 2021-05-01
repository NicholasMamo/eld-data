"""
Test the functionality of the entity extractor.
"""

import json
import os
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from apd.extractors.local.entity_extractor import EntityExtractor

from nlp.document import Document
from nlp.tokenizer import Tokenizer

class TestExtractors(unittest.TestCase):
    """
    Test the implementation and results of the different extractors.
    """

    def test_entity_extractor(self):
        """
        Test the entity extractor with normal input.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Liverpool falter against Tottenham Hotspur",
            "Mourinho under pressure as Tottenham follow with a loss",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "liverpool", "tottenham hotspur" ]), set(candidates[0]))
        self.assertEqual(set([ "mourinho", "tottenham" ]), set(candidates[1]))

    def test_empty_corpus(self):
        """
        Test the entity extractor with an empty corpus.
        """

        extractor = EntityExtractor()
        candidates = extractor.extract([ ])
        self.assertFalse(len(candidates))

    def test_return_length(self):
        """
        Test that the entity extractor returns as many token sets as the number of documents given.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Liverpool falter against Tottenham Hotspur",
            "",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(2, len(candidates))
        self.assertEqual(set([ "liverpool", "tottenham hotspur" ]), set(candidates[0]))
        self.assertEqual(set([ ]), set(candidates[1]))

    def test_named_entity_at_start(self):
        """
        Test that the entity extractor is capable of extracting named entities at the start of a sentence.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Liverpool falter again",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertTrue("liverpool" in set(candidates[0]))

    def test_named_entity_at_end(self):
        """
        Test that the entity extractor is capable of extracting named entities at the end of a sentence.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Spiral continues for Lyon",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertTrue("lyon" in set(candidates[0]))

    def test_multiple_sentences(self):
        """
        Test that the entity extractor is capable of extracting named entities from multiple sentences.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "The downward spiral continues for Lyon. Bruno Genesio under threat.",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "lyon", "bruno genesio" ]), set(candidates[0]))

    def test_repeated_named_entities(self):
        """
        Test that the entity extractor does not filter named entities that appear multiple times.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "The downward spiral continues for Lyon. Lyon coach Bruno Genesio under threat.",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "lyon", "bruno genesio" ]), set(candidates[0]))

    def test_binary_named_entities(self):
        """
        Test that the entity extractor does not consider the entity type when the binary option is turned off.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "The downward spiral continues for Lyon. Rudi Garcia under threat.",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor(binary=False)
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "lyon", "rudi", "garcia" ]), set(candidates[0])) # 'Rudi' and 'Garcia' mistakenly have different types

        extractor = EntityExtractor(binary=True)
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "lyon", "rudi garcia" ]), set(candidates[0]))

    def test_comma_separated_entities(self):
        """
        Test that comma-separated named entities are returned individually.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Memphis Depay, Oumar Solet, Leo Dubois and Youssouf Kone all out injured",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = EntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "memphis depay", "oumar solet", 'leo dubois', 'youssouf kone' ]), set(candidates[0]))

    def test_extract_from_text(self):
        """
        Test that the entity extractor's named entities do appear in the corresponding tweet.
        """

        """
        Load the corpus.
        """
        filename = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tests', 'corpora', 'understanding', 'CRYCHE-100.json')
        corpus = [ ]
        with open(filename) as f:
            for i, line in enumerate(f):
                tweet = json.loads(line)
                original = tweet
                while "retweeted_status" in tweet:
                    tweet = tweet["retweeted_status"]

                if "extended_tweet" in tweet:
                    text = tweet["extended_tweet"].get("full_text", tweet.get("text", ""))
                else:
                    text = tweet.get("text", "")

                document = Document(text)
                corpus.append(document)

        extractor = EntityExtractor()
        candidates = extractor.extract(corpus)
        for (document, candidate_set) in zip(corpus, candidates):
            text = document.text.lower().replace('\n', ' ').replace('  ', ' ')
            self.assertTrue(all( candidate in text.lower() for candidate in candidate_set ))
