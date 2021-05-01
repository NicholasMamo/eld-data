"""
Test the functionality of the TwitterNER entity extractor.
"""

import json
import os
import re
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from apd.extractors.local.twitterner_entity_extractor import TwitterNEREntityExtractor

from nlp.document import Document
from nlp.tokenizer import Tokenizer

class TestTwitterNERExtractor(unittest.TestCase):
    """
    Test the functionality of the TwitterNER entity extractor.
    """

    def test_ner_extractor(self):
        """
        Test that the NER extractor is available even before creating the TwitterNEREntityExtractor.
        """

        self.assertTrue(TwitterNEREntityExtractor.ner)

    def test_init_ner_extractor(self):
        """
        Test that when creating a TwitterNER entity extractor, the existing NER extractor is already available.
        """

        self.assertTrue(TwitterNEREntityExtractor.ner)

    def test_extract_example(self):
        """
        Test extracting the entities using the example tweet from the GitHub repository.
        """

        extractor = TwitterNEREntityExtractor()
        corpus = [ Document(text='Beautiful day in Chicago! Nice to get away from the Florida heat.') ]
        self.assertEqual([ 'chicago', 'florida' ], extractor.extract(corpus)[0])

    def test_extract(self):
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

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual([ "liverpool", "tottenham" ], candidates[0])
        self.assertEqual([ "tottenham" ], candidates[1])

    def test_extract_empty_corpus(self):
        """
        Test that when extracting the entities from an empty corpus, an empty list is returned.
        """

        extractor = TwitterNEREntityExtractor()
        self.assertFalse(extractor.extract([ ]))

    def test_extract_empty_tweet(self):
        """
        Test that the TwitterNER entity extractor returns no candidates from an empty tweet.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [ "" ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(1, len(candidates))
        self.assertEqual([ ], candidates[0])

    def test_extract_return_length(self):
        """
        Test that the TwitterNER entity extractor returns as many candidate sets as the number of documents given.
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

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(2, len(candidates))
        self.assertEqual([ "liverpool", "tottenham" ], candidates[0])
        self.assertEqual([ ], candidates[1])

    def test_extract_named_entity_at_start(self):
        """
        Test that the TwitterNER entity extractor is capable of extracting named entities at the start of a sentence.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Liverpool falter again",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertTrue("liverpool" in candidates[0])

    def test_extract_named_entity_at_end(self):
        """
        Test that the TwitterNER entity extractor is capable of extracting named entities at the end of a sentence.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Spiral continues for Lyon",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertTrue("lyon" in set(candidates[0]))

    def test_extract_multiple_sentences(self):
        """
        Test that the TwitterNER entity extractor is capable of extracting named entities from multiple sentences.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "The downward spiral continues for Lyon. Bruno Genesio under threat.",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual([ "lyon", "bruno genesio" ], candidates[0])

    def test_extract_repeated_named_entities(self):
        """
        Test that the TwitterNER entity extractor does not filter named entities that appear multiple times.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "The downward spiral continues for Lyon. Meanwhile, Lyon coach Bruno Genesio remains under threat.",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual([ "lyon", "lyon", "bruno genesio" ], candidates[0])

    def test_extract_multiword_entities(self):
        """
        Test that the TwitterNER entity extractor is capable of extracting multi-word entities.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Lyon were delivered by Karl Toko Ekambi",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual([ 'lyon', 'karl toko ekambi' ], candidates[0])

    def test_extract_comma_separated_entities(self):
        """
        Test that comma-separated named entities are returned individually.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Memphis Depay, Leo Dubois, Martin Terrier and Karl Toko Ekambi all out injured",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual([ "memphis depay", 'leo dubois', 'martin terrier', 'karl toko ekambi' ], candidates[0])

    def test_extract_order(self):
        """
        Test that the named entities are returned in the correct order.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Memphis Depay, Leo Dubois, Martin Terrier and Karl Toko Ekambi all out injured",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual([ "memphis depay", 'leo dubois', 'martin terrier', 'karl toko ekambi' ], candidates[0])

    def test_extract_from_text(self):
        """
        Test that TwitterNER's named entities do appear in the corresponding tweet.
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

        extractor = TwitterNEREntityExtractor()
        candidates = extractor.extract(corpus)
        collapse_spaces = re.compile('\\s+')
        for (document, candidate_set) in zip(corpus, candidates):
            text = document.text.lower().replace('\n', ' ')
            text = collapse_spaces.sub(' ', text)
            self.assertTrue(all( candidate in text.lower() for candidate in candidate_set ))
