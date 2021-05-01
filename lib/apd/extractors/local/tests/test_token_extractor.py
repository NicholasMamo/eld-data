"""
Test the functionality of the token extractor.
"""

import os
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nltk.corpus import stopwords

from apd.extractors.local.token_extractor import TokenExtractor

from nlp.document import Document
from nlp.tokenizer import Tokenizer

def ignore_warnings(test):
    """
    A decorator function used to ignore NLTK warnings
    From: http://www.neuraldump.net/2017/06/how-to-suppress-python-unittest-warnings/
    More about decorator functions: https://wiki.python.org/moin/PythonDecorators

    :param test: The test to perform.
    :type test: func

    :return: The function output.
    :rtype: obj
    """
    def perform_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test(self, *args, **kwargs)
    return perform_test

class TestExtractors(unittest.TestCase):
    """
    Test the implementation and results of the token extractor.
    """

    @ignore_warnings
    def test_token_extractor(self):
        """
        Test the token extractor with normal input.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words("english"), stem=False)
        posts = [
            "Manchester United falter against Tottenham Hotspur",
            "Mourinho under pressure as Manchester United follow with a loss",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TokenExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "manchester", "united", "falter", "tottenham", "hotspur" ]), set(candidates[0]))
        self.assertEqual(set([ "mourinho", "pressure", "manchester", "united", "follow", "loss" ]), set(candidates[1]))

    @ignore_warnings
    def test_empty_corpus(self):
        """
        Test the token extractor with an empty corpus.
        """

        extractor = TokenExtractor()
        candidates = extractor.extract([ ])
        self.assertFalse(len(candidates))

    @ignore_warnings
    def test_return_length(self):
        """
        Test that the token extractor returns as many token sets as the number of documents given.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words("english"), stem=False)
        posts = [
            "Manchester United falter against Tottenham Hotspur",
            "",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TokenExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(2, len(candidates))
        self.assertEqual(set([ "manchester", "united", "falter", "tottenham", "hotspur" ]), set(candidates[0]))
        self.assertEqual(set([ ]), set(candidates[1]))

    @ignore_warnings
    def test_extract_with_custom_tokenizer(self):
        """
        Test that when a custom tokenizer is given, it is used instead of the dimensions.
        """

        """
        Create the test data, which uses stemming.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words("english"), stem=True)
        posts = [
            "Manchester United back to winning ways",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TokenExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "manchest", "unit", "back", "win", "way" ]), set(candidates[0]))

        extractor = TokenExtractor(tokenizer=Tokenizer(stopwords=stopwords.words('english'), stem=False))
        candidates = extractor.extract(corpus)
        self.assertEqual(set([ "manchester", "united", "back", "winning", "ways" ]), set(candidates[0]))

    @ignore_warnings
    def test_repeated_tokens_with_custom_tokenizer(self):
        """
        Test that when a custom tokenizer is given, repeated tokenizers appear multiple times.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words("english"), stem=False)
        posts = [
            "Manchester United back to winning ways after defeating Manchester City.",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        extractor = TokenExtractor()
        candidates = extractor.extract(corpus)
        self.assertEqual(1, candidates[0].count('manchester'))

        extractor = TokenExtractor(tokenizer=tokenizer)
        candidates = extractor.extract(corpus)
        self.assertEqual(2, candidates[0].count('manchester'))
