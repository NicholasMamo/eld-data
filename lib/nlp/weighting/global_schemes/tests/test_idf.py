"""
Run unit tests on the :class:`~nlp.weighting.global_schemes.idf.IDF` class.
"""

import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..')
if path not in sys.path:
    sys.path.append(path)

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nltk.corpus import stopwords

from idf import IDF
from document import Document
from tokenizer import Tokenizer

class TestIDF(unittest.TestCase):
    """
    Test the :class:`~nlp.weighting.global_schemes.idf.IDF` class.
    """

    def test_high_value(self):
        """
        Test that the IDF raises an error when the highest IDF value is higher than the number of documents.
        """

        idf = { 'a': 3, 'b': 1 }
        self.assertRaises(ValueError, IDF, idf, 2)

    def test_negative_value(self):
        """
        Test that the IDF raises an error when any IDF value is negative.
        """

        idf = { 'a': 3, 'b': -1 }
        self.assertRaises(ValueError, IDF, idf, 3)

    def test_negative_documents(self):
        """
        Test that the IDF raises an error when the number of documents is negative.
        """

        idf = { 'a': 3, 'b': 1 }
        self.assertRaises(ValueError, IDF, idf, -1)

    def test_idf_equal_documents(self):
        """
        Test that when the DF of a feature is cloes to the number of documents, the result is 0.
        """

        idf = { 'a': 3, 'b': 1 }
        idf = IDF(idf, 4)
        tokens = [ 'a' ]
        self.assertEqual(0, idf.score(tokens)['a'])

    def test_idf(self):
        """
        Test IDF in normal conditions.
        """

        idf = { 'a': 3, 'b': 1 }
        idf = IDF(idf, 4)
        tokens = [ 'b' ]
        self.assertEqual(0.30103, round(idf.score(tokens)['b'], 5))

    def test_idf_zero_term(self):
        """
        Test that IDF scores terms even if they do not appear in the IDF table.
        """

        idf = { 'a': 3, 'b': 1 }
        idf = IDF(idf, 4)
        tokens = [ 'c' ]
        self.assertEqual(0.60206, round(idf.score(tokens)['c'], 5))

    """
    IDF table creation.
    """

    def test_create_idf(self):
        """
        Test that the IDF correctly counts the document frequency of terms.
        """

        """
        Create the test corpus.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words("english"), stem=False)
        posts = [
            "Manchester United falter against Tottenham Hotspur",
            "Mourinho under pressure as Manchester United follow with a loss",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        """
        Create the IDF table.
        """

        idf = IDF.from_documents(corpus)
        self.assertEqual(2, idf.get('manchester'))
        self.assertEqual(2, idf.get('united'))
        self.assertEqual(1, idf.get('falter'))
        self.assertEqual(1, idf.get('tottenham'))
        self.assertEqual(1, idf.get('hotspur'))
        self.assertEqual(1, idf.get('mourinho'))
        self.assertEqual(1, idf.get('pressure'))
        self.assertEqual(1, idf.get('loss'))

    def test_empty_idf(self):
        """
        Test that when no documents are given, the IDF is an empty dictionary.
        """

        corpus = [ ]

        idf = IDF.from_documents(corpus)
        self.assertEqual({ }, idf)

    def test_document_frequency(self):
        """
        Test that the IDF counts the document frequency of terms, not the term frequency.
        """

        """
        Create the test corpus.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words("english"), stem=False)
        posts = [
            "Manchester United win the derby of Manchester against Manchester City"
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        """
        Create the IDF table.
        """

        idf = IDF.from_documents(corpus)
        self.assertEqual(1, idf.get('manchester'))

    def test_export(self):
        """
        Test exporting and importing the IDF table.
        """

        """
        Create the test corpus.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words("english"), stem=False)
        posts = [
            "Manchester United falter against Tottenham Hotspur",
            "Mourinho under pressure as Manchester United follow with a loss",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        idf = IDF(IDF.from_documents(corpus), len(corpus))
        e = idf.to_array()
        self.assertEqual(idf.documents, IDF.from_array(e).documents)
        self.assertEqual(idf.idf, IDF.from_array(e).idf)
        self.assertEqual(idf.__dict__, IDF.from_array(e).__dict__)
