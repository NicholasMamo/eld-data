"""
Run unit tests on the :class:`~nlp.weighting.tfidf.TFIDF` class.
"""

import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..')
if path not in sys.path:
    sys.path.append(path)

from tfidf import TFIDF

class TestTFIDF(unittest.TestCase):
    """
    Test the TF-IDF term-weighting scheme.
    """

    def test_tfidf(self):
        """
        Test the TF-IDF scheme.
        """

        idf = { 'a': 2, 'b': 1, 'c': 1 }
        tokens = [ 'a', 'b', 'b', 'c', 'd' ]
        tfidf = TFIDF(idf, 3)

        document = tfidf.create(tokens)
        self.assertEqual(0, document.dimensions['a'])
        self.assertEqual(0.35218, round(document.dimensions['b'], 5))
        self.assertEqual(0.17609, round(document.dimensions['c'], 5))
        self.assertEqual(0.47712, round(document.dimensions['d'], 5))

    def test_export(self):
        """
        Test exporting and importing the IDF table.
        """

        idf = { 'a': 2, 'b': 1, 'c': 1 }
        tfidf = TFIDF(idf, 3)

        e = tfidf.to_array()
        self.assertEqual(tfidf.global_scheme.documents, TFIDF.from_array(e).global_scheme.documents)
        self.assertEqual(tfidf.global_scheme.idf, TFIDF.from_array(e).global_scheme.idf)
        self.assertEqual(tfidf.local_scheme.__dict__, TFIDF.from_array(e).local_scheme.__dict__)
        self.assertEqual(tfidf.global_scheme.__dict__, TFIDF.from_array(e).global_scheme.__dict__)
