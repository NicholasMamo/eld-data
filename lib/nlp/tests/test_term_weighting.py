"""
Run unit tests on the different term weighting schemes.
"""

import os
import sys
import unittest

path = os.path.dirname(__file__)
path = os.path.join(path, '../')
if path not in sys.path:
    sys.path.append(path)

from libraries.vector.nlp import weighting

class TestTermWeightingSchemes(unittest.TestCase):
    """
    Test the different term weighting scheme classes.
    """

    def test_tf(self):
        """
        Test the TF scheme.
        """

        tokens = ["a", "b", "c", "d", "b"]

        tf = weighting.TF()
        d = tf.create(tokens)
        dimensions = d.get_dimensions()
        self.assertEqual(dimensions.get("a", 0), 1)
        self.assertEqual(dimensions.get("b", 0), 2)
        self.assertEqual(dimensions.get("e", 0), 0)

    def test_tfidf(self):
        """
        Test the TF-IDF scheme.
        """

        idf = {
            "DOCUMENTS": 3,
            "b": 2,
            "a": 1
        }
        tokens = ["a", "b", "c", "d", "b"]

        tfidf = weighting.TFIDF(idf)
        d = tfidf.create(tokens)
        dimensions = d.get_dimensions()
        self.assertEqual(round(dimensions.get("a", 0), 5), 0.47712)
        self.assertEqual(round(dimensions.get("b", 0), 5), 0.35218)
        self.assertEqual(round(dimensions.get("e", 0), 5), 0)
