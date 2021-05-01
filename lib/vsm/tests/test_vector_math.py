"""
Run unit tests on vector mathematics
"""

import math
import os
import sys
import unittest

path = os.path.dirname(__file__)
path = os.path.join(path, '..')
if path not in sys.path:
    sys.path.append(path)

from vector import Vector
from vector_math import *

class TestVectorMath(unittest.TestCase):
    """
    Test the vector math functions.
    """

    def test_normalization(self):
        """
        Test the normalization method.
        """

        v = Vector({ "x": 3, "y": 2, "z": 1 })
        self.assertEqual(math.sqrt(14), magnitude(v))
        self.assertEqual({"x": 3./math.sqrt(14), "y": math.sqrt(4./14), "z": math.sqrt(1./14)}, normalize(v).dimensions)
        self.assertEqual(1, magnitude(normalize(v)))

    def test_normalize_empty_vector(self):
        """
        Test that normalizing an empty vector returns the same empty vector.
        """

        v = Vector({ })
        self.assertEqual({ }, normalize(v).dimensions)

    def test_augmented_normalization(self):
        """
        Test the augmented normalization method.
        """

        v = Vector({ "x": 1, "y": 2, "z": 0 })
        self.assertEqual({ "x": 0.75, "y": 1., "z": 0.5 }, augmented_normalize(v, 0.5).dimensions)
        self.assertEqual({ "x": 0.6, "y": 1, "z": 0.2 }, { dimension: round(value, 5) for dimension, value in augmented_normalize(v, 0.2).dimensions.items() })
        self.assertEqual({ "x": 0.5, "y": 1, "z": 0 }, { dimension: round(value, 5) for dimension, value in augmented_normalize(v, 0).dimensions.items() })

    def test_augmented_normalization_bounds(self):
        """
        Test that the augmented normalization's augmentation is bound between 0 and 1.
        """

        v = Vector({ "x": 1, "y": 2, "z": 0 })
        augmented_normalize(v, a=0)
        augmented_normalize(v, a=1)
        self.assertRaises(ValueError, augmented_normalize, v, a=-0.1)
        self.assertRaises(ValueError, augmented_normalize, v, a=1.1)

    def test_augmented_normalize_empty_vector(self):
        """
        Test that augmented normalizing an empty vector returns the same empty vector.
        """

        v = Vector({ })
        self.assertEqual(augmented_normalize(v).dimensions, { })

    def test_concatenation(self):
        """
        Test the concatenation function.
        """

        v = Vector()
        self.assertEqual({}, concatenate([v]).dimensions)

        vectors = [
            Vector({ "a": 1, "b": 2, "c": 3 }),
            Vector({ "c": 2, "d": 1 })
        ]
        self.assertEqual({ "a": 1, "b": 2, "c": 5, "d": 1 }, concatenate(vectors).dimensions)

    def test_euclidean(self):
        """
        Test the Euclidean distance.
        """

        v1, v2 = Vector({"x": -1, "y": 2, "z": 3}), Vector({"x": 4, "z": -3})
        self.assertEqual(math.sqrt(65), euclidean(v1, v2))

    def test_euclidean_same_vector(self):
        """
        Test that the Euclidean distance between the same vectors is 0.
        """

        v1 = Vector({ "x": 1, "y": 1})
        self.assertEqual(0, euclidean(v1, v1))

    def test_euclidean_symmetry(self):
        """
        Test that the Euclidean distance is symmetrical.
        """

        v1, v2 = Vector({"x": -1, "y": 2, "z": 3}), Vector({"x": 4, "z": -3})
        self.assertEqual(euclidean(v1, v2), euclidean(v2, v1))

    def test_manhattan(self):
        """
        Test the Manhattan distance.
        """

        v1, v2 = Vector({"x": -1, "y": 2, "z": 3}), Vector({"x": 4, "z": -3})
        self.assertEqual(13, manhattan(v1, v2))

    def test_manhattan_same_vector(self):
        """
        Test that the Manhattan distance of the same vector is 0.
        """

        v1 = Vector({"x": 1, "y": 1})
        self.assertEqual(0, manhattan(v1, v1))

    def test_manhattan(self):
        """
        Test that the Manhattan distance is symmetrical.
        """

        v1, v2 = Vector({"x": -1, "y": 2, "z": 3}), Vector({"x": 4, "z": -3})
        self.assertEqual(manhattan(v1, v2), manhattan(v2, v1))

    def test_cosine(self):
        """
        Test the cosine similarity.
        """

        v1, v2 = Vector({"x": 1/3.74, "y": 2/3.74, "z": 3/3.74}), Vector({"x": 4/8.77, "y": -5/8.77, "z": 6/8.77})
        self.assertEqual(0.37, round(cosine(v1, v2), 2))

    def test_cosine_symmetrical(self):
        """
        Test that the cosine similarity is symmetrical.
        """

        v1, v2 = Vector({"x": 1/3.74, "y": 2/3.74, "z": 3/3.74}), Vector({"x": 4/8.77, "y": -5/8.77, "z": 6/8.77})
        self.assertEqual(round(cosine(v1, v2), 2), round(cosine(v2, v1), 2))

    def test_cosine_similarity_same_vector(self):
        """
        Test that the cosine similarity is 1 for the same vector.
        The rounding is there to avoid floating-point errors.
        """

        v1, v2 = Vector({ "x": 1, "y": 1}), Vector({ "x": 1, "y": 1})
        self.assertEqual(1, round(cosine(v1, v1), 4))

    def test_cosine_similarity_same_vector_different_magnitude(self):
        """
        Test that the cosine similarity is 1 for the same vector, even if the magnitude is different.
        The rounding is there to avoid floating-point errors.
        """

        v1, v2 = Vector({ "x": 1, "y": 1}), Vector({ "x": 2, "y": 2})
        self.assertEqual(1, round(cosine(v1, v1), 4))

    def test_cosine_similarity_opposite_vectors(self):
        """
        Test that the cosine similarity of opposite vectors is -1.
        The rounding is there to avoid floating-point errors.
        """

        v1, v2 = Vector({"x": 1/3.74, "y": 2/3.74, "z": 3/3.74}), Vector({"x": -1/3.74, "y": -2/3.74, "z": -3/3.74})
        self.assertEqual(-1, round(cosine(v1, v2), 4))

    def test_cosine_similarity_orthogonal_vectors(self):
        """
        Test that the cosine similarity of orthogonal (ninety-degrees) vectors is 0.
        The rounding is there to avoid floating-point errors.
        """

        v1, v2 = Vector({"x": 2, "y": 4}), Vector({"x": -2, "y": 1})
        self.assertEqual(0, round(cosine(v1, v2), 4))

    def test_cosine_distance(self):
        """
        Test the cosine distance.
        """

        v1, v2 = Vector({"x": 1/3.74, "y": 2/3.74, "z": 3/3.74}), Vector({"x": 4/8.77, "y": -5/8.77, "z": 6/8.77})
        self.assertEqual(1 - 0.37, round(cosine_distance(v1, v2), 2))
