"""
Run unit tests on the Vector class
"""

import math
import os
import sys
import unittest

path = os.path.dirname(__file__)
path = os.path.join(path, '..')
if path not in sys.path:
    sys.path.append(path)

from vector import Vector, VectorSpace
from vector_math import *

class TestVector(unittest.TestCase):
    """
    Test the Vector class.
    """

    def test_empty_constructor(self):
        """
        Test providing an empty constructor.
        """

        v = Vector()
        self.assertEqual({ }, v.dimensions)

    def test_constructor(self):
        """
        Test providing the dimensions to the vector constructor.
        """

        v = Vector({"x": 2, "y": 1})
        self.assertEqual({"x": 2, "y": 1}, v.dimensions)

    def test_dimensions(self):
        """
        Test setting and getting dimensions.
        """

        v = Vector({"x": 2, "y": 1})
        del v.dimensions["x"]
        self.assertEqual({"y": 1}, v.dimensions)

        v.dimensions["x"] = 1
        self.assertEqual({"x": 1, "y": 1}, v.dimensions)

    def test_normalization(self):
        """
        Test normalizing vectors.
        """

        v = Vector({"x": 3, "y": 1.2, "z": -2})
        v.normalize()
        self.assertEqual({ "x": 0.789474, "y": 0.315789, "z": -0.526316 }, { key: round(value, 6) for key, value in v.dimensions.items() })

    def test_double_normalization(self):
        """
        Test that normalizing the same vector twice returns the same vector as when normalized once.
        """

        v = Vector({"x": 3, "y": 1.2, "z": -2})
        v.normalize()
        w = v.copy()
        w.normalize()
        self.assertEqual(v.dimensions, w.dimensions)

    def test_normalize_empty_vector(self):
        """
        Test that when normalizing an empty vector, the resulting vector is also empty.
        """

        v = Vector({ })
        v.normalize()
        self.assertEqual({ }, v.dimensions)

    def test_normalize_zero_length_vector(self):
        """
        Test that when normalizing a vector with a zero length, the resulting vector is also empty.
        """

        v = Vector({ 'x': 0 })
        v.normalize()
        self.assertEqual({ 'x': 0 }, v.dimensions)

    def test_get_dimension(self):
        """
        Test that when getting the value of a dimension, the correct value is returned.
        """

        v = Vector({ 'x': 1 })
        self.assertEqual(1, v.dimensions['x'])

    def test_get_non_existent_dimension(self):
        """
        Test that when getting the value of a dimension that does not exist, 0 is returned.
        """

        v = Vector({ })
        self.assertEqual(0, v.dimensions['x'])

    def test_vector_space_initialization(self):
        """
        Test that when providing no dimensions, an empty vector space is created.
        """

        v = Vector()
        self.assertEqual({ }, v.dimensions)
        self.assertEqual(0, v.dimensions['x'])
        v.dimensions['x'] = 10
        self.assertEqual({ 'x': 10 }, v.dimensions)
        self.assertEqual(10, v.dimensions['x'])

    def test_dimensions_vector_space(self):
        """
        Test that dimensions are created as a vector space.
        """

        v = Vector()
        self.assertEqual(VectorSpace, type(v.dimensions))

    def test_empty_dict_dimensions_vector_space(self):
        """
        Test that dimensions are created as a vector space when given as an empty dictionary.
        """

        v = Vector({ })
        self.assertEqual(VectorSpace, type(v.dimensions))

    def test_non_empty_dict_dimensions_vector_space(self):
        """
        Test that dimensions are created as a vector space when given as a non-empty dictionary.
        """

        v = Vector({ 'x': 10 })
        self.assertEqual(VectorSpace, type(v.dimensions))

    def test_normalize_vector_space(self):
        """
        Test that when a vector is normalized, its dimensions ae a vecto space.
        """

        v = Vector({ 'x': 10 })
        self.assertEqual(VectorSpace, type(v.dimensions))
        v.normalize()
        self.assertEqual(VectorSpace, type(v.dimensions))

    def test_copy(self):
        """
        Test copying.
        """

        v = Vector({ 'x': 3 })
        n = v.copy()

        self.assertEqual(v.dimensions, n.dimensions)

        v.dimensions['x'] = 2
        self.assertEqual(2, v.dimensions['x'])
        self.assertEqual(3, n.dimensions['x'])
        v.dimensions['x'] = 3

    def test_copy_attributes(self):
        """
        Test that the attributes are also copied.
        """

        v = Vector({ 'x': 3 }, { 'y': True })
        n = v.copy()

        self.assertEqual(v.attributes, n.attributes)

        v.attributes['y'] = False
        self.assertFalse(v.attributes['y'])
        self.assertTrue(n.attributes['y'])
        v.attributes['y'] = True

    def test_export(self):
        """
        Test exporting and importing vectors.
        """

        v = Vector({ 'x': 3 })
        e = v.to_array()
        self.assertEqual(v.attributes, Vector.from_array(e).attributes)
        self.assertEqual(v.dimensions, Vector.from_array(e).dimensions)
        self.assertEqual(v.__dict__, Vector.from_array(e).__dict__)

    def test_export_attributes(self):
        """
        Test that exporting and importing vectors includes attributes.
        """

        v = Vector({ 'x': 3 }, { "y": True })
        e = v.to_array()
        self.assertEqual(v.attributes, Vector.from_array(e).attributes)
        self.assertEqual(v.__dict__, Vector.from_array(e).__dict__)
