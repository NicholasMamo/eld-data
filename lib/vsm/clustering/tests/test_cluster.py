"""
Run unit tests on the Cluster class
"""

import math
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp.document import Document
from nlp.weighting.tf import TF
from vsm import Vector, VectorSpace
from vsm import vector_math
from vsm.clustering.cluster import Cluster

class TestCluster(unittest.TestCase):
    """
    Test the Cluster class
    """

    def test_empty_cluster(self):
        """
        Test that an empty cluster has a centroid with no dimensions.
        """

        c = Cluster()
        self.assertEqual({ }, c.centroid.dimensions)

    def test_cluster_with_one_vector(self):
        """
        Test that the centroid of a cluster with a single vector has an equivalent centroid.
        """

        v = Document("a", ["a", "b", "a", "c"], scheme=TF())
        v.normalize()
        c = Cluster(v)
        self.assertEqual(v.dimensions, c.centroid.dimensions)

    def test_cluster_with_several_vectors(self):
        """
        Test creating a cluster with several vectors.
        """

        v = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF()),
        ]
        for vector in v:
            vector.normalize()

        c = Cluster(v)
        self.assertEqual(v, c.vectors)

    def test_cluster_with_several_vectors_copy(self):
        """
        Test that when creating a cluster with several vectors, a copy is created.
        """

        v = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF()),
        ]
        for vector in v:
            vector.normalize()

        c = Cluster(v)
        self.assertEqual(v, c.vectors)
        copy = list(v)
        c.vectors.remove(v[0])
        self.assertEqual([ v[1] ], c.vectors)
        self.assertEqual(copy, v)
        self.assertEqual(2, len(v))

    def test_add_vectors(self):
        """
        Test adding vectors to a cluster gradually.
        """

        c = Cluster()
        v = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF())
        ]

        self.assertEqual({}, c.centroid.dimensions)

        c.vectors.append(v[0])
        self.assertEqual([ v[0] ], c.vectors)

        c.vectors.append(v[1])
        self.assertEqual(v, c.vectors)

    def test_remove_vectors(self):
        """
        Test removing vectors from a cluster gradually.
        """

        v = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF())
        ]
        c = Cluster(v)
        c.vectors.remove(v[0])
        self.assertEqual([ v[1] ], c.vectors)

        c = Cluster(v)
        c.vectors.remove(v[1])
        self.assertEqual([ v[0] ], c.vectors)
        c.vectors.remove(v[0])
        self.assertEqual([ ], c.vectors)

    def test_setting_vectors(self):
        """
        Test setting the vectors manually.
        """

        v = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF())
        ]
        c = Cluster()
        self.assertEqual({ }, c.centroid.dimensions)
        c.vectors = v
        self.assertEqual(v, c.vectors)

    def test_cluster_similarity(self):
        """
        Test calculating the similarity between a cluster and a new vector.
        """

        v = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF())
        ]
        c = Cluster(v)

        n = Document("", ["a", "b"], scheme=TF())
        self.assertEqual(round((1.5 + 0.5)/(math.sqrt(2) * math.sqrt(1.5 ** 2 + 0.5 ** 2 + 1)), 5), round(c.similarity(n), 5))

        c.vectors.remove(v[1])
        self.assertEqual(round(3/(math.sqrt(2) * math.sqrt(2**2 + 1 + 1)), 5), round(c.similarity(n), 5))

    def test_empty_cluster_similarity(self):
        """
        Test that when calculating the similarity between a vector and an empty cluster, the similarity is 0.
        """

        c = Cluster()
        v = Document("", ["a", "c"], scheme=TF())
        self.assertEqual(0, c.similarity(v))

    def test_get_centroid(self):
        """
        Test getting the centroid.
        """

        v = Document("", ["a", "c"], scheme=TF())
        v.normalize()
        c = Cluster(v)
        self.assertTrue(all(round(v.dimensions[dimension], 10) == round(c.centroid.dimensions[dimension], 10)
                            for dimension in v.dimensions.keys() | c.centroid.dimensions))

    def test_centroid_normalized(self):
        """
        Test that the centroid is normalized.
        """

        v = Document("", ["a", "c"], scheme=TF())
        c = Cluster(v)
        self.assertEqual(1, round(vector_math.magnitude(c.centroid), 10))

    def test_centroid_normalized_several_vectors(self):
        """
        Test that the centroid is always normalized.
        """

        v = Document("", ["a", "c"], scheme=TF())
        c = Cluster(v)
        self.assertEqual(1, round(vector_math.magnitude(c.centroid), 10))
        c.vectors.append(Document("", ["a", "b", "a", "d"]))
        self.assertEqual(1, round(vector_math.magnitude(c.centroid), 10))

    def test_recalculate_centroid(self):
        """
        Test when a vector changes, and the centroid is re-calculated, it is correct.
        """

        v = [ Document("", [ ]), Document("", [ ]) ]
        c = Cluster(v)
        self.assertEqual({ }, c.centroid.dimensions)

        v[0].dimensions = { 'a': 1, 'b': 1 }
        self.assertEqual(VectorSpace, type(v[0].dimensions))
        self.assertEqual(round(math.sqrt(2)/2., 10), round(c.centroid.dimensions['a'], 10))
        self.assertEqual(round(math.sqrt(2)/2., 10), round(c.centroid.dimensions['b'], 10))
        self.assertEqual(1, round(vector_math.magnitude(c.centroid), 10))

        v[1].dimensions = { 'a': 1 }
        self.assertEqual(VectorSpace, type(v[1].dimensions))
        self.assertEqual(round(1./math.sqrt(1 ** 2 + 0.5 ** 2), 10), round(c.centroid.dimensions['a'], 10))
        self.assertEqual(round(0.5/math.sqrt(1 ** 2 + 0.5 ** 2), 10), round(c.centroid.dimensions['b'], 10))
        self.assertEqual(1, round(vector_math.magnitude(c.centroid), 10))

    def test_set_vectors_none(self):
        """
        Test that setting vectors to ``None`` overwrites existing vectors.
        """

        v = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF())
        ]
        c = Cluster(v)
        self.assertEqual(v, c.vectors)

        c.vectors = None
        self.assertEqual([ ], c.vectors)
        self.assertEqual({ }, c.centroid.dimensions)

    def test_set_one_vectors(self):
        """
        Test that setting vectors to a single vector overwrites existing vectors.
        """

        v = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF())
        ]
        c = Cluster(v)
        self.assertEqual(v, c.vectors)

        n = Document("", [ 'a' ], scheme=TF())
        c.vectors = n
        self.assertEqual([ n ], c.vectors)
        self.assertEqual(n.dimensions, c.centroid.dimensions)

    def test_set_several_vectors(self):
        """
        Test that setting vectors to several vectors overwrites existing vectors.
        """

        v = Document("", [ 'a' ], scheme=TF())
        c = Cluster(v)
        self.assertEqual([ v ], c.vectors)
        self.assertEqual(v.dimensions, c.centroid.dimensions)

        n = [
            Document("", ["a", "b", "a", "c"], scheme=TF()),
            Document("", ["a", "c"], scheme=TF())
        ]

        c.vectors = n
        self.assertEqual(n, c.vectors)

    def test_get_representative_vector(self):
        """
        Test ranking the vectors according to their similarity to the cluster.
        """

        v = [
            Document("", [ 'a', 'b', 'c' ], scheme=TF()),
            Document("", [ 'a', 'a', 'c' ], scheme=TF()),
            Document("", [ 'p' ], scheme=TF()),
        ]
        c = Cluster(v)
        self.assertEqual(Document, type(c.get_representative_vectors(1)))
        self.assertEqual(v[1], c.get_representative_vectors(1))

    def test_get_representative_vectors(self):
        """
        Test ranking the vectors according to their similarity to the cluster.
        """

        v = [
            Document("", [ 'a', 'b', 'c' ], scheme=TF()),
            Document("", [ 'a', 'a', 'c' ], scheme=TF()),
            Document("", [ 'p' ], scheme=TF()),
        ]
        c = Cluster(v)
        self.assertEqual(list, type(c.get_representative_vectors(2)))
        self.assertEqual([ v[1], v[0] ], c.get_representative_vectors(2))

    def test_get_representative_vectors_from_empty_cluster(self):
        """
        Test that when getting the representative vectors from an empty cluster, an empty list is returned.
        """

        c = Cluster()
        self.assertEqual(list, type(c.get_representative_vectors(2)))
        self.assertEqual([ ], c.get_representative_vectors(2))

    def test_get_representative_vector_from_empty_cluster(self):
        """
        Test that when getting the representative vector from an empty cluster, ``None`` is returned.
        """

        c = Cluster()
        self.assertEqual(None, c.get_representative_vectors(1))

    def test_intra_similarity_of_empty_cluster(self):
        """
        Test that the intra-similarity of an empty cluster is 0.
        """

        c = Cluster()
        self.assertEqual(0, c.get_intra_similarity())

    def test_intra_similarity_of_cluster_with_single_vector(self):
        """
        Test that the intra-similarity of a cluster with a single vector is equivalent to that vector's similarity with the cluster.
        """

        v = Document("", [ 'a', 'b' ], scheme=TF())
        c = Cluster(v)
        self.assertEqual(c.similarity(v), c.get_intra_similarity())

    def test_intra_similarity_of_cluster(self):
        """
        Test that the intra-similarity of a cluster with several vectors is equivalent to the average similarity.
        """

        v = [
            Document("", [ 'a', 'b' ], scheme=TF()),
            Document("", [ 'a', 'a' ], scheme=TF()),
        ]
        c = Cluster(v)
        self.assertEqual((c.similarity(v[0]) + c.similarity(v[1]))/2., c.get_intra_similarity())

    def test_size_empty_cluster(self):
        """
        Test that the size of an empty cluster is 0.
        """

        c = Cluster()
        self.assertEqual(0, c.size())

    def test_size(self):
        """
        Test retrieving the size of a cluster.
        """

        v = [
            Document("", [ 'a', 'b' ], scheme=TF()),
            Document("", [ 'a', 'a' ], scheme=TF()),
        ]
        c = Cluster(v)
        self.assertEqual(len(v), c.size())

    def test_export_documents(self):
        """
        Test that when importing documents, the correct class is imported.
        """

        tf = TF()
        v = [ Document("a", ["a", "b", "a", "c"], scheme=TF()),
              Document("b", ["a", "c"], scheme=TF()),
              Document("c", ["b"], scheme=TF()), ]
        c = Cluster(v)

        e = c.to_array()
        r = Cluster.from_array(e)

        self.assertTrue(all( imported.__dict__ == exported.__dict__
                            for imported, exported in zip(v, r.vectors) ))
        self.assertTrue(all( Document == type(exported)
                            for imported, exported in zip(v, r.vectors) ))

    def test_export_vectors(self):
        """
        Test that when importing vectors, the correct class is imported.
        """

        v = [ Vector({ 'a': 1, 'b': 1, 'a': 1, 'c': 1}),
              Vector({ 'a': 1, 'c': 1 }), Vector({ 'b': 1 }), ]
        c = Cluster(v)

        e = c.to_array()
        r = Cluster.from_array(e)

        self.assertTrue(all( imported.__dict__ == exported.__dict__
                            for imported, exported in zip(v, r.vectors) ))
        self.assertTrue(all( Vector == type(exported)
                            for imported, exported in zip(v, r.vectors) ))

    def test_export_attributes(self):
        """
        Test that when exporting and importing clusters, the attributes are included.
        """

        c = Cluster([ ], attributes={ 'a': 1 })
        e = c.to_array()
        self.assertEqual("<class 'vsm.clustering.cluster.Cluster'>", e['class'])
        self.assertEqual(c.attributes, Cluster.from_array(e).attributes)
        self.assertEqual(c.centroid.__dict__, Cluster.from_array(e).centroid.__dict__)
