"""
Run unit tests on the No-K-Means algorithms.
"""

import math
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp.document import Document
from vsm.clustering.cluster import Cluster
from vsm.clustering.algorithms.temporal_no_k_means import TemporalNoKMeans

class TestNoKMeans(unittest.TestCase):
    """
    Test the No-K-Means algorithms.
    """

    def test_update_age_without_vector_time(self):
        """
        Test that when updating the age of a cluster, a TypeError is raised when the last vector has no time attribute.
        """

        cluster = Cluster(Document('', [ ]))
        algo = TemporalNoKMeans(0.5, 10)
        self.assertRaises(TypeError, algo._update_age, cluster, 0, 'time')

    def test_update_age_without_vector(self):
        """
        Test that when updating the age of a cluster without vectors, a ValueError is raised.
        """

        cluster = Cluster()
        algo = TemporalNoKMeans(0.5, 10)
        self.assertRaises(IndexError, algo._update_age, cluster, 0, 'time')

    def test_update_age(self):
        """
        Test that when updating the age, the cluster attribute is updated.
        """

        cluster = Cluster(Document('', [ ], attributes={ 'timestamp': 10 }))
        algo = TemporalNoKMeans(0.5, 10)
        cluster.attributes['age'] = 8
        self.assertEqual(8, cluster.attributes['age'])
        algo._update_age(cluster, 23, 'timestamp')
        self.assertEqual(13, cluster.attributes['age'])

    def test_update_age_without_previous(self):
        """
        Test that when updating the age, the cluster attribute is updated even if there is no previous value.
        """

        cluster = Cluster(Document('', [ ], attributes={ 'timestamp': 10 }))
        algo = TemporalNoKMeans(0.5, 10)
        algo._update_age(cluster, 23, 'timestamp')
        self.assertEqual(13, cluster.attributes['age'])

    def test_update_age_most_recent_vector(self):
        """
        Test that when updating the age of a cluster, the most recent vector is used.
        """

        cluster = Cluster([ Document('', [ ], attributes={ 'timestamp': 10 }),
                             Document('', [ ], attributes={ 'timestamp': 8 })])
        algo = TemporalNoKMeans(0.5, 10)
        algo._update_age(cluster, 23, 'timestamp')
        self.assertEqual(15, cluster.attributes['age'])

    def test_cluster_freeze(self):
        """
        Test that when there is a shift in discourse, old clusters are frozen.
        """

        algo = TemporalNoKMeans(0.5, 2, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ], attributes={ 'timestamp': 1 }),
            Document('', [ 'a', 'b', 'a', 'c' ], attributes={ 'timestamp': 4 }),
            Document('', [ 'a', 'b', 'a' ], attributes={ 'timestamp': 5 }),
        ]
        for document in documents:
            document.normalize()

        clusters = algo.cluster(documents)
        self.assertEqual(2, len(clusters))
        self.assertEqual(1, len(algo.clusters))
        self.assertEqual(1, len(algo.frozen_clusters))

    def test_cluster_similar_vectors(self):
        """
        Test that similar vectors cluster together.
        """

        algo = TemporalNoKMeans(0.5, 4, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ], attributes={ 'timestamp': 1 }),
            Document('', [ 'a', 'b', 'a', 'c' ], attributes={ 'timestamp': 4 }),
            Document('', [ 'a', 'b', 'a' ], attributes={ 'timestamp': 5 }),
        ]
        for document in documents:
            document.normalize()

        clusters = algo.cluster(documents)
        self.assertEqual(2, len(clusters))
        cluster_a = [ cluster for cluster in clusters if 'a' in cluster.centroid.dimensions ][0]
        cluster_x = [ cluster for cluster in clusters if 'x' in cluster.centroid.dimensions ][0]
        self.assertTrue(all('a' in document.dimensions for document in cluster_a.vectors))
        self.assertTrue(all('x' in document.dimensions for document in cluster_x.vectors))

    def test_cluster_retun_only_updated(self):
        """
        Test that updated clusters are not returned.
        """

        algo = TemporalNoKMeans(0.5, 4, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ], attributes={ 'timestamp': 1 }),
            Document('', [ 'a', 'b', 'a', 'c' ], attributes={ 'timestamp': 4 }),
            Document('', [ 'a', 'b', 'a' ], attributes={ 'timestamp': 5 }),
        ]
        for document in documents:
            document.normalize()

        clusters = algo.cluster(documents[:1])
        self.assertEqual(1, len(clusters))
        clusters = algo.cluster(documents[1:])
        self.assertEqual(1, len(clusters))

    def test_cluster_return_updated_frozen(self):
        """
        Test that updated clusters that are frozen are still returned.
        """

        algo = TemporalNoKMeans(0.5, 1, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ], attributes={ 'timestamp': 1 }),
            Document('', [ 'a', 'b', 'a', 'c' ], attributes={ 'timestamp': 4 }),
            Document('', [ 'a', 'b', 'a' ], attributes={ 'timestamp': 5 }),
        ]
        for document in documents:
            document.normalize()

        clusters = algo.cluster(documents)
        self.assertEqual(2, len(clusters))
        active = [ cluster for cluster in clusters if 'a' in cluster.centroid.dimensions ]
        frozen = [ cluster for cluster in clusters if 'x' in cluster.centroid.dimensions ]
        self.assertEqual(active, algo.clusters)
        self.assertEqual(frozen, algo.frozen_clusters)

    def test_cluster_return_and_remove_frozen(self):
        """
        Test that when frozen clusters are not stored, they are still returned.
        """

        algo = TemporalNoKMeans(0.5, 1, store_frozen=False)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ], attributes={ 'timestamp': 1 }),
            Document('', [ 'a', 'b', 'a', 'c' ], attributes={ 'timestamp': 4 }),
            Document('', [ 'a', 'b', 'a' ], attributes={ 'timestamp': 5 }),
        ]
        for document in documents:
            document.normalize()

        clusters = algo.cluster(documents)
        self.assertEqual(2, len(clusters))
        active = [ cluster for cluster in clusters if 'a' in cluster.centroid.dimensions ]
        self.assertEqual(active, algo.clusters)
        self.assertEqual(1, len(algo.clusters))
        self.assertEqual([ ], algo.frozen_clusters)

    def test_cluster_temporal(self):
        """
        Test that clustering is temporal.
        """

        algo = TemporalNoKMeans(0.5, 0, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ], attributes={ 'timestamp': 0 }),
            Document('', [ 'a', 'b', 'a', 'c' ], attributes={ 'timestamp': 0 }),
            Document('', [ 'a', 'b', 'a' ], attributes={ 'timestamp': 0 }),
        ]
        for document in documents:
            document.normalize()

        clusters = algo.cluster(documents)
        self.assertEqual(2, len(clusters))
        self.assertEqual(2, len(algo.clusters))
        self.assertEqual([ ], algo.frozen_clusters)

    def test_cluster_chronological(self):
        """
        Test that clustering is chronological.
        """

        algo = TemporalNoKMeans(0.5, 2, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'a', 'b', 'a', 'c' ], attributes={ 'timestamp': 1 }),
            Document('', [ 'x', 'y' ], attributes={ 'timestamp': 5 }),
            Document('', [ 'a', 'b', 'a' ], attributes={ 'timestamp': 3 }),
        ]
        for document in documents:
            document.normalize()

        clusters = algo.cluster(documents)
        self.assertEqual(2, len(clusters))
        self.assertEqual(2, len(algo.clusters))
        self.assertEqual([ ], algo.frozen_clusters)
        cluster_a = [ cluster for cluster in clusters if 'a' in cluster.centroid.dimensions ][0]
        self.assertEqual(set([documents[0], documents[2]]), set(cluster_a.vectors))
