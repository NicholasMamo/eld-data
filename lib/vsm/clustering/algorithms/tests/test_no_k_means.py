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
from vsm.clustering.algorithms.no_k_means import NoKMeans

class TestNoKMeans(unittest.TestCase):
    """
    Test the No-K-Means algorithms.
    """

    def test_update_age(self):
        """
        Test that when updating the age, the cluster attribute is updated.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10)
        cluster.attributes['age'] = 10
        self.assertEqual(10, cluster.attributes['age'])
        algo._update_age(cluster, 1)
        self.assertEqual(11, cluster.attributes['age'])

    def test_update_age_without_previous(self):
        """
        Test that when updating the age, the cluster attribute is updated even if there is no previous value.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10)
        self.assertFalse(cluster.attributes.get('age'))
        algo._update_age(cluster, 1)
        self.assertEqual(1, cluster.attributes['age'])

    def test_update_age_parameter(self):
        """
        Test that when updating the age with a custom parameter, the cluster attribute is updated accordingly.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10)
        cluster.attributes['age'] = 10
        self.assertEqual(10, cluster.attributes['age'])
        algo._update_age(cluster, 12)
        self.assertEqual(22, cluster.attributes['age'])

    def test_to_freeze_low_age(self):
        """
        Test that when the cluster's age is lower than the freeze period, it is not marked as to be frozen.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10)
        cluster.attributes['age'] = 9
        self.assertFalse(algo._to_freeze(cluster))

    def test_to_freeze_same_age(self):
        """
        Test that when the cluster's age is equivalent to the freeze period, it is not marked to be frozen.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10)
        cluster.attributes['age'] = 10
        self.assertFalse(algo._to_freeze(cluster))

    def test_to_freeze_high_age(self):
        """
        Test that when the cluster's age is higher than the freeze period, it is marked to be frozen.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10)
        cluster.attributes['age'] = 11
        self.assertTrue(algo._to_freeze(cluster))

    def test_freeze_inactive_cluster(self):
        """
        Test that when freezing a cluster that is not part of the algorithm, the function raises a ValueError.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10)
        self.assertRaises(ValueError, algo._freeze, cluster)

    def test_frozen_cluster(self):
        """
        Test that a frozen cluster cannot be frozen again.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10)
        algo.clusters.append(cluster)
        algo._freeze(cluster)
        self.assertRaises(ValueError, algo._freeze, cluster)

    def test_frozen_cluster_not_stored(self):
        """
        Test that if the algorithm does not store frozen clusters, the cluster is removed.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10, store_frozen=False)
        algo.clusters.append(cluster)
        algo._freeze(cluster)
        self.assertFalse(cluster in algo.frozen_clusters)

    def test_frozen_cluster_stored(self):
        """
        Test that if the algorithm stores frozen clusters, the cluster is stored.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10, store_frozen=True)
        algo.clusters.append(cluster)
        algo._freeze(cluster)
        self.assertTrue(cluster in algo.frozen_clusters)

    def test_frozen_cluster_not_active(self):
        """
        Test that when a cluster is frozen, it moves from the active clusters to the frozen clusters.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10, store_frozen=True)
        algo.clusters.append(cluster)
        self.assertTrue(cluster in algo.clusters)
        self.assertFalse(cluster in algo.frozen_clusters)
        algo._freeze(cluster)
        self.assertFalse(cluster in algo.clusters)
        self.assertTrue(cluster in algo.frozen_clusters)

    def test_reset_age_without_previous(self):
        """
        Test that when resetting the age of a cluster that has no age, the age is set to 0.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10, store_frozen=True)
        self.assertFalse(cluster.attributes.get('age'))
        algo._reset_age(cluster)
        self.assertEqual(0, cluster.attributes['age'])

    def test_reset_age(self):
        """
        Test that when resetting the age of a cluster that has an age, the age is set to 0.
        """

        cluster = Cluster()
        algo = NoKMeans(0.5, 10, store_frozen=True)
        cluster.attributes['age'] = 10
        self.assertEqual(10, cluster.attributes['age'])
        algo._reset_age(cluster)
        self.assertEqual(0, cluster.attributes['age'])

    def test_closest_cluster_none(self):
        """
        Test that when there are no active clusters, the closest cluster returns ``None``.
        """

        algo = NoKMeans(0.5, 10, store_frozen=True)
        document = Document('', [ 'a', 'b' ])
        document.normalize()
        self.assertEqual(None, algo._closest_cluster(document))

    def test_closest_cluster_excludes_frozen(self):
        """
        Test that when computing the closest cluster, frozen clusters are not considered.
        """

        algo = NoKMeans(0.5, 10, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'a', 'b', 'a', 'c' ]), Document('', [ 'a', 'b', 'a' ]),
            Document('', [ 'x', 'y' ])
        ]
        for document in documents:
            document.normalize()

        c1 = Cluster(documents[:2])
        c2 = Cluster(documents[2:])
        algo.clusters = [ c2 ]
        algo.frozen_clusters = [ c1 ]

        document = Document('', [ 'a', 'b' ])
        document.normalize()
        closest_cluster, similarity = algo._closest_cluster(document)
        self.assertEqual(c2, closest_cluster)
        self.assertEqual(0, similarity)

    def test_closest_cluster(self):
        """
        Test that the closest cluster returns the closest cluster.
        """

        algo = NoKMeans(0.5, 10, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'a', 'b', 'a', 'c' ]), Document('', [ 'a', 'b', 'a' ]),
            Document('', [ 'x', 'y' ])
        ]
        for document in documents:
            document.normalize()

        c1 = Cluster(documents[:2])
        c2 = Cluster(documents[2:])
        algo.clusters = [ c1, c2 ]

        document = Document('', [ 'a', 'b' ])
        document.normalize()
        closest_cluster, similarity = algo._closest_cluster(document)
        self.assertEqual(c1, closest_cluster)
        self.assertGreater(similarity, 0)

    def test_cluster_freeze(self):
        """
        Test that when there is a shift in discourse, old clusters are frozen.
        """

        algo = NoKMeans(0.5, 1, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ]),
            Document('', [ 'a', 'b', 'a', 'c' ]), Document('', [ 'a', 'b', 'a' ]),
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

        algo = NoKMeans(0.5, 4, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ]),
            Document('', [ 'a', 'b', 'a', 'c' ]), Document('', [ 'a', 'b', 'a' ]),
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

        algo = NoKMeans(0.5, 1, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ]),
            Document('', [ 'a', 'b', 'a', 'c' ]), Document('', [ 'a', 'b', 'a' ]),
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

        algo = NoKMeans(0.5, 1, store_frozen=True)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ]),
            Document('', [ 'a', 'b', 'a', 'c' ]), Document('', [ 'a', 'b', 'a' ]),
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

        algo = NoKMeans(0.5, 1, store_frozen=False)

        """
        Create the test data.
        """
        documents = [
            Document('', [ 'x', 'y' ]),
            Document('', [ 'a', 'b', 'a', 'c' ]), Document('', [ 'a', 'b', 'a' ]),
        ]
        for document in documents:
            document.normalize()

        clusters = algo.cluster(documents)
        self.assertEqual(2, len(clusters))
        active = [ cluster for cluster in clusters if 'a' in cluster.centroid.dimensions ]
        self.assertEqual(active, algo.clusters)
        self.assertEqual(1, len(algo.clusters))
        self.assertEqual([ ], algo.frozen_clusters)
