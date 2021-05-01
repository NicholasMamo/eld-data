"""
Run unit tests on the cluster node.
"""

import math
import os
import sys
import time
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp.document import Document
from summarization.timeline.nodes import ClusterNode
from vsm.vector import Vector
from vsm.clustering import Cluster

class TestClusterNode(unittest.TestCase):
    """
    Test the cluster node.
    """

    def test_create_empty(self):
        """
        Test that the cluster node is created empty.
        """

        self.assertEqual([ ], ClusterNode(0).clusters)

    def test_create_empty_attributes(self):
        """
        Test that the cluster node is created with no attributes.
        """

        self.assertEqual({ }, ClusterNode(0).attributes)

    def test_create_with_timestamp_zero(self):
        """
        Test that the cluster node saves the timestamp correctly even if it is zero.
        """

        self.assertEqual(0, ClusterNode(0).created_at)

    def test_create_with_timestamp(self):
        """
        Test that the cluster node saves the timestamp correctly.
        """

        self.assertEqual(1000, ClusterNode(1000).created_at)


    def test_create_with_no_clusters(self):
        """
        Test that when creating the cluster node with no clusters, an empty list is initialized.
        """

        node = ClusterNode(0)
        self.assertEqual([ ], node.clusters)

    def test_create_with_clusters(self):
        """
        Test that when creating the cluster node with a list of clusters, it is saved.
        """

        clusters = [ Cluster(), Cluster() ]
        n1 = ClusterNode(0, clusters=clusters[:1])
        self.assertEqual(clusters[:1], n1.clusters)

        n2 = ClusterNode(0, clusters=clusters[1:])
        self.assertEqual(clusters[:1], n1.clusters)
        self.assertEqual(clusters[1:], n2.clusters)

    def test_add(self):
        """
        Test adding a cluster to the node.
        """

        node = ClusterNode(0)
        self.assertEqual([ ], node.clusters)
        cluster = Cluster()
        node.add(cluster)
        self.assertEqual([ cluster ], node.clusters)

    def test_add_repeated(self):
        """
        Test adding clusters one at a time to the node.
        """

        node = ClusterNode(0)
        self.assertEqual([ ], node.clusters)
        clusters = [ Cluster() for i in range(2)]
        node.add(clusters[0])
        self.assertEqual([ clusters[0] ], node.clusters)
        node.add(clusters[1])
        self.assertEqual(clusters, node.clusters)

    def test_add_cluster_dynamic(self):
        """
        Test that when changing a cluster, the node's cluster also changes.
        """

        node = ClusterNode(0)
        self.assertEqual([ ], node.clusters)
        cluster = Cluster()
        node.add(cluster)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(cluster, node.clusters[0])
        self.assertEqual(cluster.vectors, node.clusters[0].vectors)
        document = Document('', { 'a': 1 })
        cluster.vectors.append(document)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(cluster, node.clusters[0])
        self.assertEqual(cluster.vectors, node.clusters[0].vectors)
        self.assertEqual(document, node.clusters[0].vectors[0])

    def test_get_all_documents_empty(self):
        """
        Test that when getting all documents from an empty node, an empty list is returned.
        """

        node = ClusterNode(0)
        self.assertEqual([ ], node.get_all_documents())

    def test_get_all_documents(self):
        """
        Test that when getting all documents, the cluster documents are returned.
        """

        node = ClusterNode(0)
        clusters = [ Cluster(Document('', { })), Cluster(Document('', { })) ]
        self.assertEqual([ ], node.get_all_documents())
        node.add(clusters[0])
        self.assertEqual(clusters[0].vectors, node.get_all_documents())
        node.add(clusters[1])
        self.assertEqual(clusters[0].vectors + clusters[1].vectors, node.get_all_documents())

    def test_similarity_empty_node(self):
        """
        Test that the similarity between a cluster and an empty cluster node, the similarity is 0.
        """

        node = ClusterNode(0)
        self.assertEqual([ ], node.clusters)
        self.assertEqual(0, node.similarity(Cluster(Document('', { 'x': 1 }))))

    def test_similarity_empty_cluster(self):
        """
        Test that the similarity between a node and an empty cluster, the similarity is 0.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]
        cluster = Cluster(documents)

        node = ClusterNode(0)
        node.add(cluster)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(0, node.similarity(Cluster()))

    def test_similarity(self):
        """
        Test calculating the similarity between a node and a cluster.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = ClusterNode(0)
        cluster = Cluster(documents)
        node.add(cluster)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(math.sqrt(2)/2., node.similarity(Cluster(Document('this is not a pipe', { 'pipe': 1 }))))

    def test_similarity_lower_bound(self):
        """
        Test that the similarity lower-bound between a node and a cluster is 0.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = ClusterNode(0)
        cluster = Cluster(documents)
        node.add(cluster)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(0, node.similarity(Cluster(Document('this is a picture of dorian gray', { 'picture': 1, 'dorian': 1, 'gray': 1 }))))

    def test_similarity_upper_bound(self):
        """
        Test that the similarity upper-bound between a node and a cluster is 1.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = ClusterNode(0)
        cluster = Cluster(documents)
        node.add(cluster)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(1, node.similarity(Cluster(Document('this is not a pipe and this is not a cigar', { 'cigar': 1, 'pipe': 1 }))))

    def test_similarity_max(self):
        """
        Test that the returned similarity is the maximum between the cluster and the node's clusters.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]
        document = Document('this is a picture of dorian gray', { 'picture': 1, 'dorian': 1, 'gray': 1 })

        node = ClusterNode(0)
        cluster = Cluster(documents)
        node.add(cluster)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(0, node.similarity(Cluster(document)))
        node.add(Cluster(document))
        self.assertEqual(1, round(node.similarity(Cluster(document)), 10))

        """
        Reverse the procedure.
        """

        node = ClusterNode(0)
        cluster = Cluster(document)
        node.add(cluster)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(1, round(node.similarity(Cluster(document)), 10))
        node.add(Cluster(documents))
        self.assertEqual(1, round(node.similarity(Cluster(document)), 10))

    def test_expired_inclusive(self):
        """
        Test that the expiry is inclusive.
        """

        node = ClusterNode(created_at=1000)
        self.assertTrue(node.expired(10, 1010))

    def test_expired_far_timestamp(self):
        """
        Test that a node is expired if the timestamp is sufficiently far.
        """

        node = ClusterNode(created_at=1000)
        self.assertTrue(node.expired(10, 1011))

    def test_expired_close_timestamp(self):
        """
        Test that a node is not expired if the timestamp is close.
        """

        node = ClusterNode(created_at=1000)
        self.assertFalse(node.expired(10, 1001))

    def test_expired_past_timestamp(self):
        """
        Test that a node is not expired if the timestamp is in the past.
        """

        node = ClusterNode(created_at=1000)
        self.assertFalse(node.expired(10, 999))

    def test_expired_realtime(self):
        """
        Test that when the timestamp is not given, the current timestamp is used.
        """

        node = ClusterNode(created_at=time.time())
        self.assertFalse(node.expired(1, time.time()))

    def test_expired_realtime_sleep(self):
        """
        Test that when the timestamp is not given, the current timestamp is used.
        """

        node = ClusterNode(created_at=time.time())
        time.sleep(1)
        self.assertTrue(node.expired(1, time.time()))

    def test_expired_zero(self):
        """
        Test that a node immediately expired if the expiry is 0.
        """

        node = ClusterNode(created_at=1000)
        self.assertTrue(node.expired(0, 1000))

    def test_expired_negative(self):
        """
        Test that a ValueError is raised when the expiry is negative.
        """

        node = ClusterNode(created_at=1000)
        self.assertRaises(ValueError, node.expired, -1, 0)

    def test_export(self):
        """
        Test exporting and importing cluster nodes.
        """

        node = ClusterNode(0)
        e = node.to_array()
        self.assertEqual(node.created_at, ClusterNode.from_array(e).created_at)
        self.assertEqual(node.clusters, ClusterNode.from_array(e).clusters)
        self.assertEqual(node.__dict__, ClusterNode.from_array(e).__dict__)

    def test_export_with_clusters(self):
        """
        Test exporting cluster nodes that have clusters.
        """

        clusters = [ Cluster(Document('', { 'a': 1 }), attributes={ 'b': 2 }),
                      Cluster(Vector({ 'c': 3 }), attributes={ 'd': 4 }) ]
        node = ClusterNode(0, clusters=clusters)
        e = node.to_array()
        self.assertEqual(node.created_at, ClusterNode.from_array(e).created_at)
        self.assertTrue(all(cluster['class'] == "<class 'vsm.clustering.cluster.Cluster'>" for cluster in e['clusters']))
        self.assertEqual(clusters[0].vectors[0].dimensions, e['clusters'][0]['vectors'][0]['dimensions'])
        self.assertEqual({ 'b': 2 }, e['clusters'][0]['attributes'])
        self.assertEqual(clusters[1].vectors[0].dimensions, e['clusters'][1]['vectors'][0]['dimensions'])
        self.assertEqual({ 'd': 4 }, e['clusters'][1]['attributes'])

    def test_import(self):
        """
        Test importing cluster nodes that have clusters.
        """

        clusters = [ Cluster(Document('', { 'a': 1 }), attributes={ 'b': 2 }),
                      Cluster(Vector({ 'c': 3 }), attributes={ 'd': 4 }) ]
        node = ClusterNode(0, clusters=clusters)
        e = node.to_array()
        i = ClusterNode.from_array(e)
        self.assertEqual(node.created_at, i.created_at)
        self.assertTrue(all(type(cluster) is Cluster for cluster in i.clusters))
        self.assertEqual(clusters[0].centroid.dimensions, i.clusters[0].centroid.dimensions)
        self.assertEqual(clusters[0].attributes, i.clusters[0].attributes)
        self.assertEqual(clusters[1].centroid.dimensions, i.clusters[1].centroid.dimensions)
        self.assertEqual(clusters[1].attributes, i.clusters[1].attributes)
        self.assertEqual(Document, type(i.clusters[0].vectors[0]))
        self.assertEqual(Vector, type(i.clusters[1].vectors[0]))

    def test_merge_none(self):
        """
        Test that when merging no cluster nodes, the merge function returns a new, empty node.
        """

        node = ClusterNode.merge(10)
        self.assertEqual(10, node.created_at)
        self.assertFalse(node.clusters)
        self.assertEqual(ClusterNode, type(node))

    def test_merge_created_at_from_parameter(self):
        """
        Test that when merging cluster nodes, the timestamp is taken from the parameters.
        """

        clusters = [ Cluster(vectors=[ Document('', { 'a': 1 }), Document('', { 'b': 2 }) ]) ]
        _node = ClusterNode(created_at=0, clusters=clusters)
        node = ClusterNode.merge(10)
        self.assertEqual(10, node.created_at)

    def test_merge_one(self):
        """
        Test that when merging one cluster node, a new node with the same clusters is returned.
        """

        clusters = [ Cluster(vectors=[ Document('', { 'a': 1 }), Document('', { 'b': 2 }) ]) ]
        _node = ClusterNode(created_at=0, clusters=clusters)
        node = ClusterNode.merge(10, _node)
        self.assertEqual(_node.clusters, node.clusters)

    def test_merge_all(self):
        """
        Test that when merging cluster nodes, all of the clusters and their documents are present in the new node.
        """

        documents = [ Document('', { 'a': 1 }), Document('', { 'b': 2 }),
                       Document('', { 'c': 3 }), Document('', { 'd': 4 }) ]
        clusters = [ Cluster(vectors=documents[:2]),
                      Cluster(vectors=documents[2:]) ]
        nodes = [ ClusterNode(created_at=0, clusters=clusters[:1]),
                    ClusterNode(created_at=0, clusters=clusters[1:]) ]
        node = ClusterNode.merge(10, nodes[0], nodes[1])
        self.assertTrue(all( cluster in node.clusters for node in nodes
                                                       for cluster in node.clusters ))
        self.assertTrue(all( document in documents for node in nodes
                                                    for cluster in node.clusters
                                                   for document in cluster.vectors ))

    def test_merge_order(self):
        """
        Test that when merging cluster nodes, the order of the clusters and their documents is the same as the order of the nodes.
        """

        documents = [ Document('', { 'a': 1 }), Document('', { 'b': 2 }),
                       Document('', { 'c': 3 }), Document('', { 'd': 4 }) ]
        clusters = [ Cluster(vectors=documents[:2]),
                      Cluster(vectors=documents[2:]) ]
        nodes = [ ClusterNode(created_at=0, clusters=clusters[:1]),
                    ClusterNode(created_at=0, clusters=clusters[1:]) ]
        node = ClusterNode.merge(10, nodes[0], nodes[1])
        self.assertEqual(clusters, node.clusters)
        _documents = [ document for node in nodes
                                for cluster in node.clusters
                                for document in cluster.vectors ]
        self.assertEqual(documents, _documents)
