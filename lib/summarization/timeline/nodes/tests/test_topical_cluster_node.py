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
from summarization.timeline.nodes import TopicalClusterNode
from vsm import vector_math
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

        self.assertEqual([ ], TopicalClusterNode(0).clusters)
        self.assertEqual([ ], TopicalClusterNode(0).topics)

    def test_create_empty_attributes(self):
        """
        Test that the topical cluster node is created with no attributes.
        """

        self.assertEqual({ }, TopicalClusterNode(0).attributes)

    def test_create_with_timestamp_zero(self):
        """
        Test that the cluster node saves the timestamp correctly even if it is zero.
        """

        self.assertEqual(0, TopicalClusterNode(0).created_at)

    def test_create_with_timestamp(self):
        """
        Test that the cluster node saves the timestamp correctly.
        """

        self.assertEqual(1000, TopicalClusterNode(1000).created_at)

    def test_create_with_no_clusters(self):
        """
        Test that when creating the topical cluster node with no clusters, an empty list is initialized.
        """

        node = TopicalClusterNode(0)
        self.assertEqual([ ], node.clusters)

    def test_create_with_clusters(self):
        """
        Test that when creating the topical cluster node with a list of clusters and topics, they are saved.
        """

        clusters = [ Cluster(), Cluster() ]
        topics = [ Vector(), Vector() ]
        n1 = TopicalClusterNode(0, clusters=clusters[:1], topics=topics[:1])
        self.assertEqual(clusters[:1], n1.clusters)
        self.assertEqual(topics[:1], n1.topics)

        n2 = TopicalClusterNode(0, clusters=clusters[1:], topics=topics[1:])
        self.assertEqual(clusters[:1], n1.clusters)
        self.assertEqual(topics[:1], n1.topics)
        self.assertEqual(clusters[1:], n2.clusters)
        self.assertEqual(topics[1:], n2.topics)

    def test_create_with_unequal_clusters_topics(self):
        """
        Test that when creating the topical cluster node with an unequal number of clusters and topics, a ValueError is raised.
        """

        clusters = [ Cluster(), Cluster() ]
        topics = [ Vector(), Vector() ]
        self.assertRaises(ValueError, TopicalClusterNode, 0, clusters=clusters[:1])
        self.assertRaises(ValueError, TopicalClusterNode, 0, topics=topics[:1])
        self.assertRaises(ValueError, TopicalClusterNode, 0, clusters=clusters, topics=topics[:1])
        self.assertRaises(ValueError, TopicalClusterNode, 0, clusters=clusters[:1], topics=topics)

    def test_add(self):
        """
        Test adding a cluster to the node.
        """

        node = TopicalClusterNode(0)
        self.assertEqual([ ], node.clusters)
        self.assertEqual([ ], node.topics)
        cluster = Cluster()
        topic = Vector()
        node.add(cluster, topic)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual([ topic ], node.topics)

    def test_add_repeated(self):
        """
        Test adding clusters one at a time to the node.
        """

        node = TopicalClusterNode(0)
        self.assertEqual([ ], node.clusters)
        clusters = [ Cluster() for i in range(2)]
        topics = [ Vector() for i in range(2)]
        node.add(clusters[0], topics[0])
        self.assertEqual([ clusters[0] ], node.clusters)
        self.assertEqual([ topics[0] ], node.topics)
        node.add(clusters[1], topics[1])
        self.assertEqual(clusters, node.clusters)
        self.assertEqual(topics, node.topics)

    def test_add_cluster_dynamic(self):
        """
        Test that when changing a topic, the node's topic also changes.
        """

        node = TopicalClusterNode(0)
        self.assertEqual([ ], node.topics)
        topic = Vector()
        node.add(Cluster(), topic)
        self.assertEqual(topic, node.topics[0])
        self.assertEqual({ }, node.topics[0].dimensions)

        topic.dimensions['a'] = 1
        self.assertEqual({ 'a': 1 }, node.topics[0].dimensions)
        self.assertEqual(topic, node.topics[0])

    def test_get_all_documents(self):
        """
        Test that when getting all documents, the cluster documents are returned.
        """

        node = TopicalClusterNode(0)
        clusters = [ Cluster(Document('', { })), Cluster(Document('', { })) ]
        self.assertEqual([ ], node.get_all_documents())
        node.add(clusters[0], Vector())
        self.assertEqual(clusters[0].vectors, node.get_all_documents())
        node.add(clusters[1], Vector())
        self.assertEqual(clusters[0].vectors + clusters[1].vectors, node.get_all_documents())

    def test_similarity_empty_node(self):
        """
        Test that the similarity between a cluster and an empty cluster node, the similarity is 0.
        """

        node = TopicalClusterNode(0)
        self.assertEqual([ ], node.clusters)
        self.assertEqual(0, node.similarity(Cluster(Document('', { 'x': 1 })), Vector()))

    def test_similarity_empty_cluster(self):
        """
        Test that the similarity between a node and an empty topic, the similarity is 0.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]
        cluster = Cluster(documents)

        node = TopicalClusterNode(0)
        node.add(cluster, cluster.centroid)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual([ cluster.centroid ], node.topics)
        self.assertEqual(0, node.similarity(Cluster(), Vector()))

    def test_similarity(self):
        """
        Test calculating the similarity between a node and a topic.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = TopicalClusterNode(0)
        cluster = Cluster(documents)
        node.add(cluster, cluster.centroid)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(math.sqrt(2)/2., node.similarity(Cluster(), Vector({ 'pipe': 1 })))

    def test_similarity_lower_bound(self):
        """
        Test that the similarity lower-bound between a node and a cluster is 0.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = TopicalClusterNode(0)
        cluster = Cluster(documents)
        node.add(cluster, cluster.centroid)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(0, node.similarity(Cluster(), Vector({ 'picture': 1 })))

    def test_similarity_upper_bound(self):
        """
        Test that the similarity upper-bound between a node and a cluster is 1.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = TopicalClusterNode(0)
        cluster = Cluster(documents)
        node.add(cluster, cluster.centroid)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(1, node.similarity(Cluster(), vector_math.normalize(Vector({ 'cigar': 1, 'pipe': 1 }))))

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

        node = TopicalClusterNode(0)
        cluster = Cluster(documents)
        node.add(cluster, cluster.centroid)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(0, node.similarity(Cluster(document), Cluster(document).centroid))
        node.add(Cluster(document), Cluster(document).centroid)
        self.assertEqual(1, round(node.similarity(Cluster(document), Cluster(document).centroid), 10))

        """
        Reverse the procedure.
        """

        node = TopicalClusterNode(0)
        cluster = Cluster(document)
        node.add(cluster, cluster.centroid)
        self.assertEqual([ cluster ], node.clusters)
        self.assertEqual(1, round(node.similarity(Cluster(document), Cluster(document).centroid), 10))
        node.add(Cluster(documents), Cluster(documents).centroid)
        self.assertEqual(1, round(node.similarity(Cluster(document), Cluster(document).centroid), 10))

    def test_similarity_with_node_topic(self):
        """
        Test that similarity is computed with the node's topic, not the cluster.
        """

        """
        Create the test data.
        """
        document = Document('this is a pipe and this is a cigar', { 'cigar': 1, 'pipe': 1 })

        node = TopicalClusterNode(0)
        cluster = Cluster(document)
        node.add(cluster, Vector({ 'pipe': 1 }))

        self.assertEqual(1, node.similarity(Document('this is a pipe', { 'pipe': 1 }), Vector({ 'pipe': 1 })))
        self.assertEqual(0, node.similarity(Document('this is a cigar', { 'cigar': 1 }), Vector({ 'cigar': 1 })))

    def test_similarity_with_topic(self):
        """
        Test that similarity is computed with the given topic, not the given cluster.
        """

        """
        Create the test data.
        """
        document = Document('this is a pipe and this is a cigar', { 'cigar': 1, 'pipe': 1 })

        node = TopicalClusterNode(0)
        cluster = Cluster(document)
        node.add(cluster, Vector({ 'pipe': 1 }))

        self.assertEqual(1, node.similarity(Document('this is a cigar', { 'cigar': 1 }), Vector({ 'pipe': 1 })))
        self.assertEqual(0, node.similarity(Document('this is a pipe', { 'pipe': 1 }), Vector({ 'cigar': 1 })))

    def test_export(self):
        """
        Test exporting and importing topical cluster nodes.
        """

        node = TopicalClusterNode(0)
        e = node.to_array()
        self.assertEqual(node.created_at, TopicalClusterNode.from_array(e).created_at)
        self.assertEqual(node.clusters, TopicalClusterNode.from_array(e).clusters)
        self.assertEqual(node.topics, TopicalClusterNode.from_array(e).topics)
        self.assertEqual(node.__dict__, TopicalClusterNode.from_array(e).__dict__)

    def test_export_with_clusters(self):
        """
        Test exporting topical cluster nodes that have clusters.
        """

        clusters = [ Cluster(Document('', { 'a': 1 }), attributes={ 'b': 2 }),
                     Cluster(Vector({ 'c': 3 }), attributes={ 'd': 4 }) ]
        topics = [ Vector({ 'p': 1 }, { 'y': 2 }), Vector({ 'q': 2 }, { 'x': 1}) ]
        node = TopicalClusterNode(0, clusters=clusters, topics=topics)
        e = node.to_array()
        self.assertEqual(node.created_at, TopicalClusterNode.from_array(e).created_at)
        self.assertTrue(all(cluster['class'] == "<class 'vsm.clustering.cluster.Cluster'>" for cluster in e['clusters']))
        self.assertEqual(clusters[0].vectors[0].dimensions, e['clusters'][0]['vectors'][0]['dimensions'])
        self.assertEqual({ 'b': 2 }, e['clusters'][0]['attributes'])
        self.assertEqual(clusters[1].vectors[0].dimensions, e['clusters'][1]['vectors'][0]['dimensions'])
        self.assertEqual({ 'd': 4 }, e['clusters'][1]['attributes'])
        self.assertEqual(topics[0].dimensions, e['topics'][0]['dimensions'])
        self.assertEqual(topics[1].dimensions, e['topics'][1]['dimensions'])
        self.assertEqual(topics[0].attributes, e['topics'][0]['attributes'])
        self.assertEqual(topics[1].attributes, e['topics'][1]['attributes'])

    def test_import(self):
        """
        Test importing topical cluster nodes that have clusters.
        """

        clusters = [ Cluster(Document('', { 'a': 1 }), attributes={ 'b': 2 }),
                     Cluster(Vector({ 'c': 3 }), attributes={ 'd': 4 }) ]
        topics = [ Vector({ 'p': 1 }, { 'y': 2 }), Document('text', { 'q': 2 }, attributes={ 'x': 1}) ]
        node = TopicalClusterNode(0, clusters=clusters, topics=topics)
        e = node.to_array()
        i = TopicalClusterNode.from_array(e)
        self.assertEqual(node.created_at, i.created_at)
        self.assertTrue(all(type(cluster) is Cluster for cluster in i.clusters))
        self.assertEqual(clusters[0].centroid.dimensions, i.clusters[0].centroid.dimensions)
        self.assertEqual(clusters[0].attributes, i.clusters[0].attributes)
        self.assertEqual(clusters[1].centroid.dimensions, i.clusters[1].centroid.dimensions)
        self.assertEqual(clusters[1].attributes, i.clusters[1].attributes)
        self.assertEqual(Document, type(i.clusters[0].vectors[0]))
        self.assertEqual(Vector, type(i.clusters[1].vectors[0]))
        self.assertEqual(topics[0].dimensions, i.topics[0].dimensions)
        self.assertEqual(topics[0].attributes, i.topics[0].attributes)
        self.assertEqual(Vector, type(topics[0]))
        self.assertEqual(topics[1].text, i.topics[1].text)
        self.assertEqual(topics[1].dimensions, i.topics[1].dimensions)
        self.assertEqual(topics[1].attributes, i.topics[1].attributes)
        self.assertEqual(Document, type(topics[1]))

    def test_merge_none(self):
        """
        Test that when merging no topical cluster nodes, the merge function returns a new, empty node.
        """

        node = TopicalClusterNode.merge(10)
        self.assertEqual(10, node.created_at)
        self.assertFalse(node.clusters)
        self.assertFalse(node.topics)
        self.assertEqual(TopicalClusterNode, type(node))

    def test_merge_created_at_from_parameter(self):
        """
        Test that when merging topical cluster nodes, the timestamp is taken from the parameters.
        """

        clusters = [ Cluster(vectors=[ Document('', { 'a': 1 }), Document('', { 'b': 2 }) ]) ]
        topics = [ Vector({ 'a': 1 }) ]
        _node = TopicalClusterNode(created_at=0, clusters=clusters, topics=topics)
        node = TopicalClusterNode.merge(10)
        self.assertEqual(10, node.created_at)

    def test_merge_one(self):
        """
        Test that when merging one topical cluster node, a new node with the same clusters and topics is returned.
        """

        clusters = [ Cluster(vectors=[ Document('', { 'a': 1 }), Document('', { 'b': 2 }) ]) ]
        topics = [ Vector({ 'a': 1 }) ]
        _node = TopicalClusterNode(created_at=0, clusters=clusters, topics=topics)
        node = TopicalClusterNode.merge(10, _node)
        self.assertEqual(_node.clusters, node.clusters)

    def no_test_merge_all(self):
        """
        Test that when merging topical cluster nodes, all of the clusters and their documents are present in the new node.
        """

        documents = [ Document('', { 'a': 1 }), Document('', { 'b': 2 }),
                       Document('', { 'c': 3 }), Document('', { 'd': 4 }) ]
        clusters = [ Cluster(vectors=documents[:2]),
                      Cluster(vectors=documents[2:]) ]
        topics = [ Vector({ 'a': 1 }), Vector({ 'b': 1 }) ]
        nodes = [ TopicalClusterNode(created_at=0, clusters=clusters[:1], topics=topics[:1]),
                   TopicalClusterNode(created_at=0, clusters=clusters[1:], topics=topics[1:]) ]
        node = TopicalClusterNode.merge(10, nodes[0], nodes[1])
        self.assertTrue(all( cluster in node.clusters for node in nodes
                                                       for cluster in node.clusters ))
        self.assertTrue(all( topic in node.topics for node in nodes
                                                   for topic in node.topics ))
        self.assertTrue(all( document in documents for node in nodes
                                                    for cluster in node.clusters
                                                   for document in cluster.vectors ))

    def test_merge_order(self):
        """
        Test that when merging topical cluster nodes, the order of the clusters and their documents is the same as the order of the nodes.
        """

        documents = [ Document('', { 'a': 1 }), Document('', { 'b': 2 }),
                       Document('', { 'c': 3 }), Document('', { 'd': 4 }) ]
        clusters = [ Cluster(vectors=documents[:2]),
                      Cluster(vectors=documents[2:]) ]
        topics = [ Vector({ 'a': 1 }), Vector({ 'b': 1 }) ]
        nodes = [ TopicalClusterNode(created_at=0, clusters=clusters[:1], topics=topics[:1]),
                   TopicalClusterNode(created_at=0, clusters=clusters[1:], topics=topics[1:]) ]
        node = TopicalClusterNode.merge(10, nodes[0], nodes[1])
        self.assertEqual(clusters, node.clusters)
        self.assertEqual(topics, node.topics)
        _documents = [ document for node in nodes
                                for cluster in node.clusters
                                for document in cluster.vectors ]
        self.assertEqual(documents, _documents)
