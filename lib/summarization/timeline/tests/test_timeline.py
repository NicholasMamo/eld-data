"""
Run unit tests on the timeline.
"""

import math
import os
import sys
import time
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp.document import Document
from summarization.timeline import Timeline
from summarization.timeline.nodes import ClusterNode, DocumentNode
from vsm.clustering import Cluster

class TestTimeline(unittest.TestCase):
    """
    Test the timeline.
    """

    def test_create_empty_timeline(self):
        """
        Test that when creating an empty timeline, the list of nodes is empty.
        """

        self.assertEqual([ ], Timeline(ClusterNode, 60, 0.5).nodes)

    def test_create_node_type_cluster_node(self):
        """
        Test that when creating a timeline with a cluster node, the node type is saved.
        """

        self.assertEqual(ClusterNode, Timeline(ClusterNode, 60, 0.5).node_type)

    def test_create_node_type_document_node(self):
        """
        Test that when creating a timeline with a document node, the node type is saved.
        """

        self.assertEqual(DocumentNode, Timeline(DocumentNode, 60, 0.5).node_type)

    def test_create_expiry(self):
        """
        Test that when creating a timeline with the expiry, it is saved.
        """

        self.assertEqual(60, Timeline(DocumentNode, 60, 0.5).expiry)

    def test_create_expiry_negative(self):
        """
        Test that when the timeline is created with a negative expiry, a ValueError is raised.
        """

        self.assertRaises(ValueError, Timeline, DocumentNode, -1, 0.5)

    def test_create_expiry_zero(self):
        """
        Test that when the timeline is created with an expiry of zero, no ValueError is raised.
        """

        self.assertTrue(Timeline(DocumentNode, 0, 0.5))

    def test_create_min_similarity(self):
        """
        Test that when creating a timeline with the minimum similarity, it is saved.
        """

        self.assertEqual(0.5, Timeline(DocumentNode, 60, 0.5).min_similarity)

    def test_create_expiry_negative(self):
        """
        Test that when the timeline is created with a negative expiry, a ValueError is raised.
        """

        self.assertRaises(ValueError, Timeline, DocumentNode, -1, -0.01)

    def test_create_with_nodes(self):
        """
        Test that when nodes are provided in the constructor, they are saved.
        """

        nodes = [ DocumentNode(0, documents=[ Document('') ]), DocumentNode(0, documents=[ Document('') ]) ]
        timeline = Timeline(DocumentNode, 60, 0.5, nodes=nodes)
        self.assertEqual(timeline.nodes, nodes)

    def test_add_first_node(self):
        """
        Test that when adding documents to an empty node, the timeline adds it to the timeline in a new node.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        self.assertEqual([ ], timeline.nodes)
        timeline.add(1000, documents)
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents, timeline.nodes[0].get_all_documents())

    def test_add_node_created_at(self):
        """
        Test that when creating a new node, the created at time is copied as given.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(1000, documents)
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(1000, timeline.nodes[0].created_at)

    def test_add_node_created_at_zero(self):
        """
        Test that when creating a new node, the created at time is copied as given even if it is zero.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(0, documents)
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(0, timeline.nodes[0].created_at)

    def test_add_node_created_at_realtime(self):
        """
        Test that when creating a new node, the created at time is the current time if it is not given.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(None, documents)
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(round(time.time()), round(timeline.nodes[0].created_at))

    def test_add_node_document_type(self):
        """
        Test that when creating a new node, the correct node type is used.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(None, documents)
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(DocumentNode, type(timeline.nodes[0]))

    def test_add_node_cluster_type(self):
        """
        Test that when creating a new node, the correct node type is used.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }) ]
        timeline = Timeline(ClusterNode, 60, 0.5)
        timeline.add(None, Cluster(documents))
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(ClusterNode, type(timeline.nodes[0]))
        self.assertEqual(documents, timeline.nodes[0].get_all_documents())

    def test_add_node_unexpired(self):
        """
        Test that when adding a node and there is an unexpired node, it absorbs the documents.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(0, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        timeline.add(59, documents[1:])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents, timeline.nodes[0].get_all_documents())

    def test_add_node_just_expired(self):
        """
        Test that when adding a node and an active one has just expired, a new node is created.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(0, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        timeline.add(60, documents[1:])
        self.assertEqual(2, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        self.assertEqual(documents[1:], timeline.nodes[1].get_all_documents())

    def test_add_node_last_absorbs(self):
        """
        Test that when adding a new node, it is only the last node that absorbs the documents.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }),
                      Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(0, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        timeline.add(61, documents[1:2])
        self.assertEqual(2, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        self.assertEqual(documents[1:2], timeline.nodes[1].get_all_documents())
        timeline.add(120, documents[2:])
        self.assertEqual(2, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        self.assertEqual(documents[1:], timeline.nodes[1].get_all_documents())

    def test_add_node_absorb(self):
        """
        Test that when a similar document is added, a node absorbs it, even if it has expired.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }),
                      Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(0, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        timeline.add(61, documents[2:])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual([ documents[0], documents[2] ] , timeline.nodes[0].get_all_documents())

    def test_add_node_absorb_max_time(self):
        """
        Test that when a similar node is added, a node does not absorb it if too much time has passed since it was created.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }),
                      Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5, max_time=600)
        timeline.add(0, documents)
        self.assertEqual(1, len(timeline.nodes))
        timeline.add(700, documents)
        self.assertEqual(2, len(timeline.nodes))

    def test_add_node_absorb_max_time_inclusive(self):
        """
        Test that when a similar node is added, the maximum time is inclusive.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }),
                      Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5, max_time=600)
        timeline.add(0, documents)
        self.assertEqual(1, len(timeline.nodes))
        timeline.add(600, documents)
        self.assertEqual(1, len(timeline.nodes))

    def test_add_node_absorb_before_max_time(self):
        """
        Test that when a similar node is added, it is absorbed if there is a node that is fairly recent.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }),
                      Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5, max_time=600)
        timeline.add(0, documents)
        self.assertEqual(1, len(timeline.nodes))
        timeline.add(599, documents)
        self.assertEqual(1, len(timeline.nodes))

    def test_add_node_absorb_inclusive(self):
        """
        Test that the minimum similarity is inclusive.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }),
                      Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 1)
        timeline.add(0, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        timeline.add(61, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1] * 2, timeline.nodes[0].get_all_documents())

    def test_add_node_absorb_zero(self):
        """
        Test that when the minimum similarity is zero, no new nodes are created.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }),
                      Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0)
        timeline.add(0, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        timeline.add(61, documents[1:2])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:2], timeline.nodes[0].get_all_documents())
        timeline.add(122, documents[2:])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents, timeline.nodes[0].get_all_documents())

    def test_add_node_absorbs_last(self):
        """
        Test that when going over the list of nodes, it is done in reverse.
        """

        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }),
                      Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(0, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        timeline.add(61, documents[1:2])
        self.assertEqual(2, len(timeline.nodes))
        self.assertEqual(documents[1:2], timeline.nodes[1].get_all_documents())
        timeline.add(122, documents[2:])
        self.assertEqual(2, len(timeline.nodes))
        self.assertEqual(documents[1:], timeline.nodes[-1].get_all_documents())

    def test_add_node_absorbs_similar(self):
        """
        Test that when going over the list of nodes, the first similar node from the end absorbs the documents.
        """

        documents = [ Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 }),
                      Document('this is a picture of dorian gray', { 'dorian': 1, 'gray': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]
        timeline = Timeline(DocumentNode, 60, 0.5)
        timeline.add(0, documents[:1])
        self.assertEqual(1, len(timeline.nodes))
        self.assertEqual(documents[:1], timeline.nodes[0].get_all_documents())
        timeline.add(61, documents[1:2])
        self.assertEqual(2, len(timeline.nodes))
        self.assertEqual(documents[1:2], timeline.nodes[1].get_all_documents())
        timeline.add(122, documents[2:])
        self.assertEqual(2, len(timeline.nodes))
        self.assertEqual([ documents[0], documents[2] ], timeline.nodes[0].get_all_documents())

    def test_create_expiry_zero(self):
        """
        Test that when the timeline is created with an expiry of zero, no ValueError is raised.
        """

        self.assertTrue(Timeline(DocumentNode, 0, 0))

    def test_create_expiry_large(self):
        """
        Test that when the timeline is created with a large expiry, a ValueError is raised.
        """

        self.assertRaises(ValueError, Timeline, DocumentNode, -1, 1.01)

    def test_create_expiry_one(self):
        """
        Test that when the timeline is created with an expiry of one, no ValueError is raised.
        """

        self.assertTrue(Timeline(DocumentNode, 0, 1))

    def test_create_document_node(self):
        """
        Test that when creating a node, the node type is as given in the constructor.
        """

        timeline = Timeline(DocumentNode, 60, 0.5)
        node = timeline._create(0)
        self.assertEqual(DocumentNode, type(node))

    def test_create_cluster_node(self):
        """
        Test that when creating a node, the node type is as given in the constructor.
        """

        timeline = Timeline(ClusterNode, 60, 0.5)
        node = timeline._create(0)
        self.assertEqual(ClusterNode, type(node))

    def test_create_node_created_at(self):
        """
        Test that when the creation time is given, it is passed on to the node.
        """

        timeline = Timeline(ClusterNode, 60, 0.5)
        node = timeline._create(created_at=1000)
        self.assertEqual(1000, node.created_at)

    def test_create_node_created_at(self):
        """
        Test that when the creation time is given, it is passed on to the node even if it is 0.
        """

        timeline = Timeline(ClusterNode, 60, 0.5)
        node = timeline._create(created_at=0)
        self.assertEqual(0, node.created_at)

    def test_create_node_created_at_none(self):
        """
        Test that when the creation time is not given, the time is real-time.
        """

        timeline = Timeline(ClusterNode, 60, 0.5)
        node = timeline._create(time.time())
        self.assertEqual(round(time.time()), round(node.created_at))

    def test_export(self):
        """
        Test exporting and importing timelines.
        """

        nodes = [ DocumentNode(0) ]
        timeline = Timeline(DocumentNode, 60, 0.5, nodes=nodes)
        e = timeline.to_array()
        self.assertEqual(timeline.node_type, Timeline.from_array(e).node_type)
        self.assertEqual(timeline.expiry, Timeline.from_array(e).expiry)
        self.assertEqual(timeline.min_similarity, Timeline.from_array(e).min_similarity)
        self.assertEqual(timeline.nodes[0].__dict__, Timeline.from_array(e).nodes[0].__dict__)

    def test_export_with_nodes(self):
        """
        Test exporting timelines that have nodes.
        """

        nodes = [ ClusterNode(0, clusters=[ Cluster(Document('text', { 'a': 1 }, attributes={ 'b': 2 }), { 'c': 3 }) ]) ]
        timeline = Timeline(ClusterNode, 120, 0.1, nodes=nodes)
        e = timeline.to_array()
        self.assertEqual(timeline.node_type, Timeline.from_array(e).node_type)
        self.assertEqual(timeline.expiry, Timeline.from_array(e).expiry)
        self.assertEqual(timeline.min_similarity, Timeline.from_array(e).min_similarity)
        self.assertEqual(len(timeline.nodes), len(e['nodes']))
        self.assertTrue(all(node['class'] == "<class 'summarization.timeline.nodes.cluster_node.ClusterNode'>" for node in e['nodes']))
        self.assertEqual({ 'a': 1 }, e['nodes'][0]['clusters'][0]['vectors'][0]['dimensions'])
        self.assertEqual({ 'b': 2 }, e['nodes'][0]['clusters'][0]['vectors'][0]['attributes'])
        self.assertEqual({ 'c': 3 }, e['nodes'][0]['clusters'][0]['attributes'])

    def test_import(self):
        """
        Test importing timelines that have nodes.
        """

        nodes = [ DocumentNode(0, documents=[ Document('text', { 'a': 1 }, attributes={ 'b': 2 }) ]),
                   DocumentNode(0, documents=[ Document('text', { 'c': 3 }, attributes={ 'd': 4 }) ]) ]
        timeline = Timeline(DocumentNode, 120, 0.1, nodes=nodes)

        e = timeline.to_array()
        i = Timeline.from_array(e)
        self.assertEqual(timeline.node_type, i.node_type)
        self.assertEqual(timeline.expiry, i.expiry)
        self.assertEqual(timeline.min_similarity, i.min_similarity)
        self.assertEqual(len(timeline.nodes), len(i.nodes))
        self.assertEqual(timeline.nodes[0].documents[0].__dict__, i.nodes[0].documents[0].__dict__)
        self.assertEqual(timeline.nodes[1].documents[0].__dict__, i.nodes[1].documents[0].__dict__)
