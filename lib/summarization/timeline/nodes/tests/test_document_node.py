"""
Run unit tests on the document node.
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
from summarization.timeline.nodes import DocumentNode
from vsm import vector_math
from vsm.clustering import Cluster

class TestDocumentNode(unittest.TestCase):
    """
    Test the document node.
    """

    def test_create_empty(self):
        """
        Test that the document node is created empty.
        """

        self.assertEqual([ ], DocumentNode(0).documents)

    def test_create_empty_attributes(self):
        """
        Test that the document node is created with no attributes.
        """

        self.assertEqual({ }, DocumentNode(0).attributes)

    def test_create_with_timestamp_zero(self):
        """
        Test that the document node saves the timestamp correctly even if it is zero.
        """

        self.assertEqual(0, DocumentNode(0).created_at)

    def test_create_with_timestamp(self):
        """
        Test that the document node saves the timestamp correctly.
        """

        self.assertEqual(1000, DocumentNode(1000).created_at)

    def test_create_with_no_documents(self):
        """
        Test that when creating the document node with no documents, an empty list is initialized.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)

    def test_create_with_documents(self):
        """
        Test that when creating the document node with a list of documents, it is saved.
        """

        documents = [ Document(''), Document('') ]
        n1 = DocumentNode(0, documents=documents[:1])
        self.assertEqual(documents[:1], n1.documents)

        n2 = DocumentNode(0, documents=documents[1:])
        self.assertEqual(documents[:1], n1.documents)
        self.assertEqual(documents[1:], n2.documents)

    def test_add(self):
        """
        Test adding documents to the node.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)
        document = Document('')
        node.add([ document ])
        self.assertEqual([ document ], node.documents)

    def test_add_one(self):
        """
        Test adding one document to the node.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)
        document = Document('')
        node.add(document)
        self.assertEqual([ document ], node.documents)

    def test_add_multiple(self):
        """
        Test adding multiple documents to the node.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)
        documents = [ Document('') for i in range(2)]
        node.add(documents)
        self.assertEqual(documents, node.documents)

    def test_add_repeated(self):
        """
        Test adding documents one at a time to the node.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)
        documents = [ Document('') for i in range(2)]
        node.add([ documents[0] ])
        self.assertEqual([ documents[0] ], node.documents)
        node.add([ documents[1] ])
        self.assertEqual(documents, node.documents)

    def test_add_copy_repeated(self):
        """
        Test that when adding documents one at a time to the node, copied documents are not kept.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)
        document = Document('text')
        for i in range(10):
            node.add([ document ])
        self.assertEqual(1, len(node.get_all_documents()))
        self.assertEqual([ document ], node.get_all_documents())

    def test_add_copy_batch(self):
        """
        Test that when adding documents to the node, copied documents are not kept.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)
        document = Document('text')
        node.add([ document ] * 10)
        self.assertEqual(1, len(node.get_all_documents()))
        self.assertEqual([ document ], node.get_all_documents())

    def test_add_dynamic(self):
        """
        Test that when changing a document, the node's document also changes.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)
        document = Document('', { })
        node.add([ document ])
        self.assertEqual([ document ], node.documents)
        document.dimensions['x'] = 1
        self.assertEqual({ 'x': 1 }, document.dimensions)
        self.assertEqual(document.dimensions, node.documents[0].dimensions)

    def test_get_all_documents_empty(self):
        """
        Test that when getting all documents from an empty node, an empty list is returned.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.get_all_documents())

    def test_get_all_documents(self):
        """
        Test that when getting all documents from a node, all documents are returned.
        """

        node = DocumentNode(0)
        corpus = [ Document('', { }), Document('', { }) ]
        self.assertEqual([ ], node.get_all_documents())
        node.add([ corpus[0] ])
        self.assertEqual([ corpus[0] ], node.get_all_documents())
        node.add([ corpus[1] ])
        self.assertEqual(corpus, node.get_all_documents())

    def test_similarity_empty_node(self):
        """
        Test that the similarity between a document and an empty document node, the similarity is 0.
        """

        node = DocumentNode(0)
        self.assertEqual([ ], node.documents)
        self.assertEqual(0, node.similarity(Document('', { 'x': 1 })))

    def test_similarity_empty_document(self):
        """
        Test that the similarity between a node and an empty document, the similarity is 0.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = DocumentNode(0)
        node.add(documents)
        self.assertEqual(documents, node.documents)
        self.assertEqual(0, node.similarity(Document('', { })))

    def test_similarity(self):
        """
        Test calculating the similarity between a node and a document.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = DocumentNode(0)
        node.add(documents)
        self.assertEqual(documents, node.documents)
        self.assertEqual(math.sqrt(2)/2., node.similarity(Document('this is not a pipe', { 'pipe': 1 })))

    def test_similarity_several(self):
        """
        Test calculating the similarity between a node and several documents.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = DocumentNode(0)
        document = Document('this is not a pipe and this is not a cigar', { 'pipe': 1, 'cigar': 1 })
        node.add([ document ])
        self.assertEqual(math.sqrt(2)/2., node.similarity(documents[0]))
        self.assertEqual(math.sqrt(2)/2., node.similarity(documents[1]))

        centroid = Cluster(documents).centroid
        centroid.normalize()
        self.assertEqual(vector_math.cosine(centroid, Cluster(node.documents).centroid), node.similarity(documents))

    def test_similarity_lower_bound(self):
        """
        Test that the similarity lower-bound between a node and a document is 0.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = DocumentNode(0)
        node.add(documents)
        self.assertEqual(documents, node.documents)
        self.assertEqual(0, node.similarity(Document('this is a picture of dorian gray', { 'picture': 1, 'dorian': 1, 'gray': 1 })))

    def test_similarity_upper_bound(self):
        """
        Test that the similarity upper-bound between a node and a document is 1.
        """

        """
        Create the test data.
        """
        documents = [ Document('this is not a pipe', { 'pipe': 1 }),
                       Document('this is not a cigar', { 'cigar': 1 }) ]

        node = DocumentNode(0)
        node.add(documents)
        self.assertEqual(documents, node.documents)
        self.assertEqual(1, round(node.similarity(Document('this is not a pipe and this is not a cigar', { 'cigar': 1, 'pipe': 1 }))))

    def test_expired_inclusive(self):
        """
        Test that the expiry is inclusive.
        """

        node = DocumentNode(created_at=1000)
        self.assertTrue(node.expired(10, 1010))

    def test_expired_far_timestamp(self):
        """
        Test that a node is expired if the timestamp is sufficiently far.
        """

        node = DocumentNode(created_at=1000)
        self.assertTrue(node.expired(10, 1011))

    def test_expired_close_timestamp(self):
        """
        Test that a node is not expired if the timestamp is close.
        """

        node = DocumentNode(created_at=1000)
        self.assertFalse(node.expired(10, 1001))

    def test_expired_past_timestamp(self):
        """
        Test that a node is not expired if the timestamp is in the past.
        """

        node = DocumentNode(created_at=1000)
        self.assertFalse(node.expired(10, 999))

    def test_expired_realtime(self):
        """
        Test that when the timestamp is not given, the current timestamp is used.
        """

        node = DocumentNode(created_at=time.time())
        self.assertFalse(node.expired(1, time.time()))

    def test_expired_realtime_sleep(self):
        """
        Test that when the timestamp is not given, the current timestamp is used.
        """

        node = DocumentNode(created_at=time.time())
        time.sleep(1)
        self.assertTrue(node.expired(1, time.time()))

    def test_expired_zero(self):
        """
        Test that a node immediately expired if the expiry is 0.
        """

        node = DocumentNode(created_at=1000)
        self.assertTrue(node.expired(0, 1000))

    def test_expired_negative(self):
        """
        Test that a ValueError is raised when the expiry is negative.
        """

        node = DocumentNode(created_at=1000)
        self.assertRaises(ValueError, node.expired, -1, 0)

    def test_export(self):
        """
        Test exporting and importing document nodes.
        """

        node = DocumentNode(0)
        e = node.to_array()
        self.assertEqual(node.created_at, DocumentNode.from_array(e).created_at)
        self.assertEqual(node.documents, DocumentNode.from_array(e).documents)
        self.assertEqual(node.__dict__, DocumentNode.from_array(e).__dict__)

    def test_export_with_documents(self):
        """
        Test exporting document nodes that have documents.
        """

        documents = [ Document('', { 'a': 1 }, attributes={ 'b': 2 }),
                      Document('', { 'c': 3 }, attributes={ 'd': 4 }) ]
        node = DocumentNode(created_at=0, documents=documents)
        e = node.to_array()
        self.assertEqual(node.created_at, DocumentNode.from_array(e).created_at)
        self.assertTrue(all(document['class'] == "<class 'nlp.document.Document'>" for document in e['documents']))
        self.assertEqual({ 'a': 1 }, e['documents'][0]['dimensions'])
        self.assertEqual({ 'b': 2 }, e['documents'][0]['attributes'])
        self.assertEqual({ 'c': 3 }, e['documents'][1]['dimensions'])
        self.assertEqual({ 'd': 4 }, e['documents'][1]['attributes'])

    def test_import(self):
        """
        Test importing document nodes that have documents.
        """

        documents = [ Document('', { 'a': 1 }, attributes={ 'b': 2 }),
                      Document('', { 'c': 3 }, attributes={ 'd': 4 }) ]
        node = DocumentNode(created_at=0, documents=documents)
        e = node.to_array()
        i = DocumentNode.from_array(e)
        self.assertEqual(node.created_at, i.created_at)
        self.assertTrue(all(type(document) is Document for document in i.documents))
        self.assertEqual(documents[0].dimensions, i.documents[0].dimensions)
        self.assertEqual(documents[0].attributes, i.documents[0].attributes)
        self.assertEqual(documents[1].dimensions, i.documents[1].dimensions)
        self.assertEqual(documents[1].attributes, i.documents[1].attributes)

    def test_merge_none(self):
        """
        Test that when merging no document nodes, the merge function returns a new, empty node.
        """

        node = DocumentNode.merge(10)
        self.assertEqual(10, node.created_at)
        self.assertFalse(node.documents)
        self.assertEqual(DocumentNode, type(node))

    def test_merge_created_at_from_parameter(self):
        """
        Test that when merging document nodes, the timestamp is taken from the parameters.
        """

        documents = [ Document('', { 'a': 1 }), Document('', { 'b': 2 }) ]
        _node = DocumentNode(created_at=0, documents=documents)
        node = DocumentNode.merge(10)
        self.assertEqual(10, node.created_at)

    def test_merge_one(self):
        """
        Test that when merging one document node, a new node with the same documents is returned.
        """

        documents = [ Document('', { 'a': 1 }), Document('', { 'b': 2 }) ]
        _node = DocumentNode(created_at=0, documents=documents)
        node = DocumentNode.merge(10, _node)
        self.assertEqual(_node.documents, node.documents)

    def test_merge_all(self):
        """
        Test that when merging document nodes, all of the documents are present in the new node.
        """

        documents = [ Document('', { 'a': 1 }), Document('', { 'b': 2 }),
                       Document('', { 'c': 3 }), Document('', { 'd': 4 }) ]
        nodes = [ DocumentNode(created_at=0, documents=documents[:2]),
                    DocumentNode(created_at=0, documents=documents[2:]) ]
        node = DocumentNode.merge(10, nodes[0], nodes[1])
        self.assertTrue(all( document in node.documents for node in nodes
                                                         for document in node.documents ))

    def test_merge_order(self):
        """
        Test that when merging document nodes, the order of the documents is the same as the order of the nodes.
        """

        documents = [ Document('', { 'a': 1 }), Document('', { 'b': 2 }),
                       Document('', { 'c': 3 }), Document('', { 'd': 4 }) ]
        nodes = [ DocumentNode(created_at=0, documents=documents[:2]),
                    DocumentNode(created_at=0, documents=documents[2:]) ]
        node = DocumentNode.merge(10, nodes[0], nodes[1])
        self.assertEqual(documents, node.documents)
