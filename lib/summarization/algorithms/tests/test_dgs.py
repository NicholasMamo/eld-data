"""
Run unit tests on the Document Graph Summarizer by Mamo et al. (2019)'s algorithm.
"""

import math
import os
import sys
import unittest

import networkx as nx

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp.document import Document
from nlp.weighting.tf import TF
from summarization import Summary
from summarization.algorithms import DGS
from vsm import vector_math, Vector

class TestDGS(unittest.TestCase):
    """
    Test Document Graph Summarizer by Mamo et al. (2019)'s algorithm.
    """

    def test_summarize_empty(self):
        """
        Test that when summarizing an empty set of documents, an empty summary is returned.
        """

        algo = DGS()
        self.assertEqual([ ], algo.summarize([ ], 100).documents)

    def test_summarize_small_length(self):
        """
        Test that when summarizing a set of documents, all of which exceed the length, an empty summary is returned.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        self.assertEqual([ ], algo.summarize(corpus, 10).documents)

    def test_summarize_exact_length(self):
        """
        Test that when summarizing a set of documents and there is only one candidate, that candidate is included in the summary.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        summary = algo.summarize(corpus, len(str(corpus[0])))
        self.assertEqual([ corpus[0] ], summary.documents)
        self.assertLessEqual(len(str(summary)), len(str(corpus[0])))

    def test_summarize_exact_full_length(self):
        """
        Test that when summarizing a set of documents, the exact length does not include all documents because of the spaces between them in the final summary.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        length = sum(len(str(document)) for document in corpus)
        summary = algo.summarize(corpus, length)
        self.assertTrue(len(set(corpus).difference(set(summary.documents))))
        self.assertLessEqual(len(str(summary)), length)

    def test_summarize_long_summary_length(self):
        """
        Test that when summarizing a set of documents and a long length is given, the number of documents included is less than the square root of the number of documents.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        length = sum(len(str(document)) for document in corpus) + len(corpus)
        summary = algo.summarize(corpus, length)
        self.assertLessEqual(len(str(summary)), length)
        self.assertGreaterEqual(math.ceil(math.sqrt(len(corpus))), len(summary.documents))

    def test_summary_large_communities(self):
        """
        Test that when summarizing, large communities are preferred.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }),
                   Document('the picture of dorian gray', { 'picture': 1, 'dorian': 1, 'gray': 1 })]
        for document in corpus:
            document.normalize()

        algo = DGS()
        length = 30
        summary = algo.summarize(corpus, length)
        self.assertLessEqual(len(str(summary)), length)
        self.assertEqual(1, len(summary.documents))

    def test_summary_one_each_community(self):
        """
        Test that when summarizing, only one document is chosen from each community.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }),
                   Document('the picture of dorian gray', { 'picture': 1, 'dorian': 1, 'gray': 1 })]
        for document in corpus:
            document.normalize()

        algo = DGS()
        length = 100
        summary = algo.summarize(corpus, length)
        self.assertEqual({ corpus[2], corpus[3] }, set(summary.documents))

    def test_summarize_custom_query(self):
        """
        Test that when summarizing a set of documents and a custom query is given, that query is used.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }),
                   Document('the original picture of a pipe', { 'picture': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        length = max(len(str(document)) for document in corpus)

        summary = algo.summarize(corpus, length)
        self.assertEqual([ corpus[2] ], summary.documents)

        query = Document('', { 'picture': 1 })
        summary = algo.summarize(corpus, length, query=query)
        self.assertEqual([ corpus[3] ], summary.documents)

    def test_summarize_documents_unchanged(self):
        """
        Test that when summarizing a set of documents, they are unchanged.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }),
                   Document('the original picture of a pipe', { 'picture': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        query = Document('', { 'picture': 1 })
        copy = query.copy()

        length = max(len(str(document)) for document in corpus)
        summary = algo.summarize(corpus, length, query=query)
        self.assertEqual(copy.dimensions, query.dimensions)

    def test_summarize_query_unchanged(self):
        """
        Test that when summarizing a set of documents with a query, it is unchanged.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }),
                   Document('the original picture of a pipe', { 'picture': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        copy = [ document.copy() for document in corpus ]

        length = max(len(str(document)) for document in corpus)
        summary = algo.summarize(corpus, length)
        for d1, d2 in zip(copy, corpus):
            self.assertEqual(d1.dimensions, d2.dimensions)

    def test_negative_length(self):
        """
        Test that when providing a negative length, the function raises a ValueError.
        """

        c = [ ]
        algo = DGS()
        self.assertRaises(ValueError, algo.summarize, c, -1)

    def test_zero_length(self):
        """
        Test that when providing a length of zero, the function raises a ValueError.
        """

        c = [ ]
        algo = DGS()
        self.assertRaises(ValueError, algo.summarize, c, 0)

    def test_compute_query_empty(self):
        """
        Test that the query is empty when no documents are given.
        """

        algo = DGS()
        self.assertEqual({ }, algo._compute_query([ ]).dimensions)

    def test_compute_query_one_document(self):
        """
        Test that the query construction is identical to the given document when there is just one document.
        """

        d = Document('this is not a pipe', { 'pipe': 1 })
        algo = DGS()
        self.assertEqual(d.dimensions, algo._compute_query([ d ]).dimensions)

    def test_compute_query_normalized(self):
        """
        Test that the query is normallized.
        """

        d = Document('this is not a pipe', { 'this': 1, 'pipe': 1 })
        d.normalize()
        algo = DGS()
        self.assertEqual(1, round(vector_math.magnitude(algo._compute_query([ d ])), 10))

    def test_compute_query(self):
        """
        Test the query construction with multiple documents.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'this': 1 }),
                    Document('this is not a pipe', { 'pipe': 1 }) ]

        algo = DGS()
        self.assertEqual(round(math.sqrt(2)/2., 10), round(algo._compute_query(corpus).dimensions['this'], 10))
        self.assertEqual(round(math.sqrt(2)/2., 10), round(algo._compute_query(corpus).dimensions['pipe'], 10))
        self.assertEqual(1, round(vector_math.magnitude(algo._compute_query(corpus)), 10))

    def test_create_empty_graph(self):
        """
        Test that when creating a graph with no documents, an empty graph is created instead.
        """

        algo = DGS()
        graph = algo._to_graph([ ])
        self.assertEqual([ ], list(graph.nodes))
        self.assertEqual([ ], list(graph.edges))

    def test_create_graph_one_document(self):
        """
        Test that when creating a graph with one document, a graph with one node and no edges is created.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'this': 1 }) ]

        algo = DGS()
        graph = algo._to_graph(corpus)
        self.assertEqual(corpus, list(graph.nodes))
        self.assertEqual([ ], list(graph.edges))

    def test_create_graph_different_documents(self):
        """
        Test that when creating a graph with different documents, no edge is created between them.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                    Document('this is not a cigar', { 'cigar': 1 }), ]

        algo = DGS()
        graph = algo._to_graph(corpus)
        self.assertEqual(corpus, list(graph.nodes))
        self.assertEqual([ ], list(graph.edges))

    def test_create_graph_similar_documents(self):
        """
        Test that when creating a graph with similar documents, edges are created between them.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                    Document('this is not a cigar', { 'cigar': 1 }),
                   Document('this is not a pipe, but a cigar', { 'pipe': 1, 'cigar': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        self.assertEqual(corpus, list(graph.nodes))
        self.assertEqual(2, len(list(graph.edges)))
        self.assertTrue((corpus[0], corpus[2]) in graph.edges)
        self.assertTrue((corpus[1], corpus[2]) in graph.edges)

    def test_create_graph_triangle(self):
        """
        Test that when creating a graph with three documents, all of which are similar, three edges are created.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                    Document('this is a different pipe', { 'pipe': 1 }),
                   Document('this is not a pipe, but a cigar', { 'pipe': 1, 'cigar': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        self.assertEqual(corpus, list(graph.nodes))
        self.assertEqual(3, len(list(graph.edges)))
        self.assertTrue((corpus[0], corpus[1]) in graph.edges)
        self.assertTrue((corpus[0], corpus[2]) in graph.edges)
        self.assertTrue((corpus[1], corpus[2]) in graph.edges)
        self.assertLess(graph.edges[(corpus[0], corpus[1])]['weight'], graph.edges[(corpus[0], corpus[2])]['weight'])
        self.assertLess(graph.edges[(corpus[1], corpus[0])]['weight'], graph.edges[(corpus[0], corpus[2])]['weight'])

    def test_create_graph_undirected_edges(self):
        """
        Test that when creating a graph with similar documents, the edges are undirected.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                    Document('this is not a cigar', { 'cigar': 1 }),
                   Document('this is not a pipe, but a cigar', { 'pipe': 1, 'cigar': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        self.assertEqual(corpus, list(graph.nodes))
        self.assertEqual(2, len(list(graph.edges)))
        self.assertTrue(graph.edges[(corpus[0], corpus[2])])
        self.assertTrue(graph.edges[(corpus[2], corpus[0])])
        self.assertTrue(graph.edges[(corpus[1], corpus[2])])
        self.assertTrue(graph.edges[(corpus[2], corpus[1])])

    def test_create_graph_weight(self):
        """
        Test that when creating a graph, the weight is the inverted similarity.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                   Document('this is a pipe', { 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        self.assertEqual(corpus, list(graph.nodes))
        self.assertEqual(1, len(list(graph.edges)))
        self.assertEqual(0, graph.edges[(corpus[0], corpus[1])]['weight'])

    def test_create_graph_duplicate_document(self):
        """
        Test that when creating a graph with two documents having the same text and attributes, they are created as separate nodes.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                   Document('this is not a pipe', { 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        self.assertEqual(corpus, list(graph.nodes))
        self.assertEqual(2, len(list(graph.nodes)))

    def test_create_graph_documents_attributes(self):
        """
        Test that when creating a graph, the documents are added as attributes.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                   Document('this is not a pipe', { 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        self.assertEqual(corpus, list(graph.nodes))
        self.assertEqual(2, len(list(graph.nodes)))
        self.assertEqual(corpus[0], graph.nodes[corpus[0]]['document'])
        self.assertEqual(corpus[1], graph.nodes[corpus[1]]['document'])

    def test_edge_centrality(self):
        """
        Test that the edge centrality correctly identifies the most central edge.
        """

        nodes =  [ 'A', 'B', 'C', 'D', 'W', 'X', 'Y', 'Z' ]
        edges = { ('A', 'B', 0.1), ('A', 'C', 0.1), ('A', 'D', 0.1),
                   ('B', 'C', 0.1), ('B', 'D', 0.1), ('C', 'D', 0.1),

                  ('W', 'X', 0.1), ('W', 'Y', 0.1), ('W', 'Z', 0.1),
                    ('X', 'Y', 0.1), ('X', 'Z', 0.1), ('Y', 'Z', 0.1),

                  ('D', 'W', 0.1)
                }

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges)

        algo = DGS()
        edge = algo._most_central_edge(graph)
        self.assertEqual(('D', 'W'), edge)

    def test_edge_centrality_multiple(self):
        """
        Test that the edge centrality correctly identifies the most central edge when there are two such edges.
        This edge should be the one with the lowest weight.
        """

        nodes =  [ 'A', 'B', 'C', 'D', 'W', 'X', 'Y', 'Z' ]
        edges = { ('A', 'B', 0.1), ('A', 'C', 0.1), ('A', 'D', 0.1),
                   ('B', 'C', 0.1), ('B', 'D', 0.1), ('C', 'D', 0.1),

                  ('W', 'X', 0.1), ('W', 'Y', 0.1), ('W', 'Z', 0.1),
                    ('X', 'Y', 0.1), ('X', 'Z', 0.1), ('Y', 'Z', 0.1),

                  ('D', 'W', 0.1), ('C', 'X', 0.05),
                }

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges)

        algo = DGS()
        edge = algo._most_central_edge(graph)
        self.assertEqual(('C', 'X'), edge)

    def test_extract_communities_empty_graph(self):
        """
        Test that when extracting communities from an empty graph, an empty list of partitions is returned.
        """

        graph = nx.Graph()
        algo = DGS()
        partitions = algo._extract_communities(graph)
        self.assertEqual([ ], partitions)

    def test_extract_communities_no_edges(self):
        """
        Test that when extracting communities from a graph with no edges, the same nodes are returned as partitions.
        """

        nodes = [ 'A', 'B', 'C', 'D' ]

        graph = nx.Graph()
        graph.add_nodes_from(nodes)

        algo = DGS()
        partitions = algo._extract_communities(graph)
        self.assertEqual(4, len(partitions))
        self.assertEqual([ { 'A' }, { 'B' }, { 'C' }, { 'D' } ], partitions)

    def test_extract_communities_weight(self):
        """
        Test that when extracting communities from an empty graph, the weight is considered primarily.
        """

        nodes = [ 'A', 'B', 'C', 'D' ]
        edges = [ ('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D') ]

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        algo = DGS()
        partitions = algo._extract_communities(graph)
        self.assertTrue({ 'A', 'B', 'C' } in partitions)
        self.assertTrue({ 'D' } in partitions)

        edges = [ ('A', 'B', 0.1), ('A', 'D', 1), ('B', 'C', 1), ('C', 'D', 0.4) ]
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges)
        partitions = algo._extract_communities(graph)
        self.assertTrue({ 'A', 'D' } in partitions)
        self.assertTrue({  'B', 'C' } in partitions)

    def test_extract_communities_square(self):
        """
        Test that when extracting communities, the number of partitions is at least equal to the square root of nodes.
        """

        nodes = [ 'A', 'B', 'C', 'D' ]
        edges = [ ('A', 'B', 0.1), ('B', 'C', 0.2), ('C', 'D', 0.4), ('A', 'D', 0.2) ]

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges)

        algo = DGS()
        partitions = algo._extract_communities(graph)
        self.assertEqual(2, len(partitions))
        self.assertTrue({ 'A', 'D' } in partitions)
        self.assertTrue({ 'B', 'C' } in partitions)

    def test_extract_document_communities(self):
        """
        Test that when extracting communities of documents, the partitions are also documents.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                   Document('this is not a cigar', { 'cigar': 1 }),
                   Document('this is not a cigar, but a pipe', { 'cigar': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        partitions = algo._extract_communities(graph)
        self.assertEqual(2, len(partitions))
        self.assertTrue({ corpus[0] } in partitions)
        self.assertTrue({ corpus[1], corpus[2] } in partitions)

    def test_extract_document_communities_trivial(self):
        """
        Test that when the quota of communities is already reached, no communities are extracted.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }),
                   Document('the picture of dorian gray', { 'picture': 1, 'dorian': 1, 'gray': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        partitions = algo._extract_communities(graph)
        self.assertEqual(2, len(partitions))
        self.assertTrue({ *corpus[:3] } in partitions)
        self.assertTrue({ corpus[3] } in partitions)

    def test_extract_document_communities_disconnected(self):
        """
        Test that when all nodes are disconnected, all are returned as distinct communities.
        """

        """
        Create the test data.
        """
        corpus = [ Document('A', { 'A': 1 }), Document('B', { 'B': 1 }),
                   Document('C', { 'C': 1 }), Document('D', { 'D': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        partitions = algo._extract_communities(graph)
        self.assertEqual(4, len(partitions))
        self.assertEqual([ { corpus[0] }, { corpus[1] }, { corpus[2] }, { corpus[3] } ], partitions)

    def test_largest_communities_empty(self):
        """
        Test that when getting the largest communities from an empty list, another empty list is returned.
        """

        algo = DGS()
        self.assertEqual([ ], algo._largest_communities([ ]))

    def test_largest_communities_one(self):
        """
        Test that when getting the largest communities of one community, that community is returned.
        """

        algo = DGS()
        self.assertEqual([ { 'A' } ], algo._largest_communities([ { 'A' } ]))

    def test_largest_communities(self):
        """
        Test that when getting the largest communities where only one stands out, that community is returned.
        """

        algo = DGS()
        self.assertEqual([ { 'A', 'B', } ], algo._largest_communities([ { 'A', 'B' } ]))

    def test_largest_communities_multiple(self):
        """
        Test that when getting the largest communities where multiple are large, all are returned.
        """

        algo = DGS()
        self.assertEqual([ { 'A', 'B', }, { 'C', 'D' } ],
                         algo._largest_communities([ { 'A', 'B' }, { 'C', 'D' }, { 'E' } ]))

    def test_score_documents_no_communities(self):
        """
        Test that when scoring no communities, an empty list is returned.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                   Document('a pipe is preferable', { 'pipe': 1 }),
                   Document('this is not a cigar', { 'cigar': 1 }),
                   Document('this is not a cigar, but a pipe', { 'cigar': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Vector({ 'cigar': 1 })
        query.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        scores = algo._score_documents(graph, [ ], query)
        self.assertEqual(0, len(scores))

    def test_score_documents_one_community(self):
        """
        Test scoring the documents in one community.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                   Document('a pipe is preferable', { 'pipe': 1 }),
                   Document('this is not a cigar', { 'cigar': 1 }),
                   Document('this is not a cigar, but a pipe', { 'cigar': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Vector({ 'cigar': 1 })
        query.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        scores = algo._score_documents(graph, [ { *corpus } ], query)
        self.assertEqual(1, len(scores))
        scores = scores[0]
        self.assertEqual(0, scores[corpus[0]])
        self.assertEqual(0, scores[corpus[1]])
        self.assertGreater(scores[corpus[2]], scores[corpus[0]])
        self.assertGreater(scores[corpus[2]], scores[corpus[1]])
        self.assertEqual(scores[corpus[-1]], max(scores.values()))

    def test_score_documents_relevance(self):
        """
        Test that when scoring a community, relevance influences results.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'this': 1, 'pipe': 1 }),
                   Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                   Document('this is a cigar, but a pipe', { 'this': 1, 'cigar': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Vector({ 'cigar': 1 })
        query.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        scores = algo._score_documents(graph, [ { *corpus } ], query)
        self.assertEqual(1, len(scores))
        scores = scores[0]
        self.assertEqual(0, scores[corpus[0]])
        self.assertEqual(scores[corpus[2]], max(scores.values()))
        self.assertEqual(scores[corpus[-1]], max(scores.values()))

    def test_score_documents_centrality(self):
        """
        Test that when scoring a community, centrality influences results.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                   Document('this is a cigar', { 'this': 1, 'cigar': 1 }),
                   Document('this is a cigar, but a pipe', { 'cigar': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Vector({ 'cigar': 1, 'this': 1, 'pipe': 1 })
        query.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        scores = algo._score_documents(graph, [ { *corpus } ], query)
        self.assertEqual(1, len(scores))
        scores = scores[0]
        self.assertEqual(scores[corpus[1]], min(scores.values()))
        self.assertEqual(scores[corpus[-1]], max(scores.values()))
        self.assertGreater(scores[corpus[-1]], scores[corpus[1]])

    def test_score_documents_multiple_communities(self):
        """
        Test scoring the documents in multiple communities.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a pipe', { 'pipe': 1 }),
                   Document('a pipe is preferable', { 'pipe': 1 }),
                   Document('this is not a cigar', { 'cigar': 1 }),
                   Document('this is not a cigar, but a pipe', { 'cigar': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Vector({ 'cigar': 1 })
        query.normalize()

        algo = DGS()
        graph = algo._to_graph(corpus)
        scores = algo._score_documents(graph, [ { *corpus[:2] }, { *corpus[2:] } ], query)
        self.assertEqual(2, len(scores))
        self.assertEqual(set(corpus[:2]), set(scores[0].keys()))
        self.assertEqual(set(corpus[2:]), set(scores[1].keys()))

    def test_brevity_score(self):
        """
        Test the calculation of the brevity score.
        """

        algo = DGS()
        text = 'this is a pipe'
        self.assertEqual(0.22313, round(algo._brevity_score(text, r=10), 5))

    def test_brevity_score_equal(self):
        """
        Test that when the text has as many tokens as required, the score is 1.
        """

        algo = DGS()
        text = 'a pipe is not a cigar and a cigar is not a pipe'
        self.assertEqual(1, algo._brevity_score(text, r=4))

    def test_brevity_score_long(self):
        """
        Test that when the text has more tokens than required, the score is 1.
        """

        algo = DGS()
        text = 'a pipe is not a cigar and a cigar is not a pipe'
        self.assertEqual(1, algo._brevity_score(text, r=3))

    def test_brevity_score_bounds(self):
        """
        Test that the bounds of the brevity score are between 0 and 1.
        """

        algo = DGS()

        text = ''
        self.assertEqual(0, algo._brevity_score(text))
        text = 'a pipe is not a cigar and a cigar is not a pipe'
        self.assertEqual(1, algo._brevity_score(text, r=3))

    def test_brevity_score_custom_r(self):
        """
        Test that when a custom ideal length is given, it is used.
        """

        algo = DGS()

        text = 'a pipe is not a cigar'
        self.assertEqual(0.84648, round(algo._brevity_score(text, r=7), 5))
        text = 'a pipe is not a cigar'
        self.assertEqual(0.60653, round(algo._brevity_score(text, r=9), 5))

    def test_filter_documents_empty(self):
        """
        Test that when filtering an empty list of documents, an empty list is returned.
        """

        algo = DGS()
        self.assertEqual([ ], algo._filter_documents([ ], Summary(), 0))

    def test_filter_documents_empty_summary(self):
        """
        Test that when filtering a list of documents with an empty summary, the same documents are returned.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]

        algo = DGS()
        self.assertEqual(set(corpus), set(algo._filter_documents(corpus, Summary(), 99)))

    def test_filter_documents_in_summary(self):
        """
        Test filtering documents in the summary.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]

        algo = DGS()
        self.assertEqual([ corpus[1] ], algo._filter_documents(corpus, Summary(corpus[0]), 99))

    def test_filter_all_documents(self):
        """
        Test that when filtering all of the documents in the summary, an empty list is returned.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]

        algo = DGS()
        self.assertEqual([ ], algo._filter_documents(corpus, Summary(corpus), 99))

    def test_filter_extra_documents(self):
        """
        Test that when the summary contains extra documents, an empty list is returned.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]

        algo = DGS()
        self.assertEqual([ ], algo._filter_documents(corpus[:1], Summary(corpus), 99))

    def test_filter_zero_length(self):
        """
        Test that when filtering with a length of zero, no documents are retained.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]

        algo = DGS()
        self.assertEqual([ ], algo._filter_documents(corpus, Summary(), 0))

    def test_filter_exact_length(self):
        """
        Test that when a document has the same length as the filter length, it is retained.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]

        algo = DGS()
        self.assertEqual(set(corpus), set(algo._filter_documents(corpus, Summary(), len(str(corpus[0])))))

    def test_filter_length(self):
        """
        Test that when filtering with a length, longer documents are excluded.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]

        algo = DGS()
        self.assertEqual(corpus[1:], algo._filter_documents(corpus, Summary(), len(str(corpus[1]))))
