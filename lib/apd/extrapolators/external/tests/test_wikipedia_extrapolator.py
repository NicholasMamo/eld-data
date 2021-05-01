"""
Test the functionality of the Wikipedia extrapolator.
"""

import os
import random
import re
import string
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

import networkx as nx
from nltk.corpus import stopwords

from apd.extractors.local.entity_extractor import EntityExtractor
from apd.scorers.local.tf_scorer import TFScorer
from apd.filters.local.threshold_filter import ThresholdFilter
from apd.resolvers.external.wikipedia_search_resolver import WikipediaSearchResolver
from apd.extrapolators.external.wikipedia_extrapolator import WikipediaExtrapolator

from nlp.document import Document
from nlp.tokenizer import Tokenizer
from nlp.weighting.tf import TF

class TestWikipediaExtrapolator(unittest.TestCase):
    """
    Test the implementation and results of the Wikipedia extrapolator.
    """

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

        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertEqual(('D', 'W'), extrapolator._most_central_edge(graph))

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

        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertEqual(('C', 'X'), extrapolator._most_central_edge(graph))

    def test_year_check(self):
        """
        Test that when checking for a year, the function returns a boolean.
        """

        article = 'Youssouf Koné (footballer, born 1995)'
        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertTrue(extrapolator._has_year(article))

    def test_year_check_range(self):
        """
        Test that when checking for a year in a range, the function returns `True`.
        """

        article = '2019–20 Premier League'
        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertTrue(extrapolator._has_year(article))

        article = '2019-20 Premier League'
        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertTrue(extrapolator._has_year(article))

    def test_year_check_short_number(self):
        """
        Test that when checking for a year with a short number, the function does not detect a year.
        """

        article = 'Area 51'
        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertFalse(extrapolator._has_year(article))

    def test_year_check_long_number(self):
        """
        Test that when checking for a year with a long number, the function does not detect a year.
        """

        article = '1234567890'
        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertFalse(extrapolator._has_year(article))

    def test_remove_brackets(self):
        """
        Test that when removing brackets, they are completely removed.
        """

        article = 'Youssouf Koné (footballer, born 1995)'
        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertEqual('Youssouf Koné', extrapolator._remove_brackets(article).strip())

    def test_remove_unclosed_brackets(self):
        """
        Test that when removing brackets that are not closed, they are not removed.
        """

        article = 'Youssouf Koné (footballer, born 1995'
        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        self.assertEqual('Youssouf Koné (footballer, born 1995', extrapolator._remove_brackets(article).strip())

    def test_get_first_sentence(self):
        """
        Test that when getting the first sentence from text, only the first sentence is returned.
        """

        text = "Memphis Depay (Dutch pronunciation: [ˈmɛmfɪs dəˈpɑi]; born 13 February 1994), \
                commonly known simply as Memphis,[2] is a Dutch professional \
                footballer and music artist who plays as a forward and captains \
                French club Lyon and plays for the Netherlands national team. \
                He is known for his pace, ability to cut inside, dribbling, \
                distance shooting and ability to play the ball off the ground."

        resolver = WikipediaExtrapolator(TF(), Tokenizer(), 0, [ ])
        self.assertEqual("Memphis Depay (Dutch pronunciation: [ˈmɛmfɪs dəˈpɑi]; born 13 February 1994), commonly known simply as Memphis,[2] is a Dutch professional footballer and music artist who plays as a forward and captains French club Lyon and plays for the Netherlands national team.",
                         re.sub('([ \t]+)', ' ', resolver._get_first_sentence(text)).strip())

    def test_get_first_sentence_full(self):
        """
        Test that when getting the first sentence from a text that has only one sentence, the whole text is returned.
        """

        text = "Youssouf Koné (born 5 July 1995) is a Malian professional footballer who plays for French side Olympique Lyonnais and the Mali national team as a left-back."
        resolver = WikipediaExtrapolator(TF(), Tokenizer(), 0, [ ])
        self.assertEqual(text, resolver._get_first_sentence(text))

    def test_get_first_sentence_full_without_period(self):
        """
        Test that when getting the first sentence from a text that has only one sentence, but without punctuation, the whole text is returned.
        """

        text = "Youssouf Koné (born 5 July 1995) is a Malian professional footballer who plays for French side Olympique Lyonnais and the Mali national team as a left-back"
        resolver = WikipediaExtrapolator(TF(), Tokenizer(), 0, [ ])
        self.assertEqual(text, resolver._get_first_sentence(text))

    def test_get_first_sentence_empty(self):
        """
        Test that when getting the first sentence from an empty string, an empty string is returned.
        """

        text = ""
        resolver = WikipediaExtrapolator(TF(), Tokenizer(), 0, [ ])
        self.assertEqual(text, resolver._get_first_sentence(text))

    def test_link_frequency(self):
        """
        Test that the link frequency count is accurate.
        """

        links = {
            'a': [ 'x', 'y' ],
            'b': [ 'y', 'z' ],
            'c': [ 'w', 'x', 'y', 'z' ],
        }
        extrapolator = WikipediaExtrapolator([ ], Tokenizer(), TF())
        frequencies = extrapolator._link_frequency(links)
        self.assertEqual(2, frequencies.get('x'))
        self.assertEqual(3, frequencies.get('y'))
        self.assertEqual(2, frequencies.get('z'))
        self.assertEqual(1, frequencies.get('w'))
        self.assertFalse(frequencies.get('a'))

    def test_add_to_graph_low_threshold(self):
        """
        Test adding nodes and edges to a graph with a low threshold.
        """

        graph = nx.Graph()
        links = {
            'Olympique Lyonnais': [ 'Ligue 1', 'AS Monaco' ],
        }

        tokenizer = Tokenizer(stem=True, stopwords=stopwords.words('english'))
        extrapolator = WikipediaExtrapolator([ ], tokenizer, TF())
        extrapolator._add_to_graph(graph, links, threshold=0)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual(2, len(graph.edges))
        self.assertTrue('Olympique Lyonnais' in graph.nodes)
        self.assertTrue(len(graph.nodes['Olympique Lyonnais']['document'].dimensions))
        self.assertTrue('Ligue 1' in graph.nodes)
        self.assertTrue('AS Monaco' in graph.nodes)
        self.assertTrue(('Olympique Lyonnais', 'Ligue 1') in graph.edges)
        self.assertTrue(('Olympique Lyonnais', 'AS Monaco') in graph.edges)
        self.assertFalse(('Ligue 1', 'AS Monaco') in graph.edges)
        self.assertGreater(graph.edges[('Olympique Lyonnais', 'Ligue 1')]['weight'], 0)

    def test_add_to_graph_non_zero_threshold(self):
        """
        Test that adding nodes and edges to a graph with a non-zero threshold excludes some edges.
        """

        graph = nx.Graph()
        links = {
            'Olympique Lyonnais': [ 'Ligue 1', 'AS Monaco' ],
        }

        tokenizer = Tokenizer(stem=True, stopwords=stopwords.words('english'))
        extrapolator = WikipediaExtrapolator([ ], tokenizer, TF())
        extrapolator._add_to_graph(graph, links, threshold=0.4)
        self.assertFalse(('Olympique Lyonnais', 'Ligue 1') in graph.edges)
        self.assertTrue(('Olympique Lyonnais', 'AS Monaco') in graph.edges)

    def test_add_to_graph_high_threshold(self):
        """
        Test that adding nodes and edges to a graph with a high threshold excludes some edges.
        """

        graph = nx.Graph()
        links = {
            'Olympique Lyonnais': [ 'Ligue 1', 'AS Monaco' ],
        }

        tokenizer = Tokenizer(stem=True, stopwords=stopwords.words('english'))
        extrapolator = WikipediaExtrapolator([ ], tokenizer, TF())
        extrapolator._add_to_graph(graph, links, threshold=1)
        self.assertFalse(('Olympique Lyonnais', 'Ligue 1') in graph.edges)
        self.assertFalse(('Olympique Lyonnais', 'AS Monaco') in graph.edges)

    def test_add_to_graph_node_with_no_page(self):
        """
        Test that when adding a node that does not have an article, it is not added to the graph.
        """

        random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(32))
        graph = nx.Graph()
        links = {
            'Olympique Lyonnais': [ 'Ligue 1', random_string ],
        }

        tokenizer = Tokenizer(stem=True, stopwords=stopwords.words('english'))
        extrapolator = WikipediaExtrapolator([ ], tokenizer, TF())
        extrapolator._add_to_graph(graph, links, threshold=1)
        self.assertEqual(2, len(graph.nodes))
        self.assertTrue('Olympique Lyonnais' in graph.nodes)
        self.assertTrue('Ligue 1' in graph.nodes)
        self.assertFalse(random_string in graph.nodes)

    def test_extrapolate_excludes_resolved(self):
        """
        Test that when extrapolating, resolved participants are not returned.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "Chojniczanka Chojnice is a small Polish football team",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]
        extrapolator = WikipediaExtrapolator(corpus, tokenizer, TF(),
                                             first_level_links=5, second_level_links=10)
        participants = extrapolator.extrapolate([ 'Chojniczanka Chojnice', 'Chrobry Głogów' ])
        self.assertFalse('Chojniczanka Chojnice' in participants)
        self.assertFalse('Chrobry Głogów' in participants)

    def test_extrapolate_returns_related_participants(self):
        """
        Test that when extrapolating, related participants are returned.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "The LigaPro is the second-highest division of the Portuguese football league system.",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]
        extrapolator = WikipediaExtrapolator(corpus, tokenizer, TF(),
                                             first_level_links=15, second_level_links=15)
        participants = extrapolator.extrapolate([ 'Associação Académica de Coimbra – O.A.F.',
                                                  'Académico de Viseu F.C.',
                                                  'S.L. Benfica B', 'FC Porto B' ])

        other_participants = [ 'Casa Pia A.C.', 'G.D. Chaves', 'C.D. Cova da Piedade',
                               'S.C. Covilhã', 'G.D. Estoril Praia', 'S.C. Farense',
                               'C.D. Feirense', 'Leixões S.C.', 'C.D. Mafra',
                               'C.D. Nacional', 'U.D. Oliveirense', 'F.C. Penafiel',
                               'Varzim S.C.', 'U.D. Vilafranquense' ]
        self.assertGreaterEqual(len(set(participants).intersection(set(other_participants))), 4)
