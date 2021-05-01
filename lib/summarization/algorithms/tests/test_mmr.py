"""
Run unit tests on Carbonell and Goldstein (1998)'s algorithm.
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
from summarization import Summary
from summarization.algorithms import MMR
from vsm import vector_math

class TestMMR(unittest.TestCase):
    """
    Test Carbonell and Goldstein (1998)'s algorithm.
    """

    def test_summarize_empty(self):
        """
        Test that when summarizing an empty set of documents, an empty summary is returned.
        """

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
        length = sum(len(str(document)) for document in corpus)
        summary = algo.summarize(corpus, length)
        self.assertTrue(len(set(corpus).difference(set(summary.documents))))
        self.assertLessEqual(len(str(summary)), length)

    def test_summarize_exact_summary_length(self):
        """
        Test that when summarizing a set of documents and the length is equal to the projected summary length, including spaces, all documents are included in the summary.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = MMR()
        length = sum(len(str(document)) for document in corpus) + len(corpus) - 1
        summary = algo.summarize(corpus, length)
        self.assertEqual(set(corpus), set(summary.documents))
        self.assertEqual(len(str(summary)), length)

    def test_summarize_long_summary_length(self):
        """
        Test that when summarizing a set of documents and a long length is given, all documents are included in the summary.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is a pipe.', { 'pipe': 1 }),
                    Document('this is a cigar.', { 'cigar': 1 }),
                   Document('this is a cigar and this is a pipe.', { 'cigar': 1, 'pipe': 1 }), ]
        for document in corpus:
            document.normalize()

        algo = MMR()
        length = sum(len(str(document)) for document in corpus) + len(corpus)
        summary = algo.summarize(corpus, length)
        self.assertEqual(set(corpus), set(summary.documents))
        self.assertLessEqual(len(str(summary)), length)

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

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
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
        algo = MMR()
        self.assertRaises(ValueError, algo.summarize, c, -1)

    def test_zero_length(self):
        """
        Test that when providing a length of zero, the function raises a ValueError.
        """

        c = [ ]
        algo = MMR()
        self.assertRaises(ValueError, algo.summarize, c, 0)

    def test_negative_lambda(self):
        """
        Test that when lambda is negative, the function raises a ValueError.
        """

        c = [ ]
        self.assertRaises(ValueError, MMR, l=-0.1)

    def test_large_lambda(self):
        """
        Test that when lambda is large, the function raises a ValueError.
        """

        c = [ ]
        self.assertRaises(ValueError, MMR, l=1.1)

    def test_compute_query_empty(self):
        """
        Test that the query is empty when no documents are given.
        """

        algo = MMR()
        self.assertEqual({ }, algo._compute_query([ ]).dimensions)

    def test_compute_query_one_document(self):
        """
        Test that the query construction is identical to the given document when there is just one document.
        """

        d = Document('this is not a pipe', { 'pipe': 1 })
        algo = MMR()
        self.assertEqual(d.dimensions, algo._compute_query([ d ]).dimensions)

    def test_compute_query_normalized(self):
        """
        Test that the query is normallized.
        """

        d = Document('this is not a pipe', { 'this': 1, 'pipe': 1 })
        d.normalize()
        algo = MMR()
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

        algo = MMR()
        self.assertEqual(round(math.sqrt(2)/2., 10), round(algo._compute_query(corpus).dimensions['this'], 10))
        self.assertEqual(round(math.sqrt(2)/2., 10), round(algo._compute_query(corpus).dimensions['pipe'], 10))
        self.assertEqual(1, round(vector_math.magnitude(algo._compute_query(corpus)), 10))

    def test_compute_similarity_matrix_documents_empty(self):
        """
        Test that the cimilarity matrix has only one row and column when no documents are given.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = MMR()
        query = algo._compute_query(corpus)
        matrix = algo._compute_similarity_matrix([ ], query)
        self.assertEqual(1, len(matrix))
        self.assertEqual(1, len(matrix[query]))

    def test_compute_similarity_matrix_documents_unchanged(self):
        """
        Test that the documents given to the similarity matrix are unchanged.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = MMR()
        query = algo._compute_query(corpus)
        copy = list(corpus)
        matrix = algo._compute_similarity_matrix(corpus, query)
        self.assertEqual(copy, corpus)

    def test_compute_similarity_matrix_query_unchanged(self):
        """
        Test that the query given to the similarity matrix is unchanged.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = MMR()
        query = algo._compute_query(corpus)
        copy = query.copy()
        matrix = algo._compute_similarity_matrix(corpus, query)
        self.assertEqual(copy.dimensions, query.dimensions)

    def test_compute_similarity_matrix_diagonal(self):
        """
        Test that the diagonal of a similarity matrix is 1.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = MMR()
        query = algo._compute_query(corpus)

        matrix = algo._compute_similarity_matrix(corpus, query)
        self.assertTrue(all(round(matrix[document][document], 10) == 1 for document in corpus + [query]))

    def test_compute_similarity_matrix_symmetrical(self):
        """
        Test that the similarity matrix is symmetrical.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()

        algo = MMR()
        query = algo._compute_query(corpus)

        matrix = algo._compute_similarity_matrix(corpus, query)
        self.assertTrue(all(matrix[document][other] == matrix[other][document]
                            for document in corpus + [query]
                            for other in matrix[document]))

    def test_filter_documents_empty(self):
        """
        Test that when filtering an empty list of documents, an empty list is returned.
        """

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
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

        algo = MMR()
        self.assertEqual(corpus[1:], algo._filter_documents(corpus, Summary(), len(str(corpus[1]))))

    def test_get_next_document_empty(self):
        """
        Test that when getting the next document from an empty list, ``None`` is returned.
        """

        algo = MMR(0.5)
        self.assertEqual(None, algo._get_next_document([ ], Summary(), Document(), { }))

    def test_get_next_document_empty_summary(self):
        """
        Test that when the summary is empty, the next document is the one that is most similar to the query.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Document('cigars', [ 'cigar' ], scheme=TF())
        query.normalize()

        algo = MMR(0.5)
        matrix = algo._compute_similarity_matrix(corpus, query)
        self.assertEqual(corpus[0], algo._get_next_document(corpus, Summary(), query, matrix))

    def test_get_next_document_redundancy(self):
        """
        Test that redundancy affects which document is chosen.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }),
                   Document('this is a pipe and a cigar', { 'this': 1, 'pipe': 1, 'cigar': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Document('cigars', [ 'pipe', 'cigar' ], scheme=TF())
        query.normalize()

        algo = MMR(0.5)
        matrix = algo._compute_similarity_matrix(corpus, query)
        self.assertEqual(corpus[1], algo._get_next_document(corpus[1:], Summary(corpus[0]), query, matrix))

    def test_get_next_document_redundancy_only(self):
        """
        Test that when only redundancy is considered, repeat documents are filtered out.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }),
                   Document('this is a pipe and a cigar', { 'this': 1, 'pipe': 1, 'cigar': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Document('cigars', [ 'cigar' ], scheme=TF())
        query.normalize()

        algo = MMR(0)
        matrix = algo._compute_similarity_matrix(corpus, query)
        self.assertEqual(corpus[1], algo._get_next_document(corpus[1:], Summary(corpus[0]), query, matrix))

    def test_get_next_document_relevance_only(self):
        """
        Test that when only relevance is considered, repeat documents are not filtered out.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }),
                   Document('this is a pipe and a cigar', { 'this': 1, 'pipe': 1, 'cigar': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Document('cigars', [ 'cigar' ], scheme=TF())
        query.normalize()

        algo = MMR(1)
        matrix = algo._compute_similarity_matrix(corpus, query)
        self.assertEqual(corpus[2], algo._get_next_document(corpus[1:], Summary(corpus[0]), query, matrix))

    def test_get_next_document_documents_unchanged(self):
        """
        Test that when getting the next document, the documents are unchanged.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }),
                   Document('this is a pipe and a cigar', { 'this': 1, 'pipe': 1, 'cigar': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Document('cigars', [ 'cigar' ], scheme=TF())
        query.normalize()

        algo = MMR(1)
        matrix = algo._compute_similarity_matrix(corpus, query)
        copy = list(corpus)
        algo._get_next_document(corpus[1:], Summary(corpus[0]), query, matrix)
        self.assertEqual(copy, corpus)

    def test_get_next_document_query_unchanged(self):
        """
        Test that when getting the next document, the query is unchanged.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }),
                   Document('this is a pipe and a cigar', { 'this': 1, 'pipe': 1, 'cigar': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Document('cigars', [ 'cigar' ], scheme=TF())
        query.normalize()

        algo = MMR(1)
        matrix = algo._compute_similarity_matrix(corpus, query)
        copy = query.copy()
        algo._get_next_document(corpus[1:], Summary(corpus[0]), query, matrix)
        self.assertEqual(copy.dimensions, query.dimensions)

    def test_get_next_document_matrix_unchanged(self):
        """
        Test that when getting the next document, the matrix is unchanged.
        """

        """
        Create the test data.
        """
        corpus = [ Document('this is not a cigar', { 'this': 1, 'cigar': 1 }),
                    Document('this is a pipe', { 'this': 1, 'pipe': 1 }),
                   Document('this is a pipe and a cigar', { 'this': 1, 'pipe': 1, 'cigar': 1 }) ]
        for document in corpus:
            document.normalize()
        query = Document('cigars', [ 'cigar' ], scheme=TF())
        query.normalize()

        algo = MMR(1)
        matrix = algo._compute_similarity_matrix(corpus, query)
        copy = dict(matrix)
        algo._get_next_document(corpus[1:], Summary(corpus[0]), query, matrix)
        self.assertEqual(copy, matrix)

    def test_worked_example(self):
        """
        Test the `worked example <http://www.cs.bilkent.edu.tr/~canf/CS533/hwSpring14/eightMinPresentations/handoutMMR.pdf>`_.
        """

        """
        Create the test data.
        """
        d = [ Document('', {}) for i in range(5) ]
        q = Document('')
        matrix = {
            d[0]: { d[0]: 1.00, d[1]: 0.11, d[2]: 0.23, d[3]: 0.76, d[4]: 0.25, q: 0.91, },
            d[1]: { d[0]: 0.11, d[1]: 1.00, d[2]: 0.29, d[3]: 0.57, d[4]: 0.51, q: 0.90, },
            d[2]: { d[0]: 0.23, d[1]: 0.29, d[2]: 1.00, d[3]: 0.02, d[4]: 0.20, q: 0.50, },
            d[3]: { d[0]: 0.76, d[1]: 0.57, d[2]: 0.02, d[3]: 1.00, d[4]: 0.33, q: 0.06, },
            d[4]: { d[0]: 0.25, d[1]: 0.51, d[2]: 0.20, d[3]: 0.33, d[4]: 1.00, q: 0.63, },
            q:    { d[0]: 0.91, d[1]: 0.90, d[2]: 0.50, d[3]: 0.06, d[4]: 0.63, q: 1.00, },
        }

        algo = MMR(0.5)

        """
        Run the first iteration.
        """
        scores = algo._compute_scores(d, Summary(), q, matrix)
        self.assertEqual(0.91, scores[d[0]])
        self.assertEqual(d[0], max(scores.keys(), key=scores.get))

        """
        Run the second iteration.
        """
        scores = algo._compute_scores(d[1:], Summary(d[0]), q, matrix)
        self.assertEqual(0.395, scores[d[1]])
        self.assertEqual(0.135, scores[d[2]])
        self.assertEqual(-0.35, scores[d[3]])
        self.assertEqual(0.19, scores[d[4]])

        """
        Run the third iteration.
        """
        scores = algo._compute_scores(d[2:], Summary(d[:2]), q, matrix)
        self.assertEqual(0.105, round(scores[d[2]], 10))
        self.assertEqual(-0.35, scores[d[3]])
        self.assertEqual(0.06, scores[d[4]])
        self.assertEqual(0.63, round(matrix[d[0]][d[1]] +
                                     matrix[d[0]][d[2]] +
                                     matrix[d[1]][d[2]], 10))

        """
        Run the case with :math:`\\lambda` = 1.
        """
        algo = MMR(1)
        scores = algo._compute_scores(d[2:], Summary(d[:2]), q, matrix)
        self.assertEqual(d[4], max(scores.keys(), key=scores.get))
        self.assertEqual(0.87, round(matrix[d[0]][d[1]] +
                                     matrix[d[0]][d[4]] +
                                     matrix[d[1]][d[4]], 10))
