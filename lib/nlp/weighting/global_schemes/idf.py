"""
The Inverse Document Frequency (IDF) global term-weighting scheme penalizes common terms when assigning weight.
The reasoning is that tokens that are very common in a corpus are not informative generally.
On the other hand, terms that appear often in one document but rarely outside characterize that document.

The IDF :math:`idf_{t}` for term :math:`t` is computed as follows:

.. math::

    idf_{t} = \\log{\\frac{N}{n_t}}

where :math:`N` is the total number of documents in the corpus, and :math:`n_t` is the total number of documents that contain term :math:`t`.
"""

import math
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from objects.exportable import Exportable
from nlp.weighting import SchemeScorer

class IDF(Exportable, SchemeScorer):
    """
    The Inverse Document Frequency (IDF) is one of the most common term-weighting schemes.
    This scheme promotes tokens that are common in one document, and uncommon in the rest of the corpus.
    As part of the calculation, the IDF scheme needs two pieces of information:

    1. The number of documents in which tokens appear, and
    2. The number of documents in the corpus.

    .. note::

        In order to mitigate for tokens that are not present in the corpus, this class uses Laplace smoothing.

    In addition to the standard :func:`~nlp.weighting.global_schemes.idf.IDF.score` function, this class also the :func:`~nlp.weighting.global_schemes.idf.IDF.from_documents` function.
    This function receives a corpus and constructs the IDF table from it.
    The number of documents in the corpus, which is required by this class, can be extracted trivially.

    :ivar ~.idf: The IDF table used in conjunction with term weighting.
    :vartype idf: dict
    :ivar documents: The number of documents in the corpus.
    :vartype documents: int
    """

    def __init__(self, idf, documents):
        """
        Create the term-weighting scheme with the IDF table and the number of documents in that scheme.

        :param idf: The IDF table used in conjunction with term weighting.
                    The keys are the terms, and the corresponding values are the number of documents in which they appear.
        :type idf: dict
        :param documents: The number of documents in the IDF table.
        :type documents: int

        :raises ValueError: When the document frequency of a term is higher than the number of the IDF documents.
        :raises ValueError: When the document frequency of a term is negative.
        :raises ValueError: When the number of documents is negative.
        """

        if max(idf.values()) > documents:
            raise ValueError(f"The number of documents ({documents}) must be greater or equal to the most common term ({max(idf.values())})")

        if min(idf.values()) < 0:
            raise ValueError("The IDF values must be non-negative")

        if documents < 0:
            raise ValueError("The number of documents in the IDF must be non-negative")

        self.idf = idf
        self.documents = documents

    def score(self, tokens):
        """
        Score the given tokens based on the number of documents they appear.
        Tokens that appear in many documents receive a low score, whereas those that do not receive a high score.

        Since the denominator is 0 if the token does not appear in the corpus, this function uses Laplace smoothing.

        :param tokens: The list of tokens to weigh.
        :type tokens: list of str

        :return: A dictionary with the tokens as the keys and the weights as the values.
        :rtype: dict
        """

        weights = { token: math.log(self.documents / (self.idf.get(token, 0) + 1), 10) for token in tokens }
        return weights

    def to_array(self):
        """
        Export the IDF as an associative array.

        :return: The IDF as an associative array.
        :rtype: dict
        """

        return {
            'class': str(IDF),
            'documents': self.documents,
            'idf': self.idf,
        }

    @staticmethod
    def from_array(array):
        """
        Create an instance of the IDF from the given associative array.

        :param array: The associative array with the attributes to create the IDF.
        :type array: dict

        :return: A new instance of the IDF with the same attributes stored in the object.
        :rtype: :class:`~nlp.weighting.global_schemes.idf.IDF`
        """

        return IDF(documents=array.get('documents'), idf=array.get('idf'))

    @staticmethod
    def from_documents(documents):
        """
        Create the IDF table from the given set of documents.
        This function extracts all unique terms from the documents' dimensions and counts the number of documents in which they appear at least once.


        .. note::

            This function does not return the number of documents, but only the IDF table.
            The number of documents can be extracted easily using the :func:`len` function.

        :param documents: The documents from which the IDF table will be created.
        :type documents: list of :class:`~nlp.document.Document`

        :return: A dictionary, where the keys are the document tokens and the values are their document frequency.
        :rtype: dict
        """

        idf = { }

        for document in documents:
            for dimension in document.dimensions:
                idf[dimension] = idf.get(dimension, 0) + 1

        return idf
