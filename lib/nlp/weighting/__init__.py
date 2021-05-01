"""
Term-weighting schemes are responsible for assigning importance to terms.
Term-weighting schemes are made up of:

1. A local component, which scores terms based on their appearance in a document.
2. A global component, which scores terms based on their appearance in a corpus.
3. Optionally, a normalization component to make all documents similar.

This package represents term-weighting schemes in the same way, but without normalization.
So, a :class:`~nlp.weighting.TermWeightingScheme` has two components:

1. A local component (you can read more about them :ref:`here <nlp_local>`), and
2. A global component (you can read more about them :ref:`here <nlp_local>`).

The :class:`~nlp.weighting.TermWeightingScheme`'s :func:`~nlp.weighting.TermWeightingScheme.create` uses its local and global term-weighting schemes to create a :class:`~nlp.document.Document`.
This function combines the score using a simple product.

You can combine any local term-weighting scheme with any other global term-weighting scheme.
All local and global term-weighting schemes inherit the :class:`~nlp.weighting.SchemeScorer`.
That means all term-weighting schemes must implement the :class:`~nlp.weighting.SchemeScorer`'s :func:`~nlp.weighting.score` to assign a score to terms.

.. note::

    Normally, you don't have to create your own term-weighting schemes.
    EvenTDT includes some of the most common term-weighting schemes to help you get started faster.
    You can read more about these readily-available term-weighting schemes :ref:`here <nlp_common>`.

.. note::

    You can read more about term-weighting schemes in `Term-Weighting Approaches in Automatic Text Retrieval by Salton and Buckley (1998) <https://www.sciencedirect.com/science/article/abs/pii/0306457388900210>`_.
"""

from abc import ABC, abstractmethod

import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp import Document

class TermWeightingScheme(object):
    """
    A term-weighting scheme is made up of a local term-weighting scheme and a global term-weighting scheme.
    Each one gives a score to a token, and the term-weighting scheme combines the scores into one weight.

    This class stores the local and global term-weighting schemes.
    You can provide any combination of schemes in the constructor.
    After creating a term-weighting scheme, use the :func:`~nlp.weighting.TermWeightingScheme.create` function to convert tokens into a document.

    :ivar local_scheme: The local term-weighting scheme.
    :vartype local_scheme: :class:`~nlp.weighting.SchemeScorer`
    :ivar global_scheme: The global term-weighting scheme.
    :vartype global_scheme: :class:`~nlp.weighting.SchemeScorer`
    """

    def __init__(self, local_scheme, global_scheme):
        """
        A term-weighting scheme is made up of a local component and a global component.
        Each term-weighting scheme contributes a score, which when multiplied give the final token weight.

        :param local_scheme: The local term-weighting scheme.
        :type local_scheme: :class:`~nlp.weighting.SchemeScorer`
        :param global_scheme: The global term-weighting scheme.
        :type global_scheme: :class:`~nlp.weighting.SchemeScorer`
        """

        self.local_scheme = local_scheme
        self.global_scheme = global_scheme

    def create(self, tokens, *args, **kwargs):
        """
        Create a :class:`~nlp.document.Document` from the given tokens.
        This function calls the :func:`~nlp.weighting.SchemeScorer.score` function on the local and global term-weighting schemes.
        Then, it multiplies them together to get the final token weight.

        The function accepts other arguments or keyword arguments and passes them on to the :class:`~nlp.document.Document` constructor.
        You can use these arguments to instantiate the :class:`~nlp.document.Document` with ``text`` or ``attributes``.

        :param tokens: A list of tokens that can be converted into dimensions.
        :type tokens: list of str

        :return: A new document with the given tokens as the dimensions.
                 The dimensions' weights depend on the local and global term-weighting scheme.
        :rtype: :class:`~nlp.document.Document`
        """

        local_score, global_score = self.local_scheme.score(tokens), self.global_scheme.score(tokens)
        dimensions = {}
        for dimension in local_score.keys() | global_score.keys():
            dimensions[dimension] = local_score.get(dimension, 0) * global_score.get(dimension, 0)

        return Document(dimensions=dimensions, *args, **kwargs)

class SchemeScorer(ABC):
    """
    The :class:`~nlp.weighting.TermWeightingScheme` combines two :class:`~nlp.weighting.SchemeScorer` scores to weight tokens.
    This class defines the functionality that all local and global term-weighting schemes must have at least.

    Essentially, the :func:`~nlp.weighting.SchemeScorer.score` function needs to be implemented.
    This function receives a list of tokens and assigns them a score.

    .. note::

        It is important to distinguish between :class:`~nlp.weighting.TermWeightingScheme` and :class:`~nlp.weighting.SchemeScorer`.
        The former is a complete term-weighting scheme that takes local and global scheme scorers and combines their token scores.
        The latter is the component that actually scores tokens.
    """

    @abstractmethod
    def score(self, tokens, *args, **kwargs):
        """
        Score the given list of tokens.

        :param tokens: The list of tokens to weigh.
        :type tokens: list of str

        :return: A dictionary with the tokens as the keys and the weights as the values.
        :rtype: dict
        """

        pass

from .tf import TF
from .tfidf import TFIDF
