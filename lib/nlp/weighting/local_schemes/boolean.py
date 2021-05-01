"""
The boolean term-weighting scheme is a simple scheme that gives a score of 1 if the term appears in the list of tokens, and 0 otherwise:

.. math::

    bool_{t,d} = \\begin{cases}
                     1 & \\text{if } t \\in d \\\\
                     0 & \\text{otherwise}
                 \\end{cases}
"""

import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from weighting import SchemeScorer

class Boolean(SchemeScorer):
    """
    The boolean term-weighting scheme is a simple scheme that gives a score of 1 if the term appears in the list of tokens, and 0 otherwise.
    In reality, this term-weighting scheme is not aware of all possible tokens.
    Therefore it only gives a score of 1 if the term appears in the list of tokens.
    The :class:`~nlp.weighting.TermWeightingScheme` automatically assumes the score of a token is 0 if it is not set.
    """

    def score(self, tokens, *args, **kwargs):
        """
        Score the given tokens.
        The score is 1 if a feature appears in the document, 0 otherwise.

        :param tokens: The list of tokens to weigh.
        :type tokens: list of str

        :return: A dictionary with the tokens as the keys and the weights as the values.
        :rtype: dict
        """

        weights = { token: 1 for token in list(set(tokens)) }
        return weights
