"""
The filler global term-weighting scheme is used when there is no need for a global term-weighting scheme.
The scheme assigns a score of 1 to all tokens such that it does not influence their overall score assigned by the :class:`~nlp.weighting.TermWeightingScheme`.
"""

import math
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from weighting import SchemeScorer

class Filler(SchemeScorer):
    """
    The filler global term-weighting scheme is used when there is no need for a global term-weighting scheme because it does not change the overall weight of tokens.
    """

    def score(self, tokens):
        """
        Give a constant score the given tokens.
        The chosen constant is 1 so that the :class:`~nlp.weighting.TermWeightingScheme` is not influenced at al..

        :param tokens: The list of tokens to weigh.
        :type tokens: list of str

        :return: A dictionary with the tokens as the keys and the weights as the values.
        :rtype: dict
        """

        weights = { token: 1 for token in list(set(tokens)) }
        return weights
