"""
The Term Frequency (TF) local term-weighting scheme assigns a weight according to the number of times that a token appears in the list.
The term frequency :math:`tf_{t,d}` of a feature :math:`t` is calculated as:

.. math::

    tf_{t,d} = f_{t,d}

where :math:`f_{t,d}` is the frequency of token :math:`t` in document :math:`d`.
"""

import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from weighting import SchemeScorer

class TF(SchemeScorer):
    """
    The Term Frequency (TF) term-weighting scheme is a simple term weighting schemes that assigns a weight equal to the number of times that the feature appears in a document.
    """

    def score(self, tokens, *args, **kwargs):
        """
        Score the given tokens.
        The score is equal to the frequency of the token in the list.

        :param tokens: The list of tokens to weigh.
        :type tokens: list of str

        :return: A dictionary with the tokens as the keys and the weights as the values.
        :rtype: dict
        """

        weights = { token: tokens.count(token) for token in tokens }
        return weights
