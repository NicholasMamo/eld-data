"""
The rank filter extracts the top _k_ candidates.
Anything else is filtered out.
"""

import math

from ..filter import Filter

class RankFilter(Filter):
    """
    The rank filter is used to extract the top _k_ candidates.
    The _k_ parameter is passed on to the constructor and stored as an instance variable.
    It is then re-used by the :func:`~apd.filters.local.rank_filter.RankFilter.filter` method.

    :vartype k: The number of candidates to retain.
    :vartype k: int
    """

    def __init__(self, k):
        """
        Create the filter with the k that will decide whether candidate participants will be retained.

        :param k: The number of candidates to retain.
        :type k: int

        :raises ValueError: When _k_ is not an integer.
        :raises ValueError: When _k_ is not positive.
        """

        if type(k) is not int:
            raise ValueError(f"k must be an integer; received {k} ({ type(k) })")

        if k <= 0:
            raise ValueError(f"k must be a positive integer; received {k}")

        self.k = k

    def filter(self, candidates, *args, **kwargs):
        """
        Filter candidate participants that are not credible.
        The function sorts the candidates by score and then retains only the top _k_ candidates.

        :param candidates: A dictionary of candidate participants and their scores.
                           The keys are the candidate names, and the values are their scores.
                           The input candidates should be the product of a :class:`~apd.scorers.scorer.Scorer` process.
        :type candidates: dict

        :return: A dictionary of filtered candidate participants and their associated scores.
        :rtype: dict
        """

        retain = sorted(candidates, key=candidates.get, reverse=True)[:self.k]
        return { candidate: candidates.get(candidate) for candidate in retain }
