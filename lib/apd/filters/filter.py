"""
Scoring is the third step of the APD process.
This step removes candidate participants that are unlikely to be participants.
Possibly, they are removed on the basis of a low score.
The functionality revolves around the :class:`~apd.filters.filter.Filter`'s :func:`~apd.filters.filter.Filter.filter` method.
"""

class Filter(object):
    """
    The filter goes through candidate participants and removes those that are unlikely to be valid participants.

    The functionality revolves around one method: the :func:`~apd.filters.filter.Filter.filter` method.
    The input candidates should be the product of a :class:`~apd.scorers.scorer.Scorer` process.
    In other words, they should be a dictionary, with the keys being the candidates and the values being the score.
    The function returns a subset of the input candidates (and their scores) that passed the filtering test.
    """

    def filter(self, candidates, *args, **kwargs):
        """
        Filter candidate participants that are not credible.
        The basic filter returns all participants.

        :param candidates: A dictionary of candidate praticipants and their scores.
                           The keys are the candidate names, and the values are their scores.
                           The input candidates should be the product of a :class:`~apd.scorers.scorer.Scorer` process.
        :type candidates: dict

        :return: A dictionary of filtered candidate participants and their associated scores.
        :rtype: dict
        """

        return candidates
