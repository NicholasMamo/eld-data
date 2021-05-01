"""
The logarithmic term scorer is based on the :class:`~apd.scorers.local.tf_scorer.TFScorer`.
Like the term scorer, it favors candidate participants that appear several times in the same document.

The difference is that the logarithmic version decreases the difference between candidates that appear very often and those which do not.
The intuition is that if a candidate participants appears 1000 times and another appears 1200 times, the difference is not significant.
"""

import math

from .tf_scorer import TFScorer

class LogTFScorer(TFScorer):
    """
    The logarithmic term frequency scorer is based on the normal term frequency summation.
    The difference is that before normalization, the scorer takes the the logarithm of the scores.
    In this way, the scores are not overly-biased towards candidate participants that appear disproportionately.
    To calculate the logarithm, the scorer accepts the logarithmic base in the constructor.

    :ivar base: The base of the logarithm.
    :vartype base: int
    """

    def __init__(self, base=10):
        """
        Create the scorer.

        :param base: The base of the logarithm.
        :type base: int
        """

        self.base = base

    def score(self, candidates, normalize_scores=True, *args, **kwargs):
        """
        Score the given candidates based on their relevance within the corpus.
        The score is normalized using the maximum score.

        :param candidates: A list of candidates participants that were found earlier.
        :type candidates: list
        :param normalize_scores: A boolean indicating whether the scores should be normalized.
                                 Here, normalization means rescaling between 0 and 1.
        :type normalize_scores: bool

        :return: A dictionary of participants and their associated scores.
        :rtype: dict
        """

        scores = self._sum(candidates)
        scores = { candidate: math.log(score + 1, self.base) for candidate, score in scores.items() } # apply Laplace smoothing
        return self._normalize(scores) if normalize_scores else scores
