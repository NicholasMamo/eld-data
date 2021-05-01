"""
The document frequency (DF) scorer assigns a score to candidate participants based on the number of documents in which they appear.
This approach does not favor candidates if they appear multiple times in the same document.
"""

import math

from ..scorer import Scorer

class DFScorer(Scorer):
    """
    The document frequency scorer counts the number of documents in which each candidate participant appears.
    This becomes the candidate participant's score.
    """

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

        scores = self._sum(candidates, *args, **kwargs)
        return self._normalize(scores) if normalize_scores else scores

    def _sum(self, candidates, *args, **kwargs):
        """
        Score the given candidates based on the number of times they appear—a simple summation.

        :param candidates: A list of candidates participants that were found earlier.
        :type candidates: list

        :return: A dictionary of candidate participants and their scores.
        :rtype: dict
        """

        scores = {}

        """
        Go through each document, and then each of its candidate participants.
        For all of these instances, increment their score.
        """
        for candidate_set in candidates:
            for candidate in list(set(candidate_set)):
                scores[candidate] = scores.get(candidate, 0) + 1

        return scores

    def _normalize(self, scores, *args, **kwargs):
        """
        Normalize the scores.
        The function rescales them between 0 and 1, where 1 is the maximum score of the candidates.

        :param scores: The candidate participants and the number of times that they appeared.
        :type scores: dict

        :return: A dictionary of candidate participants and their associated, normalized scores.
        :rtype: dict
        """

        max_score = max(scores.values()) if len(scores) > 0 else 1
        scores = { candidate: score / max_score for candidate, score in scores.items() }

        return scores
