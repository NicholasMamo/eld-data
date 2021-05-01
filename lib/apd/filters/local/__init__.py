"""
Local filtering strategies consider only the scores of candidate participants to decide whether they are credible.
Therefore these kind of filters are usually very limited in what they can do.
"""

from .rank_filter import RankFilter
from .threshold_filter import ThresholdFilter
