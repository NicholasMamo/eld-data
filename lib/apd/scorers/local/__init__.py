"""
Local scorers assign a score to candidate participants based on evidence from the event's corpus.
"""

from .df_scorer import DFScorer
from .log_df_scorer import LogDFScorer
from .log_tf_scorer import LogTFScorer
from .tf_scorer import TFScorer
from .tfidf_scorer import TFIDFScorer
