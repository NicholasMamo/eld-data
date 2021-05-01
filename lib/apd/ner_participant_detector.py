"""
The Named Entity Recognition (NER) participant detector is a simple approach that assumes that all participants are named entities.
This participant detector does not resolve named entities to an alternate representation.
Nor does it extrapolate named entities: it simply extracts named entities and ranks them based on frequency.

.. note::

    Due to its simplicity, the NER participant detector is a good baseline for APD.
    The first paper that proposed APD used named entities as a baseline: `ELD: Event TimeLine Detection -- A Participant-Based Approach to Tracking Events by Mamo et al. (2019) <https://dl.acm.org/doi/abs/10.1145/3342220.3344921>`_.
"""

import os
import sys

path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)

from participant_detector import ParticipantDetector
from extractors.local.entity_extractor import EntityExtractor
from scorers.local.tf_scorer import TFScorer

class NERParticipantDetector(ParticipantDetector):
    """
    The NER participant detector is based on a normal participant detector.
    To extract named entities, the participant detector automatically creates an extractor and a scorer:

        - :class:`~apd.extractors.local.entity_extractor.EntityExtractor` to extract named entities, and
        - :class:`~apd.scorers.local.tf_scorer.TFScorer` to score named entities.

    The NER participant detector performs no filtering, resolution, extrapolation or post-processing.
    """

    def __init__(self):
        """
        Create the NER participant detector.
        """

        super().__init__(EntityExtractor(), TFScorer())
