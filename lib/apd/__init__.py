"""
Automatic Participant Detection (APD) is a research area originally proposed in `ELD: Event TimeLine Detection -- A Participant-Based Approach to Tracking Events by Mamo et al. (2019) <https://dl.acm.org/doi/abs/10.1145/3342220.3344921>`_.
In the paper, Mamo et al. describe APD as a way of extracting event participants before the event starts.
With this in mind, APD is a six-step process that looks for participants that are mentioned in the event's corpus, and others that may have been missed.

APD's first three steps identify participants from the event's corpus.
Resolution then maps these participants to alternative, possibly semantic, representations.
Extrapolation looks for related participants, which might be too unpopular to capture immediately from the event's corpus.
The entire process is as follows:

   #. :class:`Extract <apd.extractors.extractor>` candidate participants,
   #. :class:`Score <apd.scorers.scorer>` the candidates,
   #. :class:`Filter <apd.filters.filter>` out low-scoring candidates,
   #. :class:`Resolve <apd.resolvers.resolver>` the candidates to an alternative representation (such as a Wikipedia concept) to make them participants,
   #. :class:`Extrapolate <apd.extrapolators.extrapolator>` new participants, and
   #. :class:`Post-process <apd.postprocessors.postprocessor>` the final list of participants.

The APD process revolves around a central class: the :class:`~apd.participant_detector.ParticipantDetector`.
The class constructor accepts classes representing these six steps, and calls their main functions.
All of the step implementations are separated into local (using the corpus) or external (using a different source) methods.

Each step is represented by a base class.
Base classes define the minimum inputs and describe the expected outputs of each step.
Each base class also has a central function around which processing revolves.
For example, the :class:`~apd.extractors.extractor.Extractor` class has a :func:`~apd.extractors.extractor.Extractor.extract` function.

All APD functionality usually goes through the :class:`~apd.participant_detector.ParticipantDetector`.
This class represents APD's six steps.
It takes as inputs instances of the classes and calls them one after the other.
"""

from .participant_detector import ParticipantDetector
from .eld_participant_detector import ELDParticipantDetector
