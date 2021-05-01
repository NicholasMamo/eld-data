"""
The ELD participant detector is a complete solution based on the one proposed in `ELD: Event TimeLine Detection -- A Participant-Based Approach to Tracking Events by Mamo et al. (2019) <https://dl.acm.org/doi/abs/10.1145/3342220.3344921>`_.
By default, this participant detector uses the following configuration:

    #. :class:`~apd.extractors.local.twitterner_entity_extractor.TwitterNEREntityExtractor`,
    #. :class:`~apd.scorers.local.tf_scorer.TFScorer`,
    #. :class:`~apd.filters.local.rank_filter.RankFilter` with a ``k`` of 20,
    #. :class:`~apd.resolvers.external.wikipedia_search_resolver.WikipediaSearchResolver`,
    #. :class:`~apd.extrapolators.external.wikipedia_extrapolator.WikipediaExtrapolator`, and
    #. :class:`~apd.postprocessors.external.wikipedia_postprocessor.WikipediaPostprocessor`.

.. note::

    Implementation based on the algorithm outlined in `ELD: Event TimeLine Detection -- A Participant-Based Approach to Tracking Events by Mamo et al. (2019) <https://dl.acm.org/doi/abs/10.1145/3342220.3344921>`_.
    This version contains some changes, such as by using a :class:`~apd.filters.local.rank_filter.RankFilter` instead of a :class:`~apd.filters.local.threshold_filter.ThresholdFilter`.
"""

import os
import sys
from nltk.corpus import stopwords

paths = [ os.path.join(os.path.dirname(__file__)),
           os.path.join(os.path.dirname(__file__), '..') ]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from participant_detector import ParticipantDetector
from scorers.local import TFScorer
from filters.local import RankFilter
from resolvers.external import WikipediaSearchResolver
from extrapolators.external import WikipediaExtrapolator
from postprocessors.external import WikipediaPostprocessor
from nlp.tokenizer import Tokenizer

class ELDParticipantDetector(ParticipantDetector):
    """
    By default the ELD participant detector uses the following configuration:

        #. :class:`~apd.extractors.local.twitterner_entity_extractor.TwitterNEREntityExtractor`,
        #. :class:`~apd.scorers.local.tf_scorer.TFScorer`,
        #. :class:`~apd.filters.local.rank_filter.RankFilter` with a ``k`` of 20,
        #. :class:`~apd.resolvers.external.wikipedia_search_resolver.WikipediaSearchResolver`,
        #. :class:`~apd.extrapolators.external.wikipedia_extrapolator.WikipediaExtrapolator`, and
        #. :class:`~apd.postprocessors.external.wikipedia_postprocessor.WikipediaPostprocessor`.

    However, they can be overriden in the constructor.
    """

    def __init__(self, scheme=None, corpus=None, extractor=None, scorer=None, filter=None,
                 resolver=None, extrapolator=None, postprocessor=None):
        """
        Create the ELD participant detector.
        Any parameter that is not given uses the default configuration.


        :param scheme: The term-weighting scheme to use by the :class:`~apd.resolvers.external.wikipedia_search_resolver.WikipediaSearchResolver` and :class:`~apd.extrapolators.external.wikipedia_extrapolator.WikipediaExtrapolator`.
                       These documents are used to compare the similarity with the domain of the candidates.
        :type scheme: :class:`~nlp.weighting.TermWeightingScheme`
        :param corpus: The corpus of documents.
                       These documents may be tokenized already, but this class re-tokenizes them with its own :class:`~nlp.tokenizer.Tokenizer`.
        :type corpus: list of :class:`~nlp.document.Document`
        :param extractor: The participant detector's extractor.
                          This component is used to find candidate participants.
                          If it is not given, it defaults to the :class:`~apd.extractors.local.twitterner_entity_extractor.TwitterNEREntityExtractor`.
        :type extractor: None :class:`~apd.extractors.extractor.Extractor`
        :param scorer: The participant detector's scorer.
                       This component is used to give a score to the extractor's candidate participants.
                       If it is not given, it defaults to the :class:`~apd.scorers.local.tf_scorer.TFScorer`.
        :type scorer: None or :class:`~apd.scorers.scorer.Scorer`
        :param filter: The participant detector's filter.
                       This component is used to exclude candidates that are unlikely to be participants.
                       If it is not given, it defaults to the :class:`~apd.filters.local.rank_filter.RankFilter` with a ``k`` of 20.
        :type filter: None or :class:`~apd.filters.filter.Filter`
        :param resolver: The participant detector's resolver.
                         This component looks for the real keywords associated with a participant.
                         If it is not given, it defaults to the :class:`~apd.resolvers.external.wikipedia_search_resolver.WikipediaSearchResolver`.
        :type resolver: None or :class:`~apd.resolvers.resolver.Resolver`
        :param extrapolator: The participant detector's extrapolator.
                             This component looks for additional participants that might not be in the corpus.
                             If it is not given, it defaults to the :class:`~apd.extrapolators.external.wikipedia_extrapolator.WikipediaExtrapolator`.
        :type extrapolator: None or :class:`~apd.extrapolators.extrapolator.Extrapolator`
        :param postprocessor: The participant detector's postprocessor.
                              This component modifies the found participants.
                              If it is not given, it defaults to the :class:`~apd.postprocessors.external.wikipedia_postprocessor.WikipediaPostprocessor`.
        :type postprocessor: None or :class:`~apd.postprocessors.postprocessor.Postprocessor`
        """

        """
        Tokenize the corpus.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words('english'),
                              normalize_words=True, character_normalization_count=3,
                              remove_unicode_entities=True)
        if scheme:
            corpus = self._tokenize_corpus(corpus, scheme, tokenizer)

        """
        Set up the ELD participant detector.
        """
        if not extractor:
            from extractors.local.twitterner_entity_extractor import TwitterNEREntityExtractor
            extractor = TwitterNEREntityExtractor()
        scorer = scorer or TFScorer()
        filter = filter or RankFilter(20)
        resolver = resolver or WikipediaSearchResolver(scheme, tokenizer, 0.05, corpus)
        extrapolator = extrapolator or WikipediaExtrapolator(corpus, tokenizer, scheme, 0.05,
                                                             first_level_links=100, second_level_links=500)
        postprocessor = postprocessor or WikipediaPostprocessor()

        super().__init__(extractor, scorer, filter, resolver, extrapolator, postprocessor)

    def _tokenize_corpus(self, corpus, scheme, tokenizer):
        """
        Tokenize the corpus.

        :param corpus: The corpus of documents to tokenize.
        :type corpus: list of :class:`~nlp.document.Document`
        :param scheme: The term-weighting scheme to use by the :class:`~apd.resolvers.external.wikipedia_search_resolver.WikipediaSearchResolver` and :class:`~apd.extrapolators.external.wikipedia_extrapolator.WikipediaExtrapolator`.
                       These documents are used to compare the similarity with the domain of the candidates.
        :type scheme: :class:`~nlp.weighting.TermWeightingScheme`
        :param tokenizer: The tokenizer to use to create the new documents.
        :type tokenizer: :class:`~nlp.tokenizer.Tokenizer`

        :return: The corpus, tokenized anew.
        :rtype: list of :class:`~nlp.document.Document`
        """

        corpus = [ scheme.create(text=document.text, tokens=tokenizer.tokenize(document.text))
                   for document in corpus ]
        for document in corpus:
            document.normalize()
        return corpus
