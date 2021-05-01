"""
The Zhao et al. consumer mimicks the implementation by the same authors.
It revolves around the :class:`~tdt.algorithms.zhao.Zhao` TDT algorithm.

This consumer is concerned only with the TDT approach.
The summarization in the :class:`~queues.consumers.zhao_consumer.ZhaoConsumer` uses the :class:`~summarization.algorithms.mmr.MMR` approach.

The :class:`~queues.consumers.zhao_consumer.ZhaoConsumer` uses a dynamic, sliding time-window to detect topics.
It splits each time-window into two halves, and checks whether the second half has experienced a surge in tweets.
If it has, then that half-time-window represents a topic.
Otherwise, a larger time-window is used.
You can read more about this approach in the :class:`~tdt.algorithms.zhao.Zhao` class.

.. note::

    This implementation is based on the algorithm presented in `Human as Real-Time Sensors of Social and Physical Events: A Case Study of Twitter and Sports Games by Zhao et al. (2011) <https://arxiv.org/abs/1106.4300>`_.

    This consumer assumes that topics captured within 90 seconds of each other belong to the same topic.
    This is based on the :class:`summarization.timeline.Timeline`'s expiry.
    However, there is no additional tracking.
"""

from datetime import datetime

import asyncio
import math
import os
import sys
import time

from nltk.corpus import stopwords

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from queues.consumers.buffered_consumer import SimulatedBufferedConsumer

from nlp.document import Document
from nlp.tokenizer import Tokenizer

from logger import logger

from summarization.algorithms import MMR
from summarization.timeline import Timeline
from summarization.timeline.nodes import DocumentNode

from tdt.algorithms import Zhao
from tdt.nutrition import MemoryNutritionStore

import twitter

class ZhaoConsumer(SimulatedBufferedConsumer):
    """
    The Zhao et al. consumer is based on the implementation by the same authors.
    The algorithm revolves around the :class:`~tdt.algorithms.zhao.Zhao` algorithm.
    The algorithm examines changes in volume using a dynamic, sliding time-window.

    The consumer naturally stores its periodicity, which defines the smallest dynamic sliding time-window.
    Apart from that, it stores a :class:`~tdt.nutrition.NutritionStore`, which records the tweeting volume at each second, and a :class:`~nlp.weighting.TermWeightingScheme` to create documents.

    Finally, the consumer also maintains a list of :class:`~nlp.document.Document` instances that cover the largest sliding time-window.
    These documents can be used later for summarization by the :class:`~summarization.algorithms.mmr.MMR` algorithm.

    :ivar store: The nutrition store used to store the volume.
    :vartype store: :class:`~tdt.nutrition.store.NutritionStore`
    :ivar ~.scheme: The term-weighting scheme used to create documents.
    :vartype ~.scheme: :class:`~nlp.weighting.TermWeightingScheme`
    :ivar ~.documents: The documents that can still be used for summarization.
                       Older documents are automatically cleared.
    :vartype ~.documents: :class:`~nlp.document.Document`
    :ivar tdt: The TDT algorithm: Zhao et al.'s implementation.
    :vartype tdt: :class:`~tdt.algorithms.zhao.Zhao`
    :ivar summarization: The summarization algorithm to use.
    :vartype summarization: :class:`~summarization.algorithms.mmr.MMR`
    """

    def __init__(self, queue, periodicity=5, scheme=None, post_rate=1.7, *args, **kwargs):
        """
        Create the consumer with a :class:`~queues.Queue`.
        Simultaneously create a :class:`~tdt.nutrition.NutritionStore` and the :class:`~tdt.algorithms.zhao.Zhao` TDT algorithm.

        Additional arguments and keyword arguments are passed on to the base class.

        :param queue: The queue that the consumer will consume later.
        :type queue: :class:`~queues.Queue`
        :param periodicity: The time-window in seconds of the buffered consumer, or how often it is invoked.
                            This defaults to 5 seconds, the same span as half the smallest time-window in Zhao et al.'s algorithm.
        :type periodicity: int
        :param scheme: The term-weighting scheme that is used to create dimensions.
                       If ``None`` is given, the :class:`~nlp.weighting.tf.TF` term-weighting scheme is used.
        :type scheme: None or :class:`~nlp.weighting.TermWeightingScheme`
        :param post_rate: The minimum increase between the two halves of the sliding time window to represent a burst.
        :type post_rate: float
        """

        super(ZhaoConsumer, self).__init__(queue, periodicity, *args, **kwargs)
        self.scheme = scheme
        self.store = MemoryNutritionStore()
        self.documents = { }
        self.tdt = Zhao(self.store, post_rate)
        self.summarization = MMR()

    async def _process(self):
        """
        Find breaking develpoments based on changes in volume.

        :return: The constructed timeline.
        :rtype: :class:`~summarization.timeline.Timeline`
        """

        timeline = Timeline(DocumentNode, expiry=90, min_similarity=1)

        while self.active:
            if self.buffer.length() > 0:
                """
                If there are tweets in the buffer, dequeue them and convert them into documents.
                """
                tweets = self.buffer.dequeue_all()
                documents = self._to_documents(tweets)
                latest_timestamp = self._latest_timestamp(documents)

                """
                Add the received documents to the document list.
                Then remove old documents that are not needed anymore.
                Zhao et al. limit the dynamic window to 60 seconds.
                Therefore only documents from the past 30 seconds can be relevant.
                """
                self._add_documents(documents)
                self._remove_documents_before(latest_timestamp - 30)

                """
                Create checkpoints from the received documents.
                """
                self._create_checkpoint(documents)

                """
                Detect topics from the stream.
                """
                window = self._detect_topics(latest_timestamp)
                if window:
                    start, _ = window
                    timeline.add(latest_timestamp, self._documents_since(start))

                for node in timeline.nodes[::-1]:
                    if node.expired(timeline.expiry, latest_timestamp) and not node.attributes.get('printed'):
                        _documents = timeline.nodes[-1].get_all_documents()
                        _documents = [ _document for _document in _documents if len(_document.text) <= 140 ]
                        _documents = sorted(_documents, key=lambda document: len(document.text), reverse=True)
                        summary = self.summarization.summarize(_documents[:20], 140)
                        logger.info(f"{datetime.fromtimestamp(latest_timestamp).ctime()}: { str(summary) }")
                        node.attributes['printed'] = True

            await self._sleep()

        return timeline

    def _to_documents(self, tweets):
        """
        Convert the given tweets into documents.

        :param tweets: A list of tweets.
        :type tweets: list of dict

        :return: A list of documents created from the tweets in the same order as the given tweets.
                 Documents are normalized and store the original tweet in the `tweet` attribute.
        :rtype: list of :class:`~nlp.document.Document`
        """

        documents = [ ]

        """
        The text used for the document depend on what kind of tweet it is.
        If the tweet is too long to fit in the tweet, the full text is used;

        Retain the comment of a quoted status.
        However, if the tweet is a plain retweet, get the full text.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words("english"), remove_unicode_entities=True)
        for tweet in tweets:
            text = twitter.full_text(tweet)

            """
            Create the document and save the tweet in it.
            """
            tokens = tokenizer.tokenize(text)
            document = Document(text, tokens, scheme=self.scheme)
            document.attributes["tweet"] = tweet
            document.attributes['timestamp'] = twitter.extract_timestamp(tweet)
            document.normalize()
            documents.append(document)

        return documents

    def _latest_timestamp(self, documents):
        """
        Get the latest timestamp from the given documents.

        :param documents: The list of documents from where to get the latest timestamp.
        :type documents: list of :class:`~nlp.document.Document`

        :return: The latest timestamp in the given document set.
        :rtype: int

        :raises ValueError: When there are no documents to consider.
        """

        timestamps = [ document.attributes['timestamp'] for document in documents ]
        return max(timestamps)

    def _add_documents(self, documents):
        """
        Add the given documents to the list of stored documents.

        :param documents: The list of documents to store in this consumer.
        :type documents: list of :class:`~nlp.document.Document`
        """

        for document in documents:
            timestamp = document.attributes['timestamp']
            self.documents[timestamp] = self.documents.get(timestamp, [ ])
            self.documents[timestamp].append(document)

    def _documents_since(self, since):
        """
        Get all the documents since the given timestamp.
        The documents are ordered chronologically.

        :param since: The timestamp since when all documents should be returned.
                          This value is inclusive.
        :type since: float

        :return: The list of documents added since the given timestamp.
        :rtype: list of :class:`~nlp.document.Document`
        """

        documents = [ ]

        timestamps = [ timestamp for timestamp in self.documents if timestamp >= since ]
        for timestamp in sorted(timestamps):
            documents.extend(self.documents[timestamp])

        return documents

    def _remove_documents_before(self, until):
        """
        Remove all the documents published before the given timestamp.

        :param until: The timestamp until when all documents should be removed.
                      This value is exclusive.
        :type until: float
        """

        self.documents = { timestamp: documents for timestamp, documents in self.documents.items() if timestamp >= until }

    def _create_checkpoint(self, documents):
        """
        Create checkpoints from the documents in the
        After every time-window has elapsed, create a checkpoint from the documents.
        These documents are used to create a nutrition set for the nutrition store.
        This nutrition set represents a snapshot of the time-window.

        :param documents: The list of documents that form the checkpoint.
        :type documents: list of :class:`~nlp.document.Document`
        """

        if len(documents) > 0:
            """
            Concatenate all the documents in the buffer and normalize the dimensions
            The goal is to get a list of dimensions in the range 0 to 1
            """

            """
            Count the volume at each second.
            """
            volume = { }
            for document in documents:
                timestamp = document.attributes['timestamp']
                volume[timestamp] = volume.get(timestamp, 0) + 1

            """
            Retrieve the volume counts, if there are any, for the given timestamps.
            Only the volume at a given second is saved.
            """
            for timestamp, count in volume.items():
                self.store.add(timestamp, (self.store.get(timestamp) or 0) + (count or 0))

    def _detect_topics(self, timestamp):
        """
        Detect breaking topics using the Zhao et al. algorithm.

        :param timestamp: The timestamp at which point topics are detected.
                          This value is exclusive.
        :type timestamp: float

        :return: A tuple with the start and end timestamp of the time-window when there was a burst.
                 Note that this is a half-window, not the entire window.
                 If there was an increase in the second half of the last 60 seconds, the last 30 seconds are returned.
                 If there was no burst, `False` is returned.
        :rtype: tuple or bool
        """

        return self.tdt.detect(timestamp)
