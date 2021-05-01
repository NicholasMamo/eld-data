"""
Summarization algorithms receive documents and create a summary having the given length.
The specifics of how they create summaries change from one algorithm to the other, but the goal does not.

The :class:`~summarization.algorithms.SummarizationAlgorithm` represents this goal.
It is a base class that dictates the minimum functionality of any summarization algorithm: what input they receive and what output they return.
In all cases, the main functionality goes through the :func:`~summarization.algorithms.lgorithm.summarize` function.
"""

from abc import ABC, abstractmethod

class SummarizationAlgorithm(ABC):
    """
    Summarization algorithms vary greatly and thus there is no general state.
    However, they must all implement the :func:`~summarization.algorithms.SummarizationAlgorithm.summarize` method.
    This method is the central part of any :class:`~summarization.algorithms.SummarizationAlgorithm`.
    It receives a list of :class:`~nlp.document.Document` instances and a target length and creates a :class:`~summarization.summary.Summary`.
    """

    @abstractmethod
    def summarize(self, documents, length, *args, **kwargs):
        """
        Summarize the given list of :class:`~nlp.document.Document` instances.

        Summarization algorithms may accept more parameters, but they must accept, at least:

        1. A list of :class:`~nlp.document.Document` instances, and
        2. The maximum length of the produced :class:`~summarization.summary.Summary`.

        They must also always return a :class:`~summarization.summary.Summary` instance.

        .. note::

            The algorithm assumes that all :class:`~nlp.document.Document` instances are already unique sentences.
            Therefore it does not split them into sentences or manipulate them in any way.
            The returned :class:`~summarization.summary.Summary` is a selection of the given :class:`~nlp.document.Document` instances.

        :param documents: The list of documents to summarize.
        :type documents: list of :class:`~nlp.document.Document`
        :param length: The maximum length of the summary in characters.
                       Summarization algorithms that inherit the :class:`~summarization.algorithms.SummarizationAlgorithm` must ensure that the length of the :class:`~summarization.summary.Summary` does not exceed this length.
                       Summaries may be shorter, however.
                       When considering the length of the :class:`~summarization.summary.Summary`, the function does not apply any pre-processing.
                       All it considers is the text of the original :class:`~nlp.document.Document` instances making up the :class:`~summarization.summary.Summary`.
        :type length: float

        :return: The summary of the documents.
        :rtype: :class:`~summarization.summary.Summary`
        """

        pass


from .dgs import DGS
from .mmr import MMR
