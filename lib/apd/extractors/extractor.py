"""
Extraction is the first step in APD.
The extractor's job is to identify candidate participants.
All extractors must start from a corpus, but they may also look for candidates elsewhere.
The functionality revolves around the :class:`~apd.extractors.extractor.Extractor`'s :func:`~apd.extractors.extractor.Extractor.extract` method.
"""

from abc import ABC, abstractmethod

class Extractor(ABC):
    """
    The base extractor defines what all ATE extractors should be able to do.
    All extractors should accept a corpus and be able to return a list of candidate participants related to it.

    The functionality revolves around one method: the :func:`~apd.extractors.extractor.Extractor.extract` method.
    This function returns the candidates, separated based on the documents in which they appear.
    """

    @abstractmethod
    def extract(self, corpus, *args, **kwargs):
        """
        Extract all the potential participants from the given corpus.
        The output is a list of lists.
        Each outer list represents a document.
        Each inner list is the candidates in that document.

        :param corpus: The corpus of documents from where to extract candidate participants.
        :type corpus: list of :class:`~nlp.document.Document`

        :return: A list of candidates separated by the document in which they were found.
        :rtype: list of list of str
        """

        pass
