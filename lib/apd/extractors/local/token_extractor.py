"""
The token extractor considers all the tokens in the corpus to be candidate participants.
Therefore it does not perform any filtering whatsoever on the corpus.
"""

from ..extractor import Extractor

class TokenExtractor(Extractor):
    """
    The token extractor returns all tokens as potential candidates.
    To extract tokens, the token extractor needs a tokenizer, but it is optional.
    If no tokenizer is given, the token extractor uses documents' dimensions as tokens.

    .. note::

        Document dimensions are unique: if a token appears twice in a document, it will still have one dimension.
        Therefore a token extractor without a tokenizer returns only unique terms from each document.

    :ivar ~.tokenizer: The tokenizer used to extract the tokens.
                       If it is given, the tokens are extracted anew.
                       Otherwise, the document dimensions are used instead.
    :vartype ~.tokenizer: :class:`~nlp.tokenizer.Tokenizer` or None
    """

    def __init__(self, tokenizer=None):
        """
        Create the extractor with a tokenizer.

        :param tokenizer: The tokenizer used to extract the tokens.
                          If it is given, the tokens are extracted anew.
                          Otherwise, the document dimensions are used instead.
        :type tokenizer: :class:`~nlp.tokenizer.Tokenizer` or None
        """

        self.tokenizer = tokenizer

    def extract(self, corpus, *args, **kwargs):
        """
        Extract all the potential participants from the corpus.
        The output is a list of lists.
        Each outer list represents a document.
        Each inner list is the candidates in that document.

        :param corpus: The corpus of documents where to extract candidate participants.
        :type corpus: list

        :return: A list of candidates separated by the document in which they were found.
        :rtype: list of list of str
        """

        if self.tokenizer:
            return [ self.tokenizer.tokenize(document.text) for document in corpus ]
        else:
            return [ [ token for token in document.dimensions ] for document in corpus ]
