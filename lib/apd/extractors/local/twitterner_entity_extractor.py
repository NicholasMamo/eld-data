"""
The TwitterNER entity extractor uses TwitterNER to exctract named entities.
Like the :class:`~apd.extractors.local.entity_extractor.EntityExtractor`, it considers these named entities to be candidate participants.
The difference between the :class:`TwitterNEREntityExtractor` and the :class:`~apd.extractors.local.entity_extractor.EntityExtractor` is that the former uses a NER tool built specificially for Twitter.

.. warning::

    TwitterNER loads a lot of data every time it is invoked.
    Therefore this class creates a class-wide extractor when the module is loaded.
    This can be used by all instances of the :class:`TwitterNEREntityExtractor`.

.. note::

    A copy of TwitterNER is available in this directory.
    However, the data has to be downloaded.
    The data, and more instructions on how to get GloVe pre-trained on Twitter are available in `TwitterNER's GitHub repository <https://github.com/napsternxg/TwitterNER>`_.
"""

import os
import sys
import inspect

paths = [ os.path.join(os.path.dirname(__file__), 'TwitterNER', 'NoisyNLP'),
           os.path.join(os.path.dirname(__file__), '..', '..', '..') ]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from ..extractor import Extractor
from logger import logger
from run_ner import TwitterNER
from twokenize import tokenizeRawTweetText

class TwitterNEREntityExtractor(Extractor):
    """
    The :class:`TwitterNEREntityExtractor` uses TwitterNER to extract entities from documents.
    This class is built specifically for tweets.

    :cvar ner: The NER extractor used by this class.
    :vartype ner: :class:`TwitterNER.run_ner.TwitterNER`
    """

    """
    Do not create the TwitterNER object if this file is being used only for its documentation.
    """
    if ('sphinx-build' not in inspect.stack()[-1].filename and
        not any('sphinx/cmd/build.py' in stack_item.filename for stack_item in inspect.stack())):
        ner = TwitterNER()
        logger.info("TwitterNER finished loading features")

    def extract(self, corpus, *args, **kwargs):
        """
        Extract all the named entities from the corpus.
        The output is a list of lists.
        Each outer list represents a document.
        Each inner list is the candidates in that document.

        :param corpus: The corpus of documents where to extract candidate participants.
        :type corpus: list

        :return: A list of candidates separated by the document in which they were found.
        :rtype: list of list of str
        """

        candidates = [ ]

        for document in corpus:
            document_entities = [ ]
            tokens = tokenizeRawTweetText(document.text)

            entities = TwitterNEREntityExtractor.ner.get_entities(tokens)
            candidates.append([ " ".join(tokens[start:end]).lower()
                                for (start, end, type) in entities ])

        return candidates
