"""
Test the functionality of the ELD participant detector.
"""

from nltk.corpus import stopwords
import json
import os
import sys
import unittest

paths = [ os.path.join(os.path.dirname(__file__), '..'),
           os.path.join(os.path.dirname(__file__), '..', '..'),
          os.path.join(os.path.dirname(__file__), '..', '..', '..') ]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from eld_participant_detector import ELDParticipantDetector
from extractors.local import EntityExtractor
from scorers.local import TFScorer, LogTFScorer
from filters.local import RankFilter, ThresholdFilter
from resolvers import Resolver
from resolvers.external import WikipediaSearchResolver
from extrapolators import Extrapolator
from extrapolators.external import WikipediaExtrapolator
from postprocessors import Postprocessor
from postprocessors.external import WikipediaPostprocessor

from nlp.document import Document
from nlp.tokenizer import Tokenizer
from objects.exportable import Exportable
from vsm import vector_math

class TestELDParticipantDetector(unittest.TestCase):
    """
    Test the implementation and results of the ELD participant detector.
    """

    def test_custom_extractor(self):
        """
        Test that when a custom extractor is given, it is used.
        """

        apd = ELDParticipantDetector(extractor=EntityExtractor())
        self.assertEqual(EntityExtractor, type(apd.extractor))

    def test_custom_scorer(self):
        """
        Test that when a custom scorer is given, it is used.
        """

        apd = ELDParticipantDetector(scorer=LogTFScorer())
        self.assertEqual(LogTFScorer, type(apd.scorer))

    def test_custom_filter(self):
        """
        Test that when a custom filter is given, it is used.
        """

        apd = ELDParticipantDetector(filter=ThresholdFilter(0.5))
        self.assertEqual(ThresholdFilter, type(apd.filter))

    def test_custom_resolver(self):
        """
        Test that when a custom resolver is given, it is used.
        """

        apd = ELDParticipantDetector(resolver=Resolver())
        self.assertEqual(Resolver, type(apd.resolver))

    def test_custom_extrapolator(self):
        """
        Test that when a custom extrapolator is given, it is used.
        """

        apd = ELDParticipantDetector(extrapolator=Extrapolator())
        self.assertEqual(Extrapolator, type(apd.extrapolator))

    def test_custom_postprocessor(self):
        """
        Test that when a custom postprocessor is given, it is used.
        """

        apd = ELDParticipantDetector(postprocessor=Postprocessor())
        self.assertEqual(Postprocessor, type(apd.postprocessor))

    def test_default_configuration(self):
        """
        Test the default configuration of the ELD participant detector.
        """

        apd = ELDParticipantDetector()
        from extractors.local.twitterner_entity_extractor import TwitterNEREntityExtractor
        self.assertEqual(TwitterNEREntityExtractor, type(apd.extractor))
        self.assertEqual(TFScorer, type(apd.scorer))
        self.assertEqual(RankFilter, type(apd.filter))
        self.assertEqual(WikipediaSearchResolver, type(apd.resolver))
        self.assertEqual(WikipediaExtrapolator, type(apd.extrapolator))
        self.assertEqual(WikipediaPostprocessor, type(apd.postprocessor))

    def test_default_configuration_with_overload(self):
        """
        Test the default configuration of the ELD participant detector when overloading certain components.
        """

        apd = ELDParticipantDetector(scorer=LogTFScorer(), filter=ThresholdFilter(1))
        from extractors.local.twitterner_entity_extractor import TwitterNEREntityExtractor
        self.assertEqual(TwitterNEREntityExtractor, type(apd.extractor))
        self.assertEqual(LogTFScorer, type(apd.scorer))
        self.assertEqual(ThresholdFilter, type(apd.filter))
        self.assertEqual(WikipediaSearchResolver, type(apd.resolver))
        self.assertEqual(WikipediaExtrapolator, type(apd.extrapolator))
        self.assertEqual(WikipediaPostprocessor, type(apd.postprocessor))

    def test_tokenize_corpus_normalized(self):
        """
        Test that the documents returned by the corpus tokenization are normalized.
        """

        """
        Load the corpus.
        """
        filename = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'corpora', 'understanding', 'CRYCHE.json')
        corpus = [ ]
        with open(filename) as f:
            for i, line in enumerate(f):
                tweet = json.loads(line)
                original = tweet
                while "retweeted_status" in tweet:
                    tweet = tweet["retweeted_status"]

                if "extended_tweet" in tweet:
                    text = tweet["extended_tweet"].get("full_text", tweet.get("text", ""))
                else:
                    text = tweet.get("text", "")

                document = Document(text)
                corpus.append(document)

        """
        Load the TF-IDF scheme.
        """
        idf_filename = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'corpora', 'idf.json')
        with open(idf_filename) as f:
            scheme = Exportable.decode(json.loads(f.readline()))['tfidf']

        """
        Tokenize the corpus.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words('english'),
                              normalize_words=True, character_normalization_count=3,
                              remove_unicode_entities=True)
        apd = ELDParticipantDetector(extractor=EntityExtractor())
        corpus = apd._tokenize_corpus(corpus, scheme, tokenizer)
        self.assertTrue(all( round(vector_math.magnitude(document), 10) in [ 0, 1 ] for document in corpus ))

    def test_tokenize_corpus_same_text(self):
        """
        Test that the documents returned by the corpus tokenization retain the same text.
        """

        """
        Load the corpus.
        """
        filename = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'corpora', 'understanding', 'CRYCHE.json')
        corpus = [ ]
        with open(filename) as f:
            for i, line in enumerate(f):
                tweet = json.loads(line)
                original = tweet
                while "retweeted_status" in tweet:
                    tweet = tweet["retweeted_status"]

                if "extended_tweet" in tweet:
                    text = tweet["extended_tweet"].get("full_text", tweet.get("text", ""))
                else:
                    text = tweet.get("text", "")

                document = Document(text)
                corpus.append(document)

        """
        Load the TF-IDF scheme.
        """
        idf_filename = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'corpora', 'idf.json')
        with open(idf_filename) as f:
            scheme = Exportable.decode(json.loads(f.readline()))['tfidf']

        """
        Tokenize the corpus.
        """
        tokenizer = Tokenizer(stopwords=stopwords.words('english'),
                              normalize_words=True, character_normalization_count=3,
                              remove_unicode_entities=True)
        apd = ELDParticipantDetector(extractor=EntityExtractor())
        tokenized = apd._tokenize_corpus(corpus, scheme, tokenizer)
        for original, tokenized in zip(corpus, tokenized):
            self.assertFalse(original == tokenized)
            self.assertEqual(original.text, tokenized.text)
