"""
Test the functionality of the NER participant detector.
"""

import os
import sys
import unittest

paths = [ os.path.join(os.path.dirname(__file__), '..'),
           os.path.join(os.path.dirname(__file__), '..', '..') ]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from ner_participant_detector import NERParticipantDetector

from nlp.document import Document
from nlp.tokenizer import Tokenizer

class TestNERParticipantDetector(unittest.TestCase):
    """
    Test the implementation and results of the NER participant detector.
    """

    def test_extract_named_entities(self):
        """
        Test extracting named entities.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Liverpool falter against Tottenham Hotspur",
            "Mourinho under pressure as Tottenham follow with a loss",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]
        detector = NERParticipantDetector()

        participants, _, _ = detector.detect(corpus)
        self.assertEqual(set([ 'liverpool', 'tottenham hotspur', 'tottenham', 'mourinho' ]), set(participants))

    def test_named_entity_sorting(self):
        """
        Test that the named entities are sorted in descending order of their frequency.
        """

        """
        Create the test data.
        """
        tokenizer = Tokenizer(stem=False)
        posts = [
            "Tottenham in yet another loss, this time against Chelsea",
            "Another loss for Tottenham as Mourinho sees red",
            "Mourinho's Tottenham lose again",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]
        detector = NERParticipantDetector()

        participants, unresolved, _ = detector.detect(corpus)
        self.assertEqual([ 'tottenham', 'mourinho', 'chelsea' ], participants)
