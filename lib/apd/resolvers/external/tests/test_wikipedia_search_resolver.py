"""
Test the functionality of the Wikipedia search resolver.
"""

import os
import random
import re
import string
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nltk.corpus import stopwords

from apd.extractors.local.entity_extractor import EntityExtractor
from apd.scorers.local.tf_scorer import TFScorer
from apd.filters.local.threshold_filter import ThresholdFilter
from apd.resolvers.external.wikipedia_search_resolver import WikipediaSearchResolver

from nlp.document import Document
from nlp.tokenizer import Tokenizer
from nlp.weighting.tf import TF

class TestWikipediaSearchResolver(unittest.TestCase):
    """
    Test the implementation and results of the Wikipedia search resolver.
    """

    def test_year_check(self):
        """
        Test that when checking for a year, the function returns a boolean.
        """

        article = 'Youssouf Koné (footballer, born 1995)'
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertTrue(resolver._has_year(article))

    def test_year_check_range(self):
        """
        Test that when checking for a year in a range, the function returns `True`.
        """

        article = '2019–20 Premier League'
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertTrue(resolver._has_year(article))

        article = '2019-20 Premier League'
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertTrue(resolver._has_year(article))

    def test_year_check_short_number(self):
        """
        Test that when checking for a year with a short number, the function does not detect a year.
        """

        article = 'Area 51'
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertFalse(resolver._has_year(article))

    def test_year_check_long_number(self):
        """
        Test that when checking for a year with a long number, the function does not detect a year.
        """

        article = '1234567890'
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertFalse(resolver._has_year(article))

    def test_remove_brackets(self):
        """
        Test that when removing brackets, they are completely removed.
        """

        article = 'Youssouf Koné (footballer, born 1995)'
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertEqual('Youssouf Koné', resolver._remove_brackets(article).strip())

    def test_remove_unclosed_brackets(self):
        """
        Test that when removing brackets that are not closed, they are not removed.
        """

        article = 'Youssouf Koné (footballer, born 1995'
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertEqual('Youssouf Koné (footballer, born 1995', resolver._remove_brackets(article).strip())

    def test_get_first_sentence(self):
        """
        Test that when getting the first sentence from text, only the first sentence is returned.
        """

        text = "Memphis Depay (Dutch pronunciation: [ˈmɛmfɪs dəˈpɑi]; born 13 February 1994), \
                commonly known simply as Memphis,[2] is a Dutch professional \
                footballer and music artist who plays as a forward and captains \
                French club Lyon and plays for the Netherlands national team. \
                He is known for his pace, ability to cut inside, dribbling, \
                distance shooting and ability to play the ball off the ground."

        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertEqual("Memphis Depay (Dutch pronunciation: [ˈmɛmfɪs dəˈpɑi]; born 13 February 1994), commonly known simply as Memphis,[2] is a Dutch professional footballer and music artist who plays as a forward and captains French club Lyon and plays for the Netherlands national team.",
                         re.sub('([ \t]+)', ' ', resolver._get_first_sentence(text)).strip())

    def test_get_first_sentence_full(self):
        """
        Test that when getting the first sentence from a text that has only one sentence, the whole text is returned.
        """

        text = "Youssouf Koné (born 5 July 1995) is a Malian professional footballer who plays for French side Olympique Lyonnais and the Mali national team as a left-back."
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertEqual(text, resolver._get_first_sentence(text))

    def test_get_first_sentence_full_without_period(self):
        """
        Test that when getting the first sentence from a text that has only one sentence, but without punctuation, the whole text is returned.
        """

        text = "Youssouf Koné (born 5 July 1995) is a Malian professional footballer who plays for French side Olympique Lyonnais and the Mali national team as a left-back"
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertEqual(text, resolver._get_first_sentence(text))

    def test_get_first_sentence_empty(self):
        """
        Test that when getting the first sentence from an empty string, an empty string is returned.
        """

        text = ""
        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertEqual(text, resolver._get_first_sentence(text))

    def test_score_upper_bound(self):
        """
        Test that the score has an upper bound of 1.
        """

        tokenizer = Tokenizer(min_length=2, stem=True)
        candidate = "Cristiano Ronaldo"
        candidate_document = Document(candidate, tokenizer.tokenize(candidate))
        title_document = Document(candidate, tokenizer.tokenize(candidate))

        text = "Cristiano Ronaldo is a Portuguese professional footballer who plays as a forward for Serie A club Juventus and captains the Portugal national team."
        domain = Document(text, tokenizer.tokenize(text))
        sentence = Document(text, tokenizer.tokenize(text))

        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertEqual(1, round(resolver._compute_score(candidate_document, title_document, domain, sentence), 5))

    def test_score_lower_bound(self):
        """
        Test that the score has a lower bound of 0.
        """

        tokenizer = Tokenizer(min_length=2, stem=True)
        candidate = "Cristiano Ronaldo"
        candidate_document = Document(candidate, tokenizer.tokenize(candidate))
        text = "Cristiano Ronaldo is a Portuguese professional footballer who plays as a forward for Serie A club Juventus and captains the Portugal national team."
        domain = Document(text, tokenizer.tokenize(text))

        title_document = Document(candidate, [ ])
        sentence = Document(text, [ ])

        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        self.assertEqual(0, round(resolver._compute_score(candidate_document, title_document, domain, sentence), 5))

    def test_score_relevance(self):
        """
        Test that when two documents are provided, one more relevant than the other, the score reflects it.
        """

        tokenizer = Tokenizer(min_length=2, stem=True)
        candidate = "Ronaldo"
        candidate_document = Document(candidate, tokenizer.tokenize(candidate))
        text = "Ronaldo, speaking after Juventus' victory, says Serie A is still wide open"
        domain = Document(text, tokenizer.tokenize(text))

        title_1 = "Cristiano Ronaldo"
        text_1 = "Cristiano Ronaldo is a Portuguese professional footballer who plays as a forward for Serie A club Juventus."
        title_document_1 = Document(title_1, tokenizer.tokenize(title_1))
        sentence_document_1 = Document(text_1, tokenizer.tokenize(text_1))

        title_2 = "Ronaldo"
        text_2 = "Ronaldo is a Brazilian former professional footballer who played as a striker."
        title_document_2 = Document(title_2, tokenizer.tokenize(title_2))
        sentence_document_2 = Document(text_2, tokenizer.tokenize(text_2))

        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        score_1 = resolver._compute_score(candidate_document, title_document_1, domain, sentence_document_1)
        score_2 = resolver._compute_score(candidate_document, title_document_2, domain, sentence_document_2)
        self.assertGreater(score_1, score_2)

    def test_wikipedia_name_resolver(self):
        """
        Test the Wikipedia search resolver.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=2, stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "Ronaldo, speaking after Juventus' victory, says league is still wide open, but his team is favorite",
            "Ronaldo's goal voted goal of the year by football fans appreciative of the striker",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)

        resolver = WikipediaSearchResolver(TF(), tokenizer, 0, corpus)
        resolved, unresolved = resolver.resolve(scores)
        self.assertTrue('Cristiano Ronaldo' in resolved)
        self.assertTrue('Juventus F.C.' in resolved)

    def test_all_resolved_or_unresolved(self):
        """
        Test that the resolver either resolves or does not resolve named entities.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=2, stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "Ronaldo, speaking after Juventus' victory, says league is still wide open, but his team is favorite",
            "Ronaldo's goal voted goal of the year by football fans appreciative of the striker",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)

        resolver = WikipediaSearchResolver(TF(), tokenizer, 0, corpus)
        resolved, unresolved = resolver.resolve(scores)
        self.assertEqual(len(scores), len(resolved + unresolved))

    def test_random_string_unresolved(self):
        """
        Test that a random string is unresolved.
        """

        random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(32))

        tokenizer = Tokenizer(min_length=1, stem=False)
        resolver = WikipediaSearchResolver(TF(), tokenizer, 0, [ ])
        resolved, unresolved = resolver.resolve({ random_string: 1 })
        self.assertTrue(random_string in unresolved)

    def test_threshold(self):
        """
        Test that when the threshold is not zero, it excludes some candidates.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=2, stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "Ronaldo, speaking after Juventus' victory, says league is still wide open, but his team is favorite",
            "Ronaldo's goal voted goal of the year by football fans appreciative of the striker",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)

        resolver = WikipediaSearchResolver(TF(), tokenizer, 0.1, corpus)
        resolved, unresolved = resolver.resolve(scores)
        self.assertTrue('Cristiano Ronaldo' in resolved)
        self.assertFalse('Juventus F.C.' in resolved)
        self.assertTrue('juventus' in unresolved)

    def test_high_threshold(self):
        """
        Test that when the threshold is high, it excludes all candidates.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=2, stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "Ronaldo, speaking after Juventus' victory, says league is still wide open, but his team is favorite",
            "Ronaldo's goal voted goal of the year by football fans appreciative of the striker",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)

        resolver = WikipediaSearchResolver(TF(), tokenizer, 1, corpus)
        resolved, unresolved = resolver.resolve(scores)
        self.assertFalse(len(resolved))
        self.assertEqual(set(scores.keys()), set(unresolved))

    def test_resolve_empty(self):
        """
        Test that when resolving an empty set of candidates, the resolver returns empty lists.
        """

        resolver = WikipediaSearchResolver(TF(), Tokenizer(), 0, [ ])
        resolved, unresolved = resolver.resolve({ })
        self.assertFalse(len(resolved))
        self.assertFalse(len(unresolved))

    def test_resolve_no_duplicates(self):
        """
        Test that resolution does not include duplicates.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=2, stem=True, stopwords=list(stopwords.words("english")))
        posts = [
            "Tottenham fighting for the English Premiear League",
            "Tottenham Hotspur keep Champions Leagues hopes alive",
            "Premier League: Tottenham on the brink of Champions League football",
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = { 'tottenham': 1, 'tottenham hotspur': 1, 'hotspur': 1 }
        resolver = WikipediaSearchResolver(TF(), tokenizer, 0, corpus)
        resolved, unresolved = resolver.resolve(candidates)
        self.assertEqual(list(set(resolved)), resolved)
        self.assertEqual([ 'Tottenham Hotspur F.C.' ], resolved)

    def test_sorting(self):
        """
        Test that the resolver sorts the named entities in descending order of score.
        """

        """
        Create the test data
        """
        tokenizer = Tokenizer(min_length=3, stem=True, case_fold=True)
        posts = [
            "In the most heated football match of the season, Liverpool falter against Manchester City",
            "Liverpool unable to avoid defeat to Watford, Manchester City close in on football title"
        ]
        corpus = [ Document(post, tokenizer.tokenize(post)) for post in posts ]

        candidates = EntityExtractor().extract(corpus, binary=True)
        scores = TFScorer().score(candidates)
        scores = ThresholdFilter(0).filter(scores)
        resolved, unresolved = WikipediaSearchResolver(TF(), tokenizer, 0, corpus).resolve(scores)
        self.assertEqual('Liverpool F.C.', resolved[0])
        self.assertEqual('Manchester City F.C.', resolved[1])
        self.assertEqual('Watford F.C.', resolved[2])
