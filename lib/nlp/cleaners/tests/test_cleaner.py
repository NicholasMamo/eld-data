"""
Test the functionality of the base cleaner.
"""

import asyncio
import json
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp.cleaners import Cleaner

class TestCleaner(unittest.TestCase):
    """
    Test the implementation of the base cleaner.
    """

    def test_clean_no_configuration(self):
        """
        Test that when cleaning without any configuration, the text is returned the same.
        """

        cleaner = Cleaner()

        text = 'Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings'
        self.assertEqual(text, cleaner.clean(text))

    def test_clean_strip_start(self):
        """
        Test that the text in the beginning is always stripped.
        """

        cleaner = Cleaner()

        text = ' Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings'
        self.assertEqual('Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings', cleaner.clean(text))

    def test_clean_strip_end(self):
        """
        Test that the text in the end is always stripped.
        """

        cleaner = Cleaner()

        text = 'Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings '
        self.assertEqual('Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings', cleaner.clean(text))

    def test_clean_strip(self):
        """
        Test that the text is always stripped on both ends.
        """

        cleaner = Cleaner()

        text = ' Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings '
        self.assertEqual('Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings', cleaner.clean(text))

    def test_strip_after_processing(self):
        """
        Test that the text is stripped after processing.
        """

        cleaner = Cleaner(remove_alt_codes=True)

        text = ' Our prediction based on #FIFA Rankings, Country Risk Ratings &amp;'
        self.assertEqual('Our prediction based on #FIFA Rankings, Country Risk Ratings', cleaner.clean(text))

    def test_collapse_new_lines_none(self):
        """
        Test that when there are no new lines to collapse, the text is returned the same.
        """

        cleaner = Cleaner(collapse_new_lines=True)

        text = 'Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings'
        self.assertEqual(text, cleaner.clean(text))

    def test_collapse_new_lines_single(self):
        """
        Test that when collapsing text with one new line, it is converted into a single white-space.
        """

        cleaner = Cleaner(collapse_new_lines=True, collapse_whitespaces=True)

        text = """SPECIAL OFFER
                  From now until the end of April all digital versions (apart from Issue 15) will be available to download for FREE."""
        self.assertEqual('SPECIAL OFFER From now until the end of April all digital versions (apart from Issue 15) will be available to download for FREE.', cleaner.clean(text))

    def test_collapse_new_lines_multiple(self):
        """
        Test that when collapsing text with multiple new lines, it is converted into a single white-space.
        """

        cleaner = Cleaner(collapse_new_lines=True, collapse_whitespaces=True)

        text = """SPECIAL OFFER
                  From now until the end of April all digital versions (apart from Issue 15) will be available to download for FREE.
                  So please either send the link or share the files with anyone who's self-isolating and could do with some reading material."""
        self.assertEqual('SPECIAL OFFER From now until the end of April all digital versions (apart from Issue 15) will be available to download for FREE. So please either send the link or share the files with anyone who\'s self-isolating and could do with some reading material.', cleaner.clean(text))

    def test_collapse_new_lines_empty(self):
        """
        Test that when collapsing new lines, empty lines are not retained.
        """

        cleaner = Cleaner(collapse_new_lines=True, collapse_whitespaces=True)

        text = """SPECIAL OFFER

                  From now until the end of April all digital versions (apart from Issue 15) will be available to download for FREE.
                  So please either send the link or share the files with anyone who's self-isolating and could do with some reading material."""
        self.assertEqual('SPECIAL OFFER From now until the end of April all digital versions (apart from Issue 15) will be available to download for FREE. So please either send the link or share the files with anyone who\'s self-isolating and could do with some reading material.', cleaner.clean(text))

    def test_collapse_new_lines_complete_sentences(self):
        """
        Test that when collapsing new lines with the flag to complete sentences, periods are added where necessary.
        """

        cleaner = Cleaner(complete_sentences=True, collapse_new_lines=True, collapse_whitespaces=True)

        text = """SPECIAL OFFER
                  From now until the end of April all digital versions (apart from Issue 15) will be available to download for FREE.
                  So please either send the link or share the files with anyone who's self-isolating and could do with some reading material."""
        self.assertEqual('SPECIAL OFFER. From now until the end of April all digital versions (apart from Issue 15) will be available to download for FREE. So please either send the link or share the files with anyone who\'s self-isolating and could do with some reading material.', cleaner.clean(text))

    def test_remove_alt_codes(self):
        """
        Test that when remove alt-codes, they are indeed removed.
        """

        cleaner = Cleaner(remove_alt_codes=True)

        text = 'Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings'
        self.assertEqual('Our prediction based on #FIFA Rankings,  Country Risk Ratings', cleaner.clean(text))

    def test_complete_sentences_empty(self):
        """
        Test that when completing sentences of empty text, nothing changes.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = ''
        self.assertEqual(text, cleaner.clean(text))

    def test_complete_sentences_ends_in_punctuation(self):
        """
        Test completing a sentence that already ends in punctuation.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = 'Congratulations to  @Keir_Starmer, the new Leader of the Labour Party!'
        self.assertEqual(text, cleaner.clean(text))

    def test_complete_sentences_single_quote_ends_in_punctuation(self):
        """
        Test completing a sentence that already ends in punctuation before a single quote.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '\'Congratulations to  @Keir_Starmer, the new Leader of the Labour Party!\''
        self.assertEqual(text, cleaner.clean(text))

    def test_complete_sentences_quote_ends_in_punctuation(self):
        """
        Test completing a sentence that already ends in punctuation before a quote.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '"Congratulations to  @Keir_Starmer, the new Leader of the Labour Party!"'
        self.assertEqual(text, cleaner.clean(text))

    def test_complete_sentences_french_quote_ends_in_punctuation(self):
        """
        Test completing a sentence that already ends in punctuation before a French-style quote.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '«Congratulations to  @Keir_Starmer, the new Leader of the Labour Party!»'
        self.assertEqual(text, cleaner.clean(text))

    def test_complete_sentences_single_quote_only(self):
        """
        Test that when completing a sentence that is just a single quote, nothing changes.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '\''
        self.assertEqual(text, cleaner.clean(text))

    def test_complete_sentences_quote_only(self):
        """
        Test that when completing a sentence that is just a quote, nothing changes.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '"'
        self.assertEqual(text, cleaner.clean(text))

    def test_complete_sentences_french_quote_only(self):
        """
        Test that when completing a sentence that is just a French-style quote, nothing changes.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '»'
        self.assertEqual(text, cleaner.clean(text))

    def test_complete_sentences_ends_with_single_quote(self):
        """
        Test that when a sentence ends with a single quote, the period is added before it.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '\'Sounds a lot like the last one\''
        self.assertEqual('\'Sounds a lot like the last one.\'', cleaner.clean(text))

    def test_complete_sentences_ends_with_quote(self):
        """
        Test that when a sentence ends with a quote, the period is added before it.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '"Sounds a lot like the last one"'
        self.assertEqual('"Sounds a lot like the last one."', cleaner.clean(text))

    def test_complete_sentences_ends_with_french_quote(self):
        """
        Test that when a sentence ends with a French-style quote, the period is added before it.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = '«Sounds a lot like the last one»'
        self.assertEqual('«Sounds a lot like the last one.»', cleaner.clean(text))

    def test_complete_sentences_single_character(self):
        """
        Test that when a sentence is one character, a period is added at the end.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = 'a'
        self.assertEqual('a.', cleaner.clean(text))

    def test_complete_sentences_sentence(self):
        """
        Test that when the text is an incomplete sentence, a period is added at the end.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = 'The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown.', cleaner.clean(text))

    def test_complete_sentences_colon(self):
        """
        Test that when the text ends in a colon (or any other punctuation), no period is added.
        """

        cleaner = Cleaner(complete_sentences=True)

        text = 'The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown. More details:'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown. More details:', cleaner.clean(text))

    def test_collapse_whitespaces_single(self):
        """
        Test that when the sentence has a single space, it is not removed.
        """

        cleaner = Cleaner(collapse_whitespaces=True)

        text = 'The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown', cleaner.clean(text))

    def test_collapse_whitespaces_long(self):
        """
        Test that when the sentence has long spaces, they are removed.
        """

        cleaner = Cleaner(collapse_whitespaces=True)

        text = 'The NBA is   "angling" to cancel the 2019-20 season after China\'s CBA shutdown'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown', cleaner.clean(text))

    def test_collapse_whitespaces_multiple(self):
        """
        Test that when ther are multiple whitespaces, they are collapsed into a single space.
        """

        cleaner = Cleaner(collapse_whitespaces=True)

        text = 'The  NBA  is  "angling"  to  cancel  the  2019-20  season  after  China\'s  CBA  shutdown'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown', cleaner.clean(text))

    def test_collapse_whitespaces_tab(self):
        """
        Test that tabs are also collapsed into a single whitespace.
        """

        cleaner = Cleaner(collapse_whitespaces=True)

        text = 'The NBA is    "angling" to cancel the 2019-20 season after China\'s CBA shutdown'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown', cleaner.clean(text))

    def test_collapse_whitespaces_tabs(self):
        """
        Test that multiple tabs are also collapsed into a single whitespace.
        """

        cleaner = Cleaner(collapse_whitespaces=True)

        text = 'The NBA is    "angling"    to cancel the 2019-20 season after China\'s CBA shutdown'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown', cleaner.clean(text))

    def test_collapse_whitespaces_long_tab(self):
        """
        Test that long tabs are also collapsed into a single whitespace.
        """

        cleaner = Cleaner(collapse_whitespaces=True)

        text = 'The NBA is        "angling"        to cancel the 2019-20 season after China\'s CBA shutdown'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown', cleaner.clean(text))

    def test_collapse_whitespaces_after_altcode_removal(self):
        """
        Test that whitespaces are collapsed after alt-code removal.
        Otherwise, there could be multiple whitespaces.
        """

        cleaner = Cleaner(remove_alt_codes=True, collapse_whitespaces=True)

        text = 'Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings'
        self.assertEqual('Our prediction based on #FIFA Rankings, Country Risk Ratings', cleaner.clean(text))

    def test_collapse_whitespaces_mix(self):
        """
        Test that mixes of whitespaces and tabs are also collapsed into a single whitespace.
        """

        cleaner = Cleaner(collapse_whitespaces=True)

        text = 'The NBA is     "angling"      to cancel the 2019-20 season after China\'s CBA shutdown'
        self.assertEqual('The NBA is "angling" to cancel the 2019-20 season after China\'s CBA shutdown', cleaner.clean(text))

    def test_collapse_whitespaces_before_periods(self):
        """
        Test that whitespaces are removed before periods.
        """

        cleaner = Cleaner(remove_alt_codes=True, collapse_whitespaces=True, complete_sentences=True)

        text = ' Our prediction based on #FIFA Rankings . Country Risk Ratings &amp;'
        self.assertEqual('Our prediction based on #FIFA Rankings. Country Risk Ratings.', cleaner.clean(text))

    def test_capitalize_first_empty(self):
        """
        Test that when capitalizing the first character of empty text, nothing changes.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = ''
        self.assertEqual(text, cleaner.clean(text))

    def test_capitalize_first_single_quote_only(self):
        """
        Test that when capitalizing the first character of a text that is just a single quote, nothing changes.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = '\''
        self.assertEqual(text, cleaner.clean(text))

    def test_capitalize_first_quote_only(self):
        """
        Test that when capitalizing the first character of a text that is just a quote, nothing changes.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = '"'
        self.assertEqual(text, cleaner.clean(text))

    def test_capitalize_first_french_quote_only(self):
        """
        Test that when capitalizing the first character of a text that is just a French-style quote, nothing changes.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = '»'
        self.assertEqual(text, cleaner.clean(text))

    def test_capitalize_first_starts_with_single_quote(self):
        """
        Test that when a sentence starts with a single quote, the first character after it is capitalized.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = '\'sounds a lot like the last one.\''
        self.assertEqual('\'Sounds a lot like the last one.\'', cleaner.clean(text))

    def test_capitalize_first_starts_with_quote(self):
        """
        Test that when a sentence starts with a quote, the first character after it is capitalized.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = '"sounds a lot like the last one."'
        self.assertEqual('"Sounds a lot like the last one."', cleaner.clean(text))

    def test_capitalize_first_starts_with_french_quote(self):
        """
        Test that when a sentence starts with a French-style quote, the first character after it is capitalized.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = '«sounds a lot like the last one.»'
        self.assertEqual('«Sounds a lot like the last one.»', cleaner.clean(text))

    def test_capitalize_first_single_character(self):
        """
        Test that when a sentence is one character, that character is capitalized.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = 'a'
        self.assertEqual('A', cleaner.clean(text))

    def test_capitalize_symbol(self):
        """
        Test that when the text starts with a symbol, cleaning does not crash.
        """

        cleaner = Cleaner(capitalize_first=True)

        text = '@NicholasMamo Allez l\'OL'
        self.assertEqual('@NicholasMamo Allez l\'OL', cleaner.clean(text))
