"""
Document cleaners are classes that improve the presentation of text.
Cleaning can be used in summarization, for example, when the output text may be noisy or informal (such as when working with tweets).
However, it can also be used as a pre-processing step to remove undesired fragments.

The :class:`~nlp.cleaners.Cleaner` is a base class for all cleaners.
However, it also performs simple processing on text to remove common mistakes, such as missing punctuation or multiple contiguous white-spaces.

All cleaners are configured in their constructor to ensure that cleaning is uniform across all documents.
The main functionality resides in the :func:`~nlp.cleaners.Cleaner.clean` function, which receives text and returns a cleaner version of it.
Other cleaners that inherit the :class:`~nlp.cleaners.Cleaner` class would likely need to change this function.
"""

import re
import string

class Cleaner(object):
    """
    The base cleaner is meant to perform only basing pre-processing and cleaning.
    These functions are generally applicable to any type of text.

    You need to specify all cleaner settings in the constructor.
    This design is purposeful so that the cleaner processes all text in the same way.
    By default, the cleaner only removes whitespaces or tabs at the start and end of the text.

    After creating a cleaner, use the :func:`~nlp.cleaners.Cleaner.clean` function to clean the text.

    :ivar remove_alt_codes: A boolean indicating whether alt-codes should be removed.
    :vartype remove_alt_codes: bool
    :ivar complete_sentences: A boolean indicating whether the sentences should be completed.
                              The cleaner cannot identify any incomplete sentences in the middle of the text, so it only completes the last sentence.
    :vartype complete_sentences: bool
    :ivar collapse_new_lines: A boolean indicating whether new lines should be collapsed into whitespaces.
    :vartype collapse_new_lines: bool
    :ivar collapse_whitespaces: A boolean indicating whether consecutive whitespaces and tabs should be collapsed into a single whitespace.
                                This also removes whitespaces just before periods.
    :vartype collapse_whitespaces: bool
    :ivar capitalize_first: A boolean indicating whether to capitalize the first letter.
    :vartype capitalize_first: bool
    """

    def __init__(self, remove_alt_codes=False,
                       complete_sentences=False,
                       collapse_new_lines=False,
                       collapse_whitespaces=False,
                       capitalize_first=False):
        """
        Create the cleaner with the basic configuration.

        :param remove_alt_codes: A boolean indicating whether alt-codes should be removed.
        :type remove_alt_codes: bool
        :param complete_sentences: A boolean indicating whether the sentences should be completed.
                                   The cleaner cannot identify any incomplete sentences in the middle of the text, so it only completes the last sentence.
        :type complete_sentences: bool
        :param collapse_new_lines: A boolean indicating whether new lines should be collapsed into whitespaces.
        :type collapse_new_lines: bool
        :param collapse_whitespaces: A boolean indicating whether consecutive whitespaces and tabs should be collapsed into a single whitespace.
                                     This also removes whitespaces just before periods.
        :type collapse_whitespaces: bool
        :param capitalize_first: A boolean indicating whether to capitalize the first letter.
        :type capitalize_first: bool
        """

        self.remove_alt_codes = remove_alt_codes
        self.complete_sentences = complete_sentences
        self.collapse_new_lines = collapse_new_lines
        self.collapse_whitespaces = collapse_whitespaces
        self.capitalize_first = capitalize_first

    def clean(self, text):
        """
        Clean the given text based on the parameters specified when creating the cleaner.
        The basic cleaner always strips empty whitespaces before and after all processing.

        :param text: The text to clean.
        :type text: str

        :return: The cleaned text.
        :rtype: str
        """

        text = text.strip()
        text = self._collapse_new_lines(text) if self.collapse_new_lines else text
        text = self._remove_alt_codes(text) if self.remove_alt_codes else text
        text = self._complete_sentences(text) if self.complete_sentences else text
        text = self._collapse_whitespaces(text) if self.collapse_whitespaces else text
        text = self._capitalize_first(text) if self.capitalize_first else text
        text = text.strip()

        return text

    def _collapse_new_lines(self, text):
        """
        Collapse new lines into white spaces.
        """

        lines = text.split('\n')
        lines = [ line for line in lines if len(line) ]
        lines = [ self._complete_sentences(line) for line in lines ] if self.complete_sentences else lines
        return ' '.join(lines)

    def _remove_alt_codes(self, text):
        """
        Remove alt-codes from the given text.

        :param text: The text to clean.
        :type text: str

        :return: The text without alt-codes.
        :rtype: str
        """

        alt_code_pattern = re.compile('&.+?;')
        return alt_code_pattern.sub('', text)

    def _complete_sentences(self, text):
        """
        Add a period if the sentence does not end with punctuation.
        There is one exception to this rule: if the sentence ends with a quote.
        In this case, the period is added before the quote if there is no punctuation there..

        :param text: The text to clean.
        :type text: str

        :return: The text without alt-codes.
        :rtype: str
        """

        quotes = ['\'', '"', '»']

        """
        If the text is empty, return immediately.
        """
        if not text:
            return text

        """
        If the text already ends in punctuation, return immediately.
        """
        if ((text[-1] in string.punctuation and text[-1] not in quotes) or
            (len(text) > 1 and text[-1] in quotes and text[-2] in string.punctuation)):
            return text

        """
        If the text is just a quote, return immediately.
        """
        if text in quotes:
            return text

        """
        If the text ends with a quote, but is not a complete sentence before the quote, add a period.
        """
        if len(text) > 1 and text[-1] in quotes and text[-2] not in string.punctuation:
            return f"{text[:-1]}.{text[-1]}"

        """
        Otherwise, add a period at the end.
        """
        return f"{text}."

    def _collapse_whitespaces(self, text):
        """
        Remove consecutive whitespaces and tabs, and replace them with a single space.
        This also removes whitespaces just before periods.

        :param text: The text to clean.
        :type text: str

        :return: The text without any consectuive spaces or tabs.
        :rtype: str
        """

        whitespace_pattern = re.compile('(\\s{2,}|\\t)+')
        text = whitespace_pattern.sub(' ', text)

        whitespace_period_pattern =  re.compile('\\s\\.')
        text = whitespace_period_pattern.sub('.', text)

        return text

    def _capitalize_first(self, text):
        """
        Capitalize the first letter of the given text.

        :param text: The text to clean.
        :type text: str

        :return: The text with the first letter capitalized.
        :rtype: str
        """

        quotes = ['\'', '"', '«']

        """
        If the text is empty, return immediately.
        """
        if not text:
            return text

        """
        If the text is just a quote, return immediately.
        """
        if text in quotes:
            return text

        """
        If the text starts with a quote capitalize the next letter.
        """
        if len(text) > 1 and text[0] in quotes:
            text = f"{ text[0] }{ text[1].upper() }{ text[2:] }"
            return text

        """
        Otherwise, capitalize the first letter.
        """
        text = f"{ text[0].upper() }{ text[1:] }"
        return text

from .tweet_cleaner import TweetCleaner
