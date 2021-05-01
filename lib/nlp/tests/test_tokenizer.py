"""
Run unit tests on the :class:`~nlp.tokenizer.Tokenizer` class.
"""

from nltk.corpus import stopwords

import os
import sys
import time
import unittest

import warnings

path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)

from tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    """
    Run unit tests on the :class:`~nlp.tokenizer.Tokenizer` class.
    """

    def ignore_warnings(test):
        """
        A decorator function used to ignore NLTK warnings.
        From: http://www.neuraldump.net/2017/06/how-to-suppress-python-unittest-warnings/
        More about decorator functions: https://wiki.python.org/moin/PythonDecorators
        """
        def perform_test(self, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test(self, *args, **kwargs)
        return perform_test

    def test_remove_mentions(self):
        """
        Test the mention removal functionality.
        """

        s = "@WRenard with the header"
        t = Tokenizer(remove_mentions=True)
        self.assertEqual([ "with", "the", "header" ], t.tokenize(s))

    def test_remove_multiple_mentions(self):
        """
        Test that the mention removal functionality removes multiple mentions.
        """

        s = "@WRenard with the header from @AdaStolsmo's cross"
        t = Tokenizer(remove_mentions=True)
        self.assertEqual([ "with", "the", "header", "from", "cross" ], t.tokenize(s))

    def test_keep_mentions(self):
        """
        Test that mentions can be retained by the mention removal functionality.
        """

        s = "@WRenard with the header"
        t = Tokenizer(remove_mentions=False)
        self.assertEqual([ "wrenard", "with", "the", "header" ], t.tokenize(s))

    def test_keep_multiple_mentions(self):
        """
        Test that multiple mentions can be retained by the mention removal functionality.
        """

        s = "@WRenard with the header from @AdaStolsmo's cross"
        t = Tokenizer(remove_mentions=False)
        self.assertEqual([ "wrenard", "with", "the", "header", "from", "adastolsmo", "cross" ], t.tokenize(s))

    def test_remove_hashtag(self):
        """
        Test that the hashtag removal functionality removes a single hashtag.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #FPL"
        t = Tokenizer(remove_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet" ], t.tokenize(s))

    def test_remove_hashtags(self):
        """
        Test that the hashtag removal functionality removes all hashtags.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #FPL #LEICHE"
        t = Tokenizer(remove_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet" ], t.tokenize(s))

    def test_remove_hashtags_mixed_case(self):
        """
        Test that the hashtag removal functionality removes all hashtags, regardless of the case.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #FPL #LeiChe"
        t = Tokenizer(remove_hashtags=True, split_hashtags=False, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet" ], t.tokenize(s))

    def test_retain_hashtags(self):
        """
        Test that the hashtag removal functionality can retain all hashtags.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #FPL #LEICHE"
        t = Tokenizer(remove_hashtags=False, split_hashtags=False, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'fpl', 'leiche' ], t.tokenize(s))

    def test_retain_hashtags_mixed_case(self):
        """
        Test that the hashtag removal functionality can retain all hashtags, regardless of the case.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #FPL #LeiChe"
        t = Tokenizer(remove_hashtags=False, split_hashtags=False, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'fpl', 'leiche' ], t.tokenize(s))

    def test_retain_hashtags_with_splitting(self):
        """
        Test that when hashtags are removed, split hashtags are retained.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #FPL #LeiChe"
        t = Tokenizer(remove_hashtags=True, split_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'lei', 'che' ], t.tokenize(s))

    def test_split_hashtag(self):
        """
        Test the hashtag splitting functionality.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #LeiChe"
        t = Tokenizer(split_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'lei', 'che' ], t.tokenize(s))

    def test_split_hashtag_all_upper(self):
        """
        Test that trying to split a hashtag that is made up of only uppercase letters does not split it.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #FPL"
        t = Tokenizer(split_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'fpl' ], t.tokenize(s))

    def test_split_hashtag_all_lower(self):
        """
        Test that trying to split a hashtag that is made up of only lowercase letters does not split it.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #fpl"
        t = Tokenizer(split_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'fpl' ], t.tokenize(s))

    def test_split_hashtag_multiple_components(self):
        """
        Test that hashtags with multiple components are split properly.
        """

        s = "Hello! I'm Harry Styles, I'm sixteen and I work in a bakery #HappyBirthdayHarry"
        t = Tokenizer(split_hashtags=True, stem=False)
        self.assertEqual([ "hello", "harry", "styles", "sixteen", "and", "work", "bakery", 'happy', 'birthday', 'harry' ], t.tokenize(s))

    def test_split_hashtag_repeated(self):
        """
        Test that when a hashtag is repeated, splitting is applied to both.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #LeiChe #LeiChe"
        t = Tokenizer(split_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'lei', 'che', 'lei', 'che' ], t.tokenize(s))

    def test_split_hashtag_with_numbers(self):
        """
        Test that hashtags are treated as words when splitting hashtags.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #EPL2020"
        t = Tokenizer(split_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'epl', '2020' ], t.tokenize(s))

        s = "The Vardy party has gone very quiet 💤 😢 #2020EPL"
        t = Tokenizer(split_hashtags=True, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", '2020', 'epl' ], t.tokenize(s))

    def test_do_not_split_hashtags(self):
        """
        Test that hashtags aren't split if the flag is not provided.
        """

        s = "The Vardy party has gone very quiet 💤 😢 #EPL2020"
        t = Tokenizer(split_hashtags=False, stem=False)
        self.assertEqual([ "the", "vardy", "party", "has", "gone", "very", "quiet", 'epl2020' ], t.tokenize(s))

    def test_remove_numbers(self):
        """
        Test that numbers are removed when the flag is provided.
        """

        s = "There are 7 changes to last week's XI"
        t = Tokenizer(remove_numbers=True, stem=False)
        self.assertEqual([ "there", "are", "changes", "last", "week" ], t.tokenize(s))

    def test_remove_mixed_numbers(self):
        """
        Test that numbers that are part of a string are not removed.
        """

        s = "Nate Diaz wants Conor McGregor, Joe Rogan to apologize to Stephen A. Smith before UFC246"
        t = Tokenizer(remove_numbers=True, stem=False)
        self.assertEqual([ "nate", "diaz", "wants", "conor", "mcgregor", "joe", "rogan", "apologize", "stephen", "smith", "before", "ufc246" ], t.tokenize(s))

    def test_retain_years(self):
        """
        Test that removing numbers does not remove years.
        """

        s = "Real paid 5M€ for him back in 2016."
        t = Tokenizer(remove_numbers=True, stem=False)
        self.assertEqual([ "real", "paid", '5m€', "for", "him", "back", "2016" ], t.tokenize(s))

        s = "Real paid 500 for him back in 2016."
        t = Tokenizer(remove_numbers=True, stem=False)
        self.assertEqual([ "real", "paid", "for", "him", "back", "2016" ], t.tokenize(s))

        s = "Real paid 50000 for him back in 2016."
        t = Tokenizer(remove_numbers=True, stem=False)
        self.assertEqual([ "real", "paid", "for", "him", "back", "2016" ], t.tokenize(s))

    def test_retain_numbers(self):
        """
        Test that the tokenizer correctly retains numbers.
        """

        s = "Real paid 5000000 for him back in 2016."
        t = Tokenizer(remove_numbers=False, stem=False)
        self.assertEqual([ "real", "paid", "5000000", "for", "him", "back", "2016" ], t.tokenize(s))

    def test_remove_url(self):
        """
        Test the URL removal functionality.
        """

        s = "Thank you @BillGates. It's amazing, almost as incredible as the fact that you use Gmail. https://t.co/drawyFHHQM"
        t = Tokenizer(remove_urls=True, stem=False)
        self.assertEqual([ "thank", "you", "amazing", "almost", "incredible", "the", "fact", "that", "you", "use", "gmail" ], t.tokenize(s))

    def test_remove_url_without_protocol(self):
        """
        Test the URL removal functionality when there is no protocol.
        """

        s = "Thank you @BillGates. It's amazing, almost as incredible as the fact that you use Gmail. t.co/drawyFHHQM"
        t = Tokenizer(remove_urls=True, stem=False)
        self.assertEqual([ "thank", "you", "amazing", "almost", "incredible", "the", "fact", "that", "you", "use", "gmail" ], t.tokenize(s))

    def test_remove_url_with_http_protocol(self):
        """
        Test the URL removal functionality when the protocol is http.
        """

        s = "Thank you @BillGates. It's amazing, almost as incredible as the fact that you use Gmail. http://t.co/drawyFHHQM"
        t = Tokenizer(remove_urls=True, stem=False)
        self.assertEqual([ "thank", "you", "amazing", "almost", "incredible", "the", "fact", "that", "you", "use", "gmail" ], t.tokenize(s))

    def test_remove_subdomain(self):
        """
        Test that URL removal includes subdomains.
        """

        s = "Visit Multiplex's documentation for more information: https://nicholasmamo.github.io/multiplex-plot/"
        t = Tokenizer(remove_urls=True, stem=False)
        self.assertEqual([ "visit", "multiplex", "documentation", "for", "more", "information" ], t.tokenize(s))

    def test_remove_subdomain_without_protocol(self):
        """
        Test that URL removal includes subdomains even if they have no protocol.
        """

        s = "Visit Multiplex's documentation for more information: nicholasmamo.github.io/multiplex-plot/"
        t = Tokenizer(remove_urls=True, stem=False)
        self.assertEqual([ "visit", "multiplex", "documentation", "for", "more", "information" ], t.tokenize(s))

    def test_retain_url(self):
        """
        Test the URL retention functionality.
        """

        s = "Thank you @BillGates. It's amazing, almost as incredible as the fact that you use Gmail. https://t.co/drawyFHHQM"
        t = Tokenizer(remove_urls=False, stem=False)
        self.assertEqual([ "thank", "you", "amazing", "almost", "incredible", "the", "fact", "that", "you", "use", "gmail", "https", "drawyfhhqm" ], t.tokenize(s))

    def test_alt_code_removal(self):
        """
        Test the alt-code removal functionality works as it is supposed to.
        """

        s = "Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings"
        t = Tokenizer(remove_alt_codes=True)
        self.assertEqual([ "our", "predict", "base", "fifa", "rank", "countri", "risk", "rate" ], t.tokenize(s))

    def test_retain_alt_codes(self):
        """
        Test that when alt-codes are not set to be removed, they are retained.
        """

        s = "Our prediction based on #FIFA Rankings, &amp; Country Risk Ratings"
        t = Tokenizer(remove_alt_codes=False)
        self.assertEqual([ "our", "predict", "base", "fifa", "rank", "amp", "countri", "risk", "rate" ], t.tokenize(s))

    def test_word_normalization(self):
        """
        Test that word normalization reduces normalized characters.
        """

        t = Tokenizer(normalize_words=True, character_normalization_count=2, stem=False, min_length=2)

        s = "YYYYYEEESSSSSSSSS OOXXXXXXXXXXXX!!!"
        self.assertEqual([ "yes", "ox" ], t.tokenize(s))

        s = "GOOOOOOOOOOOOOOAAAAALL!!!"
        self.assertEqual([ "goal" ], t.tokenize(s))

    def test_word_normalization_character_limit(self):
        """
        Test that word normalization reduces normalized characters.
        """

        s = "YYYYYEEESSSSSSSSS OOXXXXXXXXXXXX!!!"
        t = Tokenizer(normalize_words=True, character_normalization_count=2, stem=False, min_length=2)
        self.assertEqual([ "yes", "ox" ], t.tokenize(s))

    def test_word_normalization_character_exact_limit(self):
        """
        Test that word normalization reduces repeated characters if the count is exact.
        """

        s = "YYYYYEEESSSSSSSSS OOXXXXXXXXXXXX!!!"
        t = Tokenizer(normalize_words=True, character_normalization_count=3, stem=False, min_length=2)
        self.assertEqual([ "yes", "oox" ], t.tokenize(s))

    def test_word_normalization_character_high_limit(self):
        """
        Test that word normalization does not reduce repeated characters if the limit is higher.
        """

        s = "YYYYYEEESSSSSSSSS OOXXXXXXXXXXXX!!!"
        t = Tokenizer(normalize_words=True, character_normalization_count=4, stem=False, min_length=2)
        self.assertEqual([ "yeees", "oox" ], t.tokenize(s))

    def test_no_word_normalization(self):
        """
        Test that word normalization is not permitted, no words are normalized.
        """

        s = "YYYYYEEESSSSSSSSS OOXXXXXXXXXXXX!!!"
        t = Tokenizer(normalize_words=False, character_normalization_count=2, stem=False, min_length=2)
        self.assertEqual([ "yyyyyeeesssssssss", "ooxxxxxxxxxxxx" ], t.tokenize(s))

    def test_unicode_removal(self):
        """
        Test that the unicode entity removal functionality removes unicode characters.
        """

        s = "\u0632\u0648\u062f_\u0641\u0648\u0644\u0648\u0631\u0632_\u0645\u0639_\u0627\u0644\u0645\u0628\u0627\u062d\u062b"
        t = Tokenizer(remove_unicode_entities=True, stem=False, min_length=1, remove_punctuation=False)
        self.assertEqual([ '___' ], t.tokenize(s))

    def test_unicode_removal_includes_emojis(self):
        """
        Test that the unicode entity removal functionality also removes emojis.
        """

        s = "Je veux 😂😂😂🦁"
        t = Tokenizer(remove_unicode_entities=True, stem=False, min_length=1, remove_punctuation=False)
        self.assertEqual([ 'je', 'veux' ], t.tokenize(s))

    def test_retain_unicode(self):
        """
        Test unicode character retention.
        """

        s = "\u0632\u0648\u062f_\u0641\u0648\u0644\u0648\u0631\u0632_\u0645\u0639_\u0627\u0644\u0645\u0628\u0627\u062d\u062b"
        t = Tokenizer(remove_unicode_entities=False, stem=False, min_length=1, remove_punctuation=False)
        self.assertEqual([ 'زود_فولورز_مع_المباحث' ], t.tokenize(s))

    def test_unicode_retention_includes_emojis(self):
        """
        Test that the unicode entity retention also allows emojis.
        """

        s = "Je veux 😂😂😂🦁"
        t = Tokenizer(remove_unicode_entities=False, stem=False, min_length=1, remove_punctuation=False)
        self.assertEqual([ 'je', 'veux', '😂😂😂🦁' ], t.tokenize(s))

    def test_minimum_length_minimum(self):
        """
        Test that when the minimum length is 1, all non-empty tokens are retained.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(min_length=1, stem=False)
        self.assertEqual([ 'gelson', 'martins', 'tries', 'to', 'shove', 'the', 'referee' ], t.tokenize(s))

    def test_minimum_length_exact(self):
        """
        Test that the minimum length does not filter tokens with exact length.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(min_length=2, stem=False)
        self.assertEqual([ 'gelson', 'martins', 'tries', 'to', 'shove', 'the', 'referee' ], t.tokenize(s))

    def test_minimum_length_filter(self):
        """
        Test that the minimum length filters tokens with length less than it.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(min_length=3, stem=False)
        self.assertEqual([ 'gelson', 'martins', 'tries', 'shove', 'the', 'referee' ], t.tokenize(s))

    def test_no_stopwords(self):
        """
        Test that when no stopwords are provided, no tokens are removed.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(stem=False)
        self.assertEqual([ 'gelson', 'martins', 'tries', 'shove', 'the', 'referee' ], t.tokenize(s))

    def test_stopwords_empty_list(self):
        """
        Test that when an empty list of stopwords is provided, no tokens are removed.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(stem=False, stopwords=[])
        self.assertEqual([ 'gelson', 'martins', 'tries', 'shove', 'the', 'referee' ], t.tokenize(s))

    def test_stopwords_empty_dict(self):
        """
        Test that when an empty dictionary of stopwords is provided, no tokens are removed.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(stem=False, stopwords={})
        self.assertEqual([ 'gelson', 'martins', 'tries', 'shove', 'the', 'referee' ], t.tokenize(s))

    def test_stopwords_list(self):
        """
        Test that when a list of stopwords is provided, tokens in that list are removed.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(stem=False, stopwords=[ 'tries' ])
        self.assertEqual([ 'gelson', 'martins', 'shove', 'the', 'referee' ], t.tokenize(s))

    def test_stopwords_dict(self):
        """
        Test that when a dictionary of stopwords is provided, tokens in that dictionary are removed.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(stem=False, stopwords={ 'tries': 1 })
        self.assertEqual([ 'gelson', 'martins', 'shove', 'the', 'referee' ], t.tokenize(s))

    @ignore_warnings
    def test_stopwords_nltk(self):
        """
        Test that when the list of NLTK stopwords is provided, the terms are removed.
        """

        s = "Gelson Martins tries to shove the referee."
        t = Tokenizer(stem=False, stopwords=list(stopwords.words("english")))
        self.assertEqual([ 'gelson', 'martins', 'tries', 'shove', 'referee' ], t.tokenize(s))

    @ignore_warnings
    def test_stopwords_nltk_beginning(self):
        """
        Test that when the list of NLTK stopwords is provided, the terms are removed even from the beginning of the sentence.
        """

        s = "The Premier League clubs have spent 291M on agent fees"
        t = Tokenizer(stem=False, stopwords=list(stopwords.words("english")))
        self.assertEqual([ 'premier', 'league', 'clubs', 'spent', '291m', 'agent', 'fees' ], t.tokenize(s))

    def test_case_folding(self):
        """
        Test that when case folding is set, all tokens are set to lower-case.
        """

        s = "#BREAKING Nine illegal miners killed by rival workers in S. Africa say police"
        t = Tokenizer(case_fold=True, stem=False)
        self.assertEqual([ "breaking", "nine", "illegal", 'miners', 'killed', 'rival', 'workers', 'africa', 'say', 'police' ], t.tokenize(s))

    def test_no_case_folding(self):
        """
        Test that when no case folding is set, all tokens are returned as provided.
        """

        s = "#BREAKING Nine illegal miners killed by rival workers in S. Africa say police"
        t = Tokenizer(case_fold=False, stem=False)
        self.assertEqual([ "BREAKING", "Nine", "illegal", 'miners', 'killed', 'rival', 'workers', 'Africa', 'say', 'police' ], t.tokenize(s))

    def test_punctuation_removal(self):
        """
        Test that when there is punctuation removal, no punctuation is retained.
        """

        s = "Toko-Ekambi scores, assisted by Mendes!"
        t = Tokenizer(remove_punctuation=True, stem=False)
        self.assertEqual([ "toko", "ekambi", "scores", "assisted", "mendes" ], t.tokenize(s))

    def test_no_punctuation_removal(self):
        """
        Test that when there is no punctuation removal, the punctuation sticks to the adjacent token.
        """

        s = "Toko-Ekambi scores, assisted by Mendes!"
        t = Tokenizer(remove_punctuation=False, stem=False)
        self.assertEqual([ "toko-ekambi", "scores,", "assisted", "mendes!" ], t.tokenize(s))

    def test_punctuation_removes_strange_characters(self):
        """
        Test that punctuation removal includes strange characters.
        """

        s = "Today’s line-up includes Kane"
        t = Tokenizer(remove_punctuation=False, stem=False)
        self.assertEqual([ "today’s", "line-up", "includes", "kane" ], t.tokenize(s))
        t = Tokenizer(remove_punctuation=True, stem=False)
        self.assertEqual([ "today", "line", "includes", "kane" ], t.tokenize(s))

    def test_punctuation_retains_emojis(self):
        """
        Test that punctuation removal does not remove emojis.
        """

        s = "Je veux 😂😂😂🦁"
        t = Tokenizer(remove_unicode_entities=False, stem=False, min_length=1, remove_punctuation=True)
        self.assertEqual([ 'je', 'veux', '😂😂😂🦁' ], t.tokenize(s))

    def test_punctuation_retains_unicode(self):
        """
        Test that punctuation removal does not remove unicode characters.
        """

        s = "\u0632\u0648\u062f_\u0641\u0648\u0644\u0648\u0631\u0632_\u0645\u0639_\u0627\u0644\u0645\u0628\u0627\u062d\u062b"
        t = Tokenizer(remove_unicode_entities=False, stem=False, min_length=1, remove_punctuation=True)
        self.assertEqual(['زود', 'فولورز', 'مع', 'المباحث'], t.tokenize(s))

    def test_stemming(self):
        """
        Test that when stemming is performed, suffixes are removed.
        """

        s = "Toko-Ekambi scores, assisted by Mendes!"
        t = Tokenizer(stem=True)
        self.assertEqual([ "toko", "ekambi", "score", "assist", "mend" ], t.tokenize(s))

    def test_stemming_cache(self):
        """
        Test that the stemmer uses a cache.
        """

        s = "Toko-Ekambi scores, assisted by Mendes!"
        t = Tokenizer(stem=True)
        start = time.time()
        self.assertEqual([ "toko", "ekambi", "score", "assist", "mend" ], t.tokenize(s))
        elapsed = time.time() - start
        start = time.time()
        self.assertEqual({ 'toko', 'ekambi', 'scores', 'assisted', 'mendes' }, set(t.stem_cache.keys()))
        self.assertEqual('toko', t.stem_cache.get('toko'))
        self.assertEqual('ekambi', t.stem_cache.get('ekambi'))
        self.assertEqual('score', t.stem_cache.get('scores'))
        self.assertEqual('assist', t.stem_cache.get('assisted'))
        self.assertEqual('mend', t.stem_cache.get('mendes'))
        t.tokenize(s)
        self.assertLess(time.time() - start, elapsed)

    def test_no_stemming(self):
        """
        Test that when there is no stemming, suffixes are retained.
        """

        s = "Toko-Ekambi scores, assisted by Mendes!"
        t = Tokenizer(stem=False)
        self.assertEqual([ "toko", "ekambi", "scores", "assisted", "mendes" ], t.tokenize(s))

    def test_retain_special_characters(self):
        """
        Test that when special_characters are not normalized, they are retained.
        """

        s = 'On ne lâche rien'
        t = Tokenizer(stem=False, min_length=2, normalize_special_characters=False)
        self.assertEqual([ 'on', 'ne', 'lâche', 'rien' ], t.tokenize(s))

    def test_retain_french_special_characters(self):
        """
        Test that when French special_characters are not normalized, they are retained.
        """

        s = 'On ne lâche rien'
        t = Tokenizer(stem=False, min_length=2, normalize_special_characters=False)
        self.assertEqual([ 'on', 'ne', 'lâche', 'rien' ], t.tokenize(s))

    def test_normalize_french_special_characters(self):
        """
        Test that when French special_characters are normalized, they are converted to the correct character.
        """

        s = 'Il joue très gros... Peut-être sa dernière chance'
        t = Tokenizer(stem=False, normalize_special_characters=True)
        self.assertEqual([ 'joue', 'tres', 'gros', 'peut', 'etre', 'derniere', 'chance' ], t.tokenize(s))

    def test_retain_germanic_special_characters(self):
        """
        Test that when Germanic special_characters are not normalized, they are retained.
        """

        s = 'Så leker Özil äntligen'
        t = Tokenizer(stem=False, min_length=2, normalize_special_characters=False)
        self.assertEqual([ 'så', 'leker', 'özil', 'äntligen' ], t.tokenize(s))

    def test_normalize_germanic_special_characters(self):
        """
        Test that when Germanic special_characters are normalized, they are converted to the correct character.
        """

        s = 'Så leker Özil äntligen'
        t = Tokenizer(stem=False, min_length=2, normalize_special_characters=True)
        self.assertEqual([ 'sa', 'leker', 'ozil', 'antligen' ], t.tokenize(s))

    def test_retain_maltese_special_characters(self):
        """
        Test that when Maltese special_characters are not normalized, they are retained.
        """

        s = 'Ċikku żar lil Ġanni il-Ħamrun'
        t = Tokenizer(stem=False, normalize_special_characters=False)
        self.assertEqual([ 'ċikku', 'żar', 'lil', 'ġanni', 'ħamrun' ], t.tokenize(s))

    def test_normalize_maltese_special_characters(self):
        """
        Test that when Maltese special_characters are normalized, they are converted to the correct character.
        Note that this does not apply to Ħ/ħ.
        """

        s = 'Ċikku żar lil Ġanni il-Ħamrun'
        t = Tokenizer(stem=False, normalize_special_characters=True)
        self.assertEqual([ 'cikku', 'zar', 'lil', 'ganni', 'ħamrun' ], t.tokenize(s))

    def test_tokenize_nouns_only(self):
        """
        Test extracting tokens with only nouns.
        """

        text = "I've gone through many revisions of this area, but I think this might be the one."
        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        self.assertEqual([ 'taxi', 'day', 'age' ], t.tokenize('Taxis can really go everywhere in this day and age'))

    def test_tokenize_verbs_only(self):
        """
        Test extracting verbs.
        """

        text = "I have no idea how we balance this but with a midfield with Pogba and Ndombele its curtains for the low blocks that once haunted us."
        t = Tokenizer(pos=[ 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ' ], stem=False)
        self.assertEqual([ 'have', 'balance', 'haunted' ], t.tokenize(text))

        t = Tokenizer(pos=[ 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ' ], stem=False)
        text = "Is Rojo really starting if he doesn't shoot from 40 yards"
        self.assertEqual([ 'starting', 'does', 'shoot' ], t.tokenize(text))

    def test_tokenize_verbs_with_stemming(self):
        """
        Test extracting verbs with stemming.
        """

        text = "I have no idea how we balance this but with a midfield with Pogba and Ndombele its curtains for the low blocks that once haunted us."
        t = Tokenizer(pos=[ 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ' ], stem=True)
        self.assertEqual([ 'have', 'balanc', 'haunt' ], t.tokenize(text))

        t = Tokenizer(pos=[ 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ' ], stem=True)
        text = "Is Rojo really starting if he doesn't shoot from 40 yards"
        self.assertEqual([ 'start', 'doe', 'shoot' ], t.tokenize(text))

    def test_tokenize_adjectives_only(self):
        """
        Test extracting adjectives.
        """

        text = "Last match of the year at @ChelseaFC.. please win it for me."
        t = Tokenizer(pos=[ 'JJ', 'JJR', 'JJS' ], stem=True)
        self.assertEqual([ 'last' ], t.tokenize(text))

    def test_tokenize_pos_multiple_sentences(self):
        """
        Test extracting tokens using parts of speech from multiple sentences.
        """

        text = "Come on boys. Let's get the 3 points!"
        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        self.assertEqual([ 'boy', 'point' ], t.tokenize(text))

    def test_tokenize_tokens(self):
        """
        Test extracting tokens with not just nouns.
        """

        text = "I've gone through many revisions of this area, but I think this might be the one."
        t = Tokenizer(pos=None, min_length=2)
        self.assertEqual([ 'taxi', 'can', 'realli', 'go', 'everywher', 'in', 'thi', 'day', 'and', 'age' ], t.tokenize('Taxis can really go everywhere in this day and age'))

    def test_tokenize_tokens_nouns_subset(self):
        """
        Test that the extracted noun tokens are a subset of all tokens.
        """

        text = "I've gone through many revisions of this area, but I think this might be the one."
        t = Tokenizer(pos=None, min_length=2)
        tokens = t.tokenize('Taxis can really go everywhere in this day and age')
        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        nouns = t.tokenize('Taxis can really go everywhere in this day and age')
        self.assertTrue(all( noun in tokens for noun in nouns ))

    def test_nouns_only_one_sentence(self):
        """
        Test extracting nouns from a single sentence.
        """

        text = "I have no idea how we balance this but with a midfield with Pogba and Ndombele its curtains for the low blocks that once haunted us."
        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        self.assertEqual([ 'idea', 'midfield', 'Pogba', 'Ndombele', 'curtains', 'blocks' ], t._pos(text))

        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        text = "Is Rojo really starting if he doesn't shoot from 40 yards"
        self.assertEqual([ 'rojo', 'yard' ], t.tokenize(text))

    def test_nouns_only_multiple_sentence(self):
        """
        Test extracting nouns from multiple sentences.
        """

        text = "Night Call is now out on Xbox One and Nintendo Switch! We're so proud to see the game there and hope you'll enjoy your ride in Paris."
        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        self.assertEqual([ 'Night', 'Call', 'Xbox', 'One', 'Nintendo', 'Switch', 'game', 'ride', 'Paris' ], t._pos(text))

    def test_nouns_only_punctuation(self):
        """
        Test that no sentence-delimiting punctuation is retained when extracting nouns.
        """

        text = "Night Call is now out on Xbox One, Android and Nintendo Switch! We're so proud to see the game there and hope you'll enjoy your ride in Paris."
        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        self.assertFalse(any( any( p in noun for p in [ '!', ',', '.' ] ) for noun in t._pos(text) ))

    def test_nouns_only_retain_case(self):
        """
        Test that when extracting nouns, the case is retained.
        """

        text = "A week in the life of Arsenal Football Club"
        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        self.assertEqual([ 'week', 'life', 'Arsenal', 'Football', 'Club' ], t._pos(text))

    def test_nouns_proper(self):
        """
        Test that proper nouns are extracted from text.
        """

        text = "Tanguy Ndombele told Jose Mourinho he never wants to play for him again following a clash earlier this week"
        t = Tokenizer(pos=[ 'NN', 'NNS', 'NNP', 'NNPS' ])
        self.assertEqual([ 'Tanguy', 'Ndombele', 'Jose', 'Mourinho', 'clash', 'week' ], t._pos(text))
