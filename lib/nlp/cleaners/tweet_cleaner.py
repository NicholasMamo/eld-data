"""
The tweet cleaner builds on the base cleaner, but adds functionality that is specific to Twitter and other informal text in general.
The Twitter-specific functionality involves retweet prefixes (`RT` at the start of the tweet), hashtags (such as `#EvenTDT`) and mentions (like `@NicholasMamo`).
The tweet cleaner is also capable of removing unicode entities, like certain emojis, and URLs.
"""

import re

from . import Cleaner

class TweetCleaner(Cleaner):
    """
    The tweet cleaner removes needless information from a text to make it presentable.
    This includes Twitter-specific syntax and URLs.

    By default, the tweet cleaner only strips the text of whitespaces at the beginning or at the end of text.
    You can add other cleaning steps using the appropriate parameters when creating a :class:`~nlp.cleaners.tweet_cleaner.TweetCleaner`.
    In addition, you can also specify additional parameters from the base :class:`~nlp.cleaners.Cleaner`.

    :ivar remove_unicode_entities: A boolean indicating whether unicode entities should be removed.
                                   Note that this also includes emojis.
    :vartype remove_unicode_entities: bool
    :ivar remove_urls: A boolean indicating whether URLs should be removed.
    :vartype remove_urls: bool
    :ivar remove_hashtags: A boolean indicating whether hashtags that cannot be split should be removed.
    :vartype remove_hashtags: bool
    :ivar split_hashtags: A boolean indicating whether hashtags should be split.
    :vartype split_hashtags: bool
    :ivar remove_retweet_prefix: A boolean indicating whether the retweet prefix should be removed.
    :vartype remove_retweet_prefix: bool
    :ivar replace_mentions: A boolean indicating whether mentions should be replaced with the user's display name.
    :vartype replace_mentions: bool
    """

    def __init__(self, remove_unicode_entities=False,
                       remove_urls=False,
                       remove_hashtags=False,
                       split_hashtags=False,
                       remove_retweet_prefix=False,
                       replace_mentions=False, *args, **kwargs):
        """
        Create the tweet cleaner by specifying the type of cleaning operations it should perform.
        This design is purposeful so that the tweet cleaner processes all text in the same way.

        The same configuration accepted by the :class:`~nlp.cleaners.Cleaner` are accepted as arguments and keyword arguments.
        They are then passed on to the parent constructor, :func:`~nlp.cleaners.Cleaner.__init__`.

        :param remove_unicode_entities: A boolean indicating whether unicode entities should be removed.
                                        Note that this also includes emojis.
        :type remove_unicode_entities: bool
        :param remove_urls: A boolean indicating whether URLs should be removed.
        :type remove_urls: bool
        :param remove_hashtags: A boolean indicating whether hashtags that cannot be split should be removed.
        :type remove_hashtags: bool
        :param split_hashtags: A boolean indicating whether hashtags should be split.
        :type split_hashtags: bool
        :param remove_retweet_prefix: A boolean indicating whether the retweet prefix should be removed.
        :type remove_retweet_prefix: bool
        :param replace_mentions: A boolean indicating whether mentions should be replaced with the user's display name.
        :type replace_mentions: bool
        """

        super(TweetCleaner, self).__init__(*args, **kwargs)

        self.remove_unicode_entities = remove_unicode_entities
        self.remove_urls = remove_urls
        self.remove_hashtags = remove_hashtags
        self.split_hashtags = split_hashtags
        self.remove_retweet_prefix = remove_retweet_prefix
        self.replace_mentions = replace_mentions

    def clean(self, text, tweet=None):
        """
        Clean the given text.
        The basic cleaner always strips empty whitespaces before any pre-processing.

        The ``tweet`` parameter is optional, and is only required when replacing tweet mentions by the respective user's screen name (for example, replacing `@NicholasMamo` to `Nicholas Mamo`).
        In this case, the function requires the original ``tweet`` object to look for the ``user`` object, which contains details for replacing the username.

        :param text: The text to clean.
        :type text: str
        :param tweet: The tweet to use to clean the tweet.
                      This is only used when mentioned users' names are replaced in the text.
        :type tweet: dict

        :return: The cleaned text.
        :rtype: str
        """

        text = text.strip()
        text = self._collapse_new_lines(text) if self.collapse_new_lines else text
        text = self._remove_alt_codes(text) if self.remove_alt_codes else text
        text = self._remove_unicode_entities(text) if self.remove_unicode_entities else text
        text = self._remove_urls(text) if self.remove_urls else text
        text = self._split_hashtags(text) if self.split_hashtags else text
        text = self._remove_hashtags(text) if self.remove_hashtags else text
        text = self._remove_retweet_prefix(text) if self.remove_retweet_prefix else text
        text = self._complete_sentences(text) if self.complete_sentences else text
        text = self._collapse_whitespaces(text) if self.collapse_whitespaces else text
        text = self._replace_mentions(text, tweet) if self.replace_mentions else text
        text = self._capitalize_first(text) if self.capitalize_first else text
        text = text.strip()

        return text

    def _remove_unicode_entities(self, text):
        """
        Remove unicode entities, including emojis, from the given text.

        :param text: The tweet to be cleaned.
        :type text: str

        :return: The tweet without unicode entities.
        :rtype: str
        """

        return text.encode('ascii', 'ignore').decode("utf-8")

    def _remove_urls(self, text):
        """
        Remove Twitter short URLs from the text.

        :param text: The text to clean().
        :type text: str

        :return: The text without URLs.
        :rtype: str
        """

        url_pattern = re.compile("(https?:\/\/)?([^\s]+)?\.[a-zA-Z0-9]+?\/?([^\s,\.]+)?")
        return url_pattern.sub(' ', text)

    def _split_hashtags(self, text):
        """
        Split the hashtags in the given text based on camel-case notation.

        :param text: The text to normalize.
        :type text: str

        :return: The text with split hashtags.
        :rtype: str
        """

        hashtag_pattern = re.compile("#([a-zA-Z0-9_]+)")
        camel_case_pattern = re.compile("(([a-z]+)?([A-Z]+|[0-9]+))")

        """
        First find all hashtags.
        Then, split them and replace the hashtag lexeme with the split components.
        """
        hashtags = hashtag_pattern.findall(text)
        for hashtag in hashtags:
            components = camel_case_pattern.sub("\g<2> \g<3>", hashtag)

            """
            Only split hashtags that have multiple components.
            If there is only one component, it's just a hashtag.
            """
            if len(components.split()) > 1:
                text = text.replace(f"#{hashtag}", components)

        return text

    def _remove_hashtags(self, text):
        """
        Remove hashtags from the given text.

        :param text: The text to clean.
        :type text: str

        :return: The text without any hashtags.
        :rtype: str
        """

        hashtag_pattern = re.compile("#([a-zA-Z0-9_]+)")
        return hashtag_pattern.sub(' ', text)

    def _remove_retweet_prefix(self, text):
        """
        Remove retweet syntax from a tweet.
        Retweets start with the text 'RT @user: '

        :param text: The text to clean.
        :type text: str

        :return: The cleaned text.
        :rtype: str
        """

        retweet_pattern = re.compile('^RT @.+?: ')

        return retweet_pattern.sub(' ', text)

    def _replace_mentions(self, text, tweet):
        """
        Replace all mentions in the text with the user's display name.

        :param text: The text to clean.
        :type text: str
        :param tweet: The tweet dictionary.
        :type tweet: dict

        :return: The cleaned text.
        :rtype: str

        :raises ValueError: When the tweet is not given and the mentions should be replaced.
        """

        if self.replace_mentions and not tweet:
            raise ValueError("The tweet must be given in order to replace mentions.")

        """
        Create a mapping between user mentions and their display names.
        User mentions can appear in:

            #. The base tweet,
            #. The retweeted tweet, and
            #. The quoted status.
        """
        mentions = { }
        mentions.update({ f"@{ mention['screen_name'] }": mention['name']
                          for mention in tweet['entities']['user_mentions'] })
        mentions.update({ f"@{ mention['screen_name'] }": mention['name']
                          for mention in tweet.get('extended_tweet', { }).get('entities', { }).get('user_mentions', { }) })
        mentions.update({ f"@{ mention['screen_name'] }": mention['name']
                          for mention in tweet.get('retweeted_status', { }).get('entities', { }).get('user_mentions', { }) })
        mentions.update({ f"@{ mention['screen_name'] }": mention['name']
                          for mention in tweet.get('retweeted_status', { }).get('extended_tweet', { }).get('entities', { }).get('user_mentions', { }) })
        mentions.update({ f"@{ mention['screen_name'] }": mention['name']
                          for mention in tweet.get('quoted_status', { }).get('entities', { }).get('user_mentions', { }) })
        mentions.update({ f"@{ mention['screen_name'] }": mention['name']
                          for mention in tweet.get('quoted_status', { }).get('extended_tweet', { }).get('entities', { }).get('user_mentions', { }) })

        for handle, name in mentions.items():
            if '\\' in name:
                continue
            pattern = re.compile(f"{ re.escape(handle) }\\b", flags=re.I)
            text = re.sub(pattern, name, text)

        return text
