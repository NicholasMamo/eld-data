"""
The Twitter package is used to facilitate collecting, reading and processing tweet corpora.
At the package-level there are functions to help with general processing tasks.
"""

from dateutil.parser import parse

def extract_timestamp(tweet):
    """
    Get the timestamp from the given tweet.
    This function looks for the timestamp in one of two fields:

    1. ``timestamp_ms``: always present in the top-level tweets, and
    2. ``created_at``: present in ``retweeted_status``, for example.

    :param tweet: The tweet from which to extract the timestamp.
    :type tweet: dict

    :return: The timestamp of the tweet.
    :rtype: int

    :raises KeyError: When no timestamp field can be found.
    """

    if 'timestamp_ms' in tweet:
        timestamp_ms = int(tweet["timestamp_ms"])
        timestamp_ms = timestamp_ms - timestamp_ms % 1000
        return timestamp_ms / 1000.
    elif 'created_at' in tweet:
        return parse(tweet['created_at']).timestamp()

    raise KeyError("Neither the 'timestamp_ms' attribute, nor the 'created_at' attribute could be found in the tweet.")

def full_text(tweet):
    """
    Extract the full text from the tweet.

    Normally, long tweets are truncated (they end with a `…`).
    This function looks for the original full text.
    If it's a retweet, the text is somewhere in the ``retweeted_status``.
    If the tweet has an ``extended_tweet`` attribute, then the ``full_text`` may be set there.
    Otherwise, the function defaults to using the ``text``.

    :param tweet: The tweet from which to extract the timestamp.
    :type tweet: dict

    :return: The full text of the tweet.
    :rtype: str
    """

    while "retweeted_status" in tweet:
        tweet = tweet["retweeted_status"]

    if "extended_tweet" in tweet:
        text = tweet["extended_tweet"].get("full_text", tweet.get("text", ""))
    else:
        text = tweet.get("text", "")

    return text

def is_retweet(tweet):
    """
    Check whether the given tweet is a retweet.
    A tweet is a retweet if it has a ``retweeted_status`` key.

    :param tweet: The tweet to check.
    :type tweet: dict

    :return: A boolean indicating whether the tweet is a retweet.
    :rtype: bool
    """

    return 'retweeted_status' in tweet
