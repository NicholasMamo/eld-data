#!/usr/bin/env python3

"""
The IDF tool creates a TF-IDF scheme from a corpus of tweets.
To generate the TF-IDF scheme, you must always provide the corpus of tweets (as collected using the :mod:`~tools.collect` tool) and the file where to write the IDF:

.. code-block:: bash

    ./tools/idf.py \\
    --file data/sample.json \\
    --output data/idf.json

You can optionally skip retweets by passing the ``--remove-retweets`` parameter.

In addition to the basic functionality, this tool lets you specify how to pre-process tokens.
The functions include common approaches, like stemming, as well as character normalization, which removes repeated characters:

.. code-block:: bash

    ./tools/idf.py \\
    --file data/sample.json \\
    --output data/idf.json \\
    --remove-unicode-entities \\
    --normalize-words --stem

The output is a JSON file with the following structure:

.. code-block:: json

    {
        "cmd": {
            "file": "data/sample.json",
            "output": "data/idf.json",
            "remove_retweets": false,
            "remove_unicode_entities": false,
            "skip_unverified": false,
            "normalize_words": false,
            "character_normalization_count": 3,
            "stem": false,
            "_date": "2020-10-24T14:41:24.809058",
            "_timestamp": 1603543284.8090684,
            "_cmd": "/home/nicholas/github/EvenTDT/tools/idf.py --file data/sample.json --output data/idf.json"
        },
        "pcmd": {
            "file": "data/sample.json",
            "output": "data/idf.json",
            "remove_retweets": false,
            "skip_unverified": false,
            "remove_unicode_entities": false,
            "normalize_words": false,
            "character_normalization_count": 3,
            "stem": false,
            "_date": "2020-10-24T14:41:24.809077",
            "_timestamp": 1603543284.80908,
            "_cmd": "/home/nicholas/github/EvenTDT/tools/idf.py --file data/sample.json --output data/idf.json"
        },
        "tfidf": {
            "class": "<class 'nlp.weighting.tfidf.TFIDF'>",
            "idf": {
                "class": "<class 'nlp.weighting.global_schemes.idf.IDF'>",
                "documents": 15135,
                "idf": {
                    "teams": 118,
                    "arsenal": 8144,
                    "burnley": 243,
                    "now": 802,
                    "order": 48,
                    "and": 2776
                }
            }
        }
    }

The full list of accepted arguments:

    - ``-f --file``                          *<Required>* The file to use to construct the TF-IDF scheme.
    - ``-o --output``                        *<Required>* The file where to save the TF-IDF scheme.
    - ``--remove-retweets``                  *<Optional>* Exclude retweets from the corpus.
    - ``--skip-unverified``                  *<Optional>* Skip tweets from unverified authors when reading tweets, defaults to False.
    - ``--remove-unicode-entities``          *<Optional>* Remove unicode entities from the TF-IDF scheme.
    - ``--normalize-words``                  *<Optional>* Normalize words with repeating characters in them.
    - ``--character-normalization-count``    *<Optional>* The number of times a character must repeat for it to be normalized. Used only with the ``--normalize-words`` flag.
    - ``--stem``                             *<Optional>* Stem the tokens when constructing the TF-IDF scheme.
"""

import argparse
import json
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(file_path, '..')
lib = os.path.join(root, 'lib')
sys.path.insert(-1, root)
sys.path.insert(-1, lib)

from nlp.weighting.tfidf import TFIDF
from nlp.tokenizer import Tokenizer
from objects.exportable import Exportable
import tools
import twitter

def setup_args():
    """
    Set up and get the list of command-line arguments.

    Accepted arguments:

        - ``-f --file``                          *<Required>* The file to use to construct the TF-IDF scheme.
        - ``-o --output``                        *<Required>* The file where to save the TF-IDF scheme.
        - ``--remove-retweets``                  *<Optional>* Exclude retweets from the corpus.
        - ``--skip-unverified``                  *<Optional>* Skip tweets from unverified authors when reading tweets, defaults to False.
        - ``--remove-unicode-entities``          *<Optional>* Remove unicode entities from the TF-IDF scheme.
        - ``--normalize-words``                  *<Optional>* Normalize words with repeating characters in them.
        - ``--character-normalization-count``    *<Optional>* The number of times a character must repeat for it to be normalized. Used only with the ``--normalize-words`` flag.
        - ``--stem``                             *<Optional>* Stem the tokens when constructing the TF-IDF scheme.

    :return: The command-line arguments.
    :rtype: :class:`argparse.Namespace`
    """

    parser = argparse.ArgumentParser(description="Create a TF-IDF scheme from a corpus of tweets.")

    """
    Parameters that define how the corpus should be collected.
    """

    parser.add_argument('-f', '--file', type=str, required=True,
                        help='<Required> The file to use to construct the TF-IDF scheme.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='<Required> The file where to save the TF-IDF scheme.')
    parser.add_argument('--remove-retweets', action="store_true",
                        help='<Optional> Exclude retweets from the corpus.')
    parser.add_argument('--skip-unverified', action="store_true",
                        help='<Optional> Skip tweets from unverified authors when reading tweets, defaults to False.')
    parser.add_argument('--remove-unicode-entities', action="store_true",
                        help='<Optional> Remove unicode entities from the TF-IDF scheme.')
    parser.add_argument('--normalize-words', action="store_true",
                        help='<Optional> Normalize words with repeating characters in them.')
    parser.add_argument('--character-normalization-count', type=int, required=False, default=3,
                        help='<Optional> The number of times a character must repeat for it to be normalized. Used only with the --normalize-words flag.')
    parser.add_argument('--stem', action="store_true",
                        help='<Optional> Stem the tokens when constructing the TF-IDF scheme.')

    args = parser.parse_args()
    return args

def main():
    """
    Main program loop.
    """

    args = setup_args()
    tfidf = construct(file=args.file, remove_retweets=args.remove_retweets, skip_unverified=args.skip_unverified,
                      normalize_words=args.normalize_words, character_normalization_count=args.character_normalization_count,
                      remove_unicode_entities=args.remove_unicode_entities, stem=args.stem)
    cmd = tools.meta(args)
    pcmd = tools.meta(args)
    tools.save(args.output, { 'cmd': cmd, 'pcmd': pcmd, 'tfidf': tfidf })

def construct(file, remove_retweets=False, skip_unverified=False, *args, **kwargs):
    """
    Construct the TF-IDF scheme from the file.
    The scheme is constructed one line at a time.

    Any additional arguments and keyword arguments are passed on to the :func:`~nlp.tokenizer.Tokenizer.__init__` constructor.

    :param file: The path to the file to use to construct the TF-IDF scheme.
    :type file: str
    :param remove_retweets: A boolean indicating whether to xclude retweets from the corpus.
    :type remove_retweets: bool
    :param skip_unverified: Skip tweets from unverified authors when reading tweets.
    :type skip_unverified: bool

    :return: The TF-IDF scheme constructed from the file.
    :rtype: :class:`~nlp.weighting.tfidf.TFIDF`
    """

    documents, idf = 0, { }
    tokenizer = Tokenizer(*args, **kwargs)

    """
    Open the file and iterate over every tweet.
    Tokenize those tweets and use them to update the TF-IDF table.
    """
    with open(file, 'r') as f:
        for line in f:
            tweet = json.loads(line)

            """
            Skip the tweet if retweets should be excluded.
            """
            if remove_retweets and twitter.is_retweet(tweet):
                continue

            """
            Skip tweets from unverified authors if they should be excluded.
            """
            if skip_unverified and not twitter.is_verified(tweet):
                continue

            documents = documents + 1
            tokens = tokenize(tweet, tokenizer)
            idf = update(idf, tokens)

    return TFIDF(documents=documents, idf=idf)

def tokenize(tweet, tokenizer):
    """
    Convert the given tweet into a document.
    The text used depends on the type of tweet.
    The full text is always sought.

    :param tweet: The tweet to tokenize.
    :type tweet: dict
    :param tokenizer: The tokenizer to use to tokenize the tweet.
    :type tokenizer: :class:`~nlp.tokenizer.Tokenizer`

    :return: A list of tokens from the tweet.
    :rtype: list of str
    """

    text = twitter.full_text(tweet)
    return tokenizer.tokenize(text)

def update(idf, tokens):
    """
    Update the given IDF table with the given tokens.

    :param idf: The IDF table as a dictionary.
                The keys are the tokens and the values are the document frequencies.
    :type idf: dict
    :param tokens: The tokens to add to the IDF.
                   The function automatically gets the set of tokens to remove duplicates.
    :type: list of str

    :return: The updated IDF table.
    :rtype: dict
    """

    for token in set(tokens):
        idf[token] = idf.get(token, 0) + 1

    return idf

if __name__ == "__main__":
    main()
