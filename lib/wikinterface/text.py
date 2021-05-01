"""
The text module contains functions to help extract text from articles.
"""

import json
import math
import urllib.request

from . import *

def collect(titles, introduction_only=False):
    """
    Get the plain text content of the pages with the given titles.

    :param titles: The title, or titles of the pages from where to get the plain text.
    :type titles: list of str or str
    :param introduction_only: A boolean indicating whether to get only the introduction's text.
    :type introduction_only: bool

    :return: A dictionary with page titles as keys and the text as values.
    :rtype: dict

    :raises RuntimeError: When the request returns an error response.
    """

    text = { }

    titles = titles if type(titles) is list else [ titles ]

    stagger = 20 if introduction_only else 1
    parameters = {
        'format': 'json',
        'action': 'query',
        'prop': 'extracts',
        'exlimit': stagger, # the number of text extracts to retrieve at a time
        'explaintext': True,
        'titles': urllib.parse.quote('|'.join(titles)),
        'redirects': True, # allow redirects
        'exintro': introduction_only, # get only the introduction
        'excontinue': 0, # the page number from where to continue
    }

    """
    When there are many page titles, the GET parameters could become far too long.
    Therefore in such cases, stagger the process.
    """
    if len(urllib.parse.quote('|'.join(titles))) > 1024:
        for i in range(0, math.ceil(len(titles) / stagger)):
            subset = collect(titles[(i * stagger):((i + 1) * stagger)],
                             introduction_only=introduction_only)
            text.update(subset)
        return text

    """
    If page titles are given, collect them.
    Pages are returned 20 at a time.
    When this happens, the response contains a continue marker.
    The loop continues fetching requests until there are no such markers.
    """
    if len(titles):
        while parameters['excontinue'] is not None:
            endpoint = construct_url(parameters)
            response = urllib.request.urlopen(endpoint)
            response = json.loads(response.read().decode("utf-8"))

            if is_error_response(response):
                raise RuntimeError(response)

            """
            Extract the pages, redirects and the text from the response.
            Put the original page titles as keys, reverting any redirects.
            """
            pages = response['query']['pages']
            redirects = response['query']['redirects'] if 'redirects' in response['query'] else {}
            pages = { page['title']: page.get('extract') for page in pages.values() if 'extract' in page }
            pages = revert_redirects(pages, redirects)
            text.update(pages)

            parameters['excontinue'] = response['continue']['excontinue'] if 'continue' in response else None

    return text
