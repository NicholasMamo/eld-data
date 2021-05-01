"""
The info collector module fetches information about Wikipedia pages.
This information is not always readily-available, such as the article type.
"""

from enum import Enum
import json
import math
import re
import urllib.request

from . import *
from . import text

class ArticleType(Enum):
    """
    This module supports several article types and records them as an enum.

    :cvar NORMAL: A normal article on Wikipedia.
                  The title of a normal page refers to one, and only one article.
    :vartype NORMAL: int
    :cvar DISAMBIGUATION: A disambiguation page is returned when a title could not be resolved to a particular Wikipedia article.
                          This happens when the title is too general, such as the surname `Gomez`, which can refer to many people.
                          Disambiguation articles articles contain links to other articles that are related to the queried title.
    :vartype NORMAL: int
    :cvar MISSING: When the title of a page could not be found, the page is said to be missing.
    :vartype MISSING: int
    :cvar LIST: List pages are different from disambiguation articles because they list a number of related concepts.
                The page of `Gomez` is a disambiguation page because many people have that surname.
                Conversely, the `List of foreign Premier League players` links to pages that represent foreign Premier League players.
    :vartype LIST: int
    """

    NORMAL = 0
    DISAMBIGUATION = 1
    MISSING = 2
    LIST = 3

def types(titles):
    """
    Get the type of page each page title returns.
    The page type is one of the values in the :class:`~wikinterface.info.ArticleType` enum.

    :param titles: A title, or a list of titles of the pages whose types will be retrieved.
    :type titles: list of str or str

    :return: A dictionary with page titles as keys and the types as :class:`~wikinterface.info.ArticleType` values.
    :rtype: dict
    """

    types = { }

    titles = titles if type(titles) is list else [ titles ]

    stagger = 20
    parameters = {
        'format': 'json',
        'action': 'query',
        'prop': 'pageprops',
        'titles': urllib.parse.quote('|'.join(titles)),
        'redirects': True, # allow redirects
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
            types.update(subset)
        return types

    """
    If page titles are given, collect information about them.
    Pages are returned 20 at a time.
    When this happens, the response contains a continue marker.
    The loop continues fetching requests until there are no such markers.
    """
    if len(titles):
        while parameters['excontinue'] is not None:
            endpoint = construct_url(parameters)
            response = urllib.request.urlopen(endpoint)
            response = json.loads(response.read().decode('utf-8'))

            if is_error_response(response):
                raise RuntimeError(response)

            """
            Extract the page types from the responses.
            """
            pages = response['query']['pages']
            redirects = response['query']['redirects'] if 'redirects' in response['query'] else {}

            """
            Go through each page and find its type.
            By default, pages are normal.
            If the page has no properties, it is considered to be missing.
            """
            for page in pages.values():
                title = page['title']

                if title.lower().startswith("list of"):
                    types[title] = ArticleType.LIST
                    continue

                if 'pageprops' not in page:
                    types[title] = ArticleType.MISSING
                    continue

                if 'disambiguation' in page.get('pageprops'):
                    types[title] = ArticleType.DISAMBIGUATION
                    continue

                types[title] = ArticleType.NORMAL

            parameters['excontinue'] = response['continue']['excontinue'] if 'continue' in response else None

        """
        Put the original page titles as keys.
        This is useful in case there were any redirects.
        """
        types = revert_redirects(types, redirects)

    return types

def is_person(titles):
    """
    Go through each page title and check whether it represents a person.
    The function assumes that an article is about a person if it mentions a birth date.

    Since date of births are not standardized across Wikipedia, this function checks the text using regular expression patterns.

    :param titles: A title, or a list of titles of the pages that will be checked.
    :type titles: list of str or str

    :return: A dictionary with page titles as keys and booleans as values indicating whether the article represents a person.
    :rtype: dict
    """

    classes = { }

    """
    Collect the introductions of the given articles.
    """
    titles = titles if type(titles) is list else [ titles ]
    extracts = text.collect(titles, introduction_only=True)

    """
    An article is about a person if it has a birth date.
    The function tries to capture this in the text using one of a number of regular expressions.
    """
    birth_patterns = [
        re.compile("born [0-9]{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) [0-9]{2,4}"),
        re.compile("born in [0-9]{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) [0-9]{2,4}"), # NOTE: Returned by API in https://en.wikipedia.org/wiki/Dar%C3%ADo_Benedetto
        re.compile("born (January|February|March|April|May|June|July|August|September|October|November|December) [0-9]{1,2},? [0-9]{2,4}"),
        re.compile("born \([0-9]{4}-[0-9]{2}-[0-9]{2}\)(January|February|March|April|May|June|July|August|September|October|November|December) [0-9]{1,2},? [0-9]{2,4}"), # NOTE: Returned by API in https://en.wikipedia.org/wiki/Valentina_Shevchenko_(fighter)
    ]

    for page, introduction in extracts.items():
        classes[page] = any(len(birth_pattern.findall(introduction)) for birth_pattern in birth_patterns)

    return classes
