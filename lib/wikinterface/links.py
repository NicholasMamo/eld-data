"""
The link collector module collects links from Wikipedia articles.
These functions are useful to collect the Wikipedia graph, connecting articles with each other through links.
"""

import json
import math
import re
import urllib.request

from . import *

def collect(titles, separate=True, introduction_only=False):
    """
    Get a list of outgoing links from the given list of articles.

    :param titles: A title, or a list of article titles from where to fetch links.
    :type titles: list of str or str
    :param separate: A boolean indicating whether the sets of links should be separated according to the articles where they were found.
    :type separate: bool
    :param introduction_only: A boolean indicating whether the links should be fetched only from the introduction.
    :type introduction_only: bool

    :return: A list of outgoing links from the article titles provided.
             If the ``separate`` boolean is set to ``True``, a dictionary is returned.
             This dictionary's keys are the article titles, and the values are lists of outgoing links.
             Otherwise, a list of article titles is returned.
    :rtype: list or dict

    :raises RuntimeError: When the request returns an error response.
    """

    links = { }

    titles = titles if type(titles) is list else [ titles ]

    stagger = 20
    """
    Some parameters are required only when using the introduction to collect links.
    Similarly, others are required only when all links are required.

    By default, ``plcontinue`` is not included.
    It is added when a valid value is given.
    When it changes to ``None``, the algorithm stops collecting links.
    """
    parameters = {
        'format': 'json',
        'action': 'query',
        'prop': 'revisions' if introduction_only else 'links',
        'redirects': True,
        'rvprop': 'content' if introduction_only else False,
        'rvsection': 0 if introduction_only else False,
        'explaintext': introduction_only,
        'pllimit': 500 if not introduction_only else False,
        'titles': urllib.parse.quote('|'.join(titles)),
        'plcontinue': False,
    }

    if len(urllib.parse.quote('|'.join(titles))) > 1024:
        """
        When there are many page titles, the GET parameters could become far too long.
        Therefore in such cases, stagger the process.
        Pages are always fetched separately, and then merged later if need be.
        """
        for i in range(0, math.ceil(len(titles)/float(stagger))):
            new_links = collect(titles[(i * stagger):((i + 1) * stagger)],
                                separate=True, introduction_only=introduction_only)
            for title, link_set in new_links.items():
                links[title] = links.get(title, []) + link_set
    elif len(titles) > 50:
        """
        At most, 50 pages may be requested at a time.
        If the number of titles exceeds this, the function splits the calls.
        """
        for i in range(0, math.ceil(len(titles)/50)):
            new_links = collect(titles[(i * 50):((i + 1) * 50)],
                                separate=True, introduction_only=introduction_only)
            for title, link_set in new_links.items():
                links[title] = links.get(title, []) + link_set
    elif len(titles):
        """
        If page titles are given, collect their links.
        Pages are returned 20 at a time.
        When this happens, the response contains a continue marker.
        The loop continues fetching requests until there are no such markers.
        """
        while parameters['plcontinue'] is not None:
            endpoint = construct_url(parameters)
            response = urllib.request.urlopen(endpoint)
            response = json.loads(response.read().decode("utf-8"))

            if is_error_response(response):
                raise RuntimeError(response)

            pages = response['query']['pages']
            redirects = response['query']['redirects'] if 'redirects' in response['query'] else {}

            """
            Extract the links from each article.
            Depending on whether the introduction is being used or the entire article, extract them differently.
            Then, revert any redirections.
            """
            for page in pages:
                internal_links = _get_intro_links(pages[page]) if introduction_only else _get_all_links(pages[page])
                links[pages[page]['title']] = links.get(pages[page]['title'], []) + internal_links

            links = revert_redirects(links, redirects)

            """
            If there are no more links to fetch, stop iterating
            If the limit was reached, the collector must have stopped midway
            Therefore, the endpoint is updated to continue looking from where it left off
            """
            if 'continue' in response:
                parameters['plcontinue'] = urllib.parse.quote(response['continue']['plcontinue'])
            else:
                break

    """
    If the links need to be separate, get the unique links for each page.
    Otherwise, flatten the links and get a unique set.
    """
    if separate:
        return { page: list(set(links[page])) for page in links }
    else:
        return list(set([ link for link_set in links.values() for link in link_set ]))

def collect_recursive(titles, level, collected_links=None, separate=True, *args, **kwargs):
    """
    Recursively fetch links, starting from a seed set.
    The function uses the :func:`~wikinterface.links.collect` function.
    Any additional arguments and keyword arguments are passed on to it.

    .. warning::

        This is a very expensive operation as it grows exponentially.

    :param titles: The articles from where to start (or continue) looking.
    :type titles: list of str or str
    :param level: The number of times to fetch links recursively.
                  If the level is 1, the links are fetched once and the function returns.
                  Internally, this parameter is decreased at each recursion level.
    :type level: int
    :param collected_links: Any links that have already been collected.
                            Normally, you do not need to provide it.
                            It is used internally to keep track of the colleccted links.
    :type collected_links: list of str or None
    :param separate: A boolean indicating whether the links should be separated according to articles.
    :type separate: bool

    :return: A list of links if the ``separate`` parameter is set to ``False``.
             Otherwise, the links are returned in a dictionary, where the keys are the page titles, and the values are the links in each page.
    :rtype: list or dict

    :raises ValueError: When the level is not positive.
    :raises ValueError: When the level is not an integer.
    """

    if level <= 0:
        raise ValueError(f"The level number be a positive integer; received {level}")

    if type(level) is not int:
        raise ValueError(f"The level must be an integer; received {level}")

    titles = titles if type(titles) is list else [ titles ]

    collected_links = collected_links or [ ]

    """
    In the base case, collect links from the seed set.
    In the recursive case, fetch the links, and look for outgoing links in these results.
    Pages for whom links have already been collected are not collected anew.
    """
    titles = list(set(titles).difference(set(collected_links)))
    links = collect(titles, separate=separate, *args, **kwargs)
    if level <= 1:
        return links

    next_titles = [ link for link_set in links.values() for link in link_set ] if separate else links

    next_links = collect_recursive(next_titles, level=(level - 1),
                                   collected_links=list(set(titles + collected_links)),
                                   separate=separate, *args, **kwargs)

    if separate:
        links.update(next_links)
        return { page: list(set(links[page])) for page in links }
    else:
        return list(set(links + next_links))

def _get_intro_links(page):
    """
    Get a list of outgoing links from the introduction of the given page.

    :param page: The page from where to get the introduction.
                 The function expects a `revisions` key containing the text of the latest revision.
    :type page: dict

    :return: A list of outgoing links found in the given text.
    :rtype: list of str

    :raise ValueError: when no revisions are found in the page.
    :raise ValueError: when no text is found in the revision.
    """

    links = [ ]

    """
    Create the regular expressions that are used to clean the text and extract links.
    """
    reference_pattern = re.compile("<ref(.|\\n)*?((\/>)|(>(.|\\n)*?<\/ref>))")
    html_comments_pattern = re.compile("<!--(.|\n)+?-->")
    link_pattern = re.compile("\[\[(?!File|#)(.*?)(\|.*?)?\]\]")

    """
    Get the text from the latest revision.
    """
    revisions = page.get('revisions', [])
    if not revisions:
        raise ValueError("No revisions found")

    text = revisions[0].get('*', None)
    if not text:
        raise ValueError("No text found in the revision")

    """
    Remove all references and HTML comments from the text.
    Then, extract the actual links, removing the labels.
    """
    text = reference_pattern.sub("", text)
    text = html_comments_pattern.sub("", text)
    links = link_pattern.findall(text)
    links = [ link[0] for link in links ]
    return links

def _get_all_links(page):
    """
    Get a list of outgoing links from the given page.

    :param page: The page from where to get the text.
                 The function expects a `links` key containing the list of links in the article
    :type page: dict

    :return: A list of outgoing links found in the given text.
    :rtype: list of str
    """

    links = [ ]

    if 'links' not in page:
        return links

    links = [ link["title"] for link in page["links"] ]
    return links
