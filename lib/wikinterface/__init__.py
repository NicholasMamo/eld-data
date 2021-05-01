"""
Wikinterface is a simple side-project bundled with EvenTDT that makes it easier to access Wikipedia.
The basic Wikipedia API interface uses the `Wikimedia API <https://www.mediawiki.org/wiki/API:Main_page>`_ to connect with Wikipedia.

The package separates the functions across modules that fetch different information from Wikipedia.
At the package-level are functions that are useful to several modules.

In all cases, responses are sent on the ``API_ENDPOINT`` and returned as a JSON string.

Under the hood, this package helps you make more efficient use of Wikipedia.
This includes functions like :func:`~wikinterface.links.collect_recursive` that provide complex functionality.
In addition, the Wikinterface package makes it easier to navigate the API's limits.
For example, functions like :func:`~wikinterface.search.collect` lets you perform searches and retrieve more than 50 results.
"""

API_ENDPOINT = "https://en.wikipedia.org/w/api.php?"
"""
:var API_ENDPOINT: The Wikipedia API endpoint.
:vartype API_ENDPOINT: str
"""

def is_error_response(response):
    """
    Validate whether the given Wikipedia response is an error.

    All error responses have an ``error`` key in them.
    This function simply checks whether this key is in the response.

    :param response: The response to validate.
    :type response: dict

    :return: A boolean indicating whether the response is an error.
    :rtype: bool
    """

    return 'error' in response

def construct_url(parameters=None):
    """
    Construct the URL using the given parameters.
    This function expects parameters to be a dictionary, with the parameter name as the key.

    This helper function automatically parses the parameter values.
    For example, if a parameter value is a boolean (and ``True``), only the parameter is added to the URL string.

    :param parameters: The list of GET parameters to send.
                       The parameter values can be either strings or booleans.
                       If a parameter is a boolean and `True`, it is added without a value.
                       If a parameter is a boolean and `False`, it is excluded altogether
    :type parameters: dict or None

    :return: The URL with any GET parameters provided.
    :rtype: str
    """

    url = f"{API_ENDPOINT}"

    """
    If parameters are given, they are filtered for `False` boolean values.
    Then, they are added as query parameters.
    """
    if parameters:
        parameters = {
            key: value for key, value in parameters.items() if (type(value) is not bool or value) and value is not None
        }

        parameter_strings = [ f"{key}" if type(value) is bool else f"{key}={value}" for key, value in parameters.items() ]
        url = url + '&'.join(parameter_strings)

    return url

def revert_redirects(results, redirects):
    """
    Wikipedia contains automatic redirections.
    For example, `Messi` redirects to `Lionel Messi`.
    This function automatically reverses that redirection.

    This action is useful because sometimes, the result is not what is expected.
    If you retrieve the text of the article about `Messi`, by default you would not find that key, but `Lionel Messi`.
    By reverting redirects, the keys are what you would expect.

    :param results: Any results obtained from Wikipedia.
                    The keys are the page titles, and the values are the returned information, such as the article text.
    :type results: dict
    :param redirects: The redirects provided by Wikipedia.
                      This dictionary has keys 'from' and 'to'.
    :type redirects: dict

    :return: A new dictionary with redirections.
    :rtype: dict
    """

    pages = dict(results)

    """
    Recreate the redirection representation.
    The redirects are represented as to-from instead of from-to.
    """
    targets = { redirect["to"]: redirect["from"] for redirect in redirects }

    for page in results:
        """
        If a page was redirected, 'redirect' it back.
        """
        if page in targets:
            pages[targets[page]] = pages[page]

    """
    In some cases, two pages may redirect to the same page.
    For example, `Striker (association football)` and `Inside forward` both point to `Forward (association football)`.
    Therefore if a redirection (the `from`) has no page, create a page for it.

    Note that when a query is split, the result could be empty for this missing page might not have been loaded yet.
    This happens when there are more than `x` results and Wikipedia returns only the first `x`.
    The redirection for a page that hasn't been loaded yet is given, but the content isn't there yet.
    Therefore this has to be checked in advance.
    """
    for page in redirects:
        if page["from"] not in pages and page["to"] in results:
            pages[page["from"]] = results[page["to"]]

    return pages
