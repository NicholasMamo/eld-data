"""
The Wikipedia search resolver is similar to the :class:`~apd.resolvers.external.wikipedia_name_resolver.WikipediaNameResolver`.
However, instead of looking for similarly-named Wikipedia articles, the search resolver searches for candidate participants on Wikipedia.
The resolver tries to map candidates to one of the top-ranked results.

The aim of this resolver is to overcome common problems with the name resolver.
In many cases, colloquial names of candidates are not the same as their formal names.
For example, `FC Barcelona` is commonly referred to simply as `Barcelona`.

For candidate participants that could be resolved, the resolver returns the page name.
This acts as a link to the concept.
"""

import os
import re
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

import nltk

from vsm import vector_math
from vsm.clustering import Cluster
from nlp.document import Document
from nlp.tokenizer import Tokenizer
from wikinterface import info, links, search, text

from ..resolver import Resolver

class WikipediaSearchResolver(Resolver):
    """
    The Wikipedia search resolver looks for pages that include candidate names.
    The matches can be both in the title, but also in the content itself.

    The Wikipedia API automatically ranks articles by relevance.
    This resolver exploits the ranking to try and match the candidate with any of the top results.
    Out of these top results, the :class:`~apd.resolvers.external.wikipedia_search_resolver.WikipediaSearchResolver` chooses the most similar page.
    The resolver bases the choice on cosine similarity.

    Cosine similarity considers the domain, or the event's corpus.
    Apart from the corpus, the resolver also requires:

        - A :class:`~nlp.tokenizer.Tokenizer` to extract tokens, which then make up the documents,
        - A term-weighting scheme to create documents, and
        - A threshold above which candidate participants are resolved to a Wikipedia page.

    These are all instance variables and are required in the constructor.

    :ivar ~.scheme: The term-weighting scheme to use to create documents from Wikipedia pages.
                  These documents are used to compare the similarity with the domain of the candidates.
    :vartype ~.scheme: :class:`~nlp.weighting.TermWeightingScheme`
    :ivar threshold: The threshold below which candidates become unresolved.
    :vartype threshold: float.
    :ivar ~.tokenizer: The tokenizer to use to create documents.
    :vartype ~.tokenizer: :class:`~nlp.tokenizer.Tokenizer`
    :ivar threshold: The similarity threshold beyond which candidate participants are resolved.
    :vartype threshold: float
    :ivar domain: The event domain.
    :vartype domain: :class:`~nlp.document.Document`
    """

    def __init__(self, scheme, tokenizer, threshold, corpus):
        """
        Create the resolver.

        :param scheme: The term-weighting scheme to use to create documents from Wikipedia pages.
                       These documents are used to compare the similarity with the domain of the candidates.
        :type scheme: :class:`~nlp.weighting.TermWeightingScheme`
        :param threshold: The threshold below which candidates become unresolved.
        :type threshold: float.
        :param tokenizer: The tokenizer to use to create documents.
        :type tokenizer: :class:`~nlp.tokenizer.Tokenizer`
        :param threshold: The similarity threshold beyond which candidate participants are resolved.
        :type threshold: float
        :param corpus: The corpus of documents.
        :type corpus: list of :class:`~nlp.document.Document`
        """

        self.scheme = scheme
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.domain = Cluster(corpus).centroid
        self.domain.normalize()

    def resolve(self, candidates, *args, **kwargs):
        """
        Resolve the given candidates.
        They are sorted according to their score.

        :param candidates: The candidates to resolve.
        :type candidates: list

        :return: A tuple containing the resolved and unresolved candidates respectively.
        :rtype: tuple of lists
        """

        resolved_candidates, unresolved_candidates = [], []

        """
        Get the possible pages for each candidate.
        From each of these pages, remove the brackets because this information is secondary.
        If there are years outside the brackets, then the page can be excluded.
        Most often, pages with years in them are not entities.
        Unfortunately, exceptions exist, such as with the name `TSG 1899 Hoffenheim`.
        """
        candidates = sorted(candidates.keys(), key=lambda candidate: candidates.get(candidate), reverse=True)
        for candidate in candidates:
            """
            The page name is retained as-is when checking the year.
            If a page had brackets in it, they are retained.
            They are only removed temporarily to check if the non-bracket part has a year in it.
            In this way, the information about pages and their text can be collected.
            """
            pages = search.collect(candidate, limit=5)
            pages = [ page for page in pages if not self._has_year(self._remove_brackets(page)) ]

            """
            Fetch the page types.
            Disambiguation, list or missing pages are removed altogether.
            If any pages remain at this point, get their text and score the pages based on relevance to the corpus.
            """
            types = info.types(pages)
            pages = [ page for page, type in types.items() if type is info.ArticleType.NORMAL ]
            if len(pages):
                articles = text.collect(pages, introduction_only=True)
                candidate_document = Document(candidate, self.tokenizer.tokenize(candidate), scheme=self.scheme)

                """
                To calculate the score, bracketed text is removed since they do not convey important information.
                Tokens that are part of the candidate name are removed from the sentence.
                """
                scores = { }
                for page, introduction in articles.items():
                    introduction = self._remove_brackets(introduction)
                    sentence = self._get_first_sentence(introduction)
                    tokens = self.tokenizer.tokenize(sentence)
                    tokens = [ token for token in tokens if token not in candidate_document.dimensions ]
                    sentence_document = Document(introduction, tokens, scheme=self.scheme)

                    title_document = Document(page, self.tokenizer.tokenize(page), scheme=self.scheme)
                    scores[page] = self._compute_score(candidate_document, title_document, sentence_document)

                """
                Get the most relevant article.
                If it exceeds the threshold, then the candidate is resolved to that article.
                If it fails to exceed the threshold, the candidate is added to the unresolved candidates.
                """
                article, score = sorted(scores.items(), key=lambda score: score[1], reverse=True)[0]
                if score >= self.threshold and article not in resolved_candidates:
                    resolved_candidates.append(article)
                    continue

            unresolved_candidates.append(candidate)

        return (resolved_candidates, unresolved_candidates)

    def _has_year(self, title):
        """
        Check whether the given title has a year in it.

        :param title: The title of the article.
        :type title: str

        :return: A boolean indicating whether the title includes a year in it.
        :rtype: bool
        """

        year_pattern = re.compile("\\b[0-9]{4}\\b")
        return len(year_pattern.findall(title)) > 0

    def _remove_brackets(self, text):
        """
        Remove brackets from the given text.

        :param text: The text from which to remove brackets.
        :type text: str

        :return: The text without any components in the brackets.
        :rtype: str
        """

        bracket_pattern = re.compile("\(.*?\)")
        return bracket_pattern.sub(' ', text)

    def _get_first_sentence(self, text):
        """
        Get the first sentence from the given text.

        :param text: The text from which to extract the first sentence.
        :type text: str

        :return: The first sentence from the given text.
        :rtype: str
        """

        if text:
            sentences = nltk.sent_tokenize(text)
            return sentences[0]
        else:
            return text

    def _compute_score(self, candidate, title, sentence):
        """
        Compute the score of an article in terms of its relevance.
        The score is made up of two factors:

            #. The similarity between the article name and the candidate;
            #. The similarity between the first sentence of the article and the domain.

        These two factors are multipled together to get the score.
        The score is bound between 0 and 1.

        :param candidate: The candidate name.
        :type candidate: `nlp.document.Document`
        :param title: The title of the article.
        :type title: `nlp.document.Document`
        :param sentence: The first sentence of the article.
        :type sentence: `nlp.document.Document`

        :return: The relevance score of the article.
        :rtype: float
        """

        candidate.normalize()
        title.normalize()
        sentence.normalize()

        title_score = vector_math.cosine(title, candidate)
        text_score = vector_math.cosine(sentence, self.domain)
        return title_score * text_score
