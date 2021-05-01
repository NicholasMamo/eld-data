"""
The Document Graph Summarizer (DGS) algorithm constructs a :class:`~summarization.summary.Summary` using a graph-based approach.
The algorithm constructs a graph between :class:`~nlp.document.Document` instances and tries to find the different subjects that they discuss.
In this way, the algorithm can maximize precision by minimizing redundancy.

The general approach can be summarized as follows:

1. Construct a document graph.

2. Split it into communities using the Girvan-Newman algorithm.

3. Iterate over the communities, in descending order of size.
   Choose the document that maximizes the product of:

   3.1. Similarity to the query,

   3.2. Centrality, and

   3.3. A brevity score based on BLEU.

   If the document is not too long, add it to the summary.

In this way, the algorithm favors the most common facets of discussion.
Apart from that, since it only picks one :class:`~nlp.document.Document` from each community, the algorithm avoids including repeated ideas in the final :class:`~summarization.summary.Summary`.

The main bottleneck of the DGS algorithm is the document graph.
Constructing it necessitates calculating the pairwise similarity between all :class:`~nlp.document.Document` instances.
Therefore it is recommended to choose a few, quality candidates to summarize, rather than summarize a large corpus.

.. note::

    This implementation is based on the algorithm presented in `ELD: Event TimeLine Detection—A Participant-Based Approach to Tracking Events by Mamo et al. (2019) <https://dl.acm.org/doi/abs/10.1145/3342220.3344921>`_.
"""

import math
import os
import sys

import networkx as nx
from networkx import edge_betweenness_centrality
from networkx.algorithms import centrality, community

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp.tokenizer import Tokenizer

from summarization import Summary
from summarization.algorithms import SummarizationAlgorithm

from vsm import vector_math
from vsm.clustering import Cluster

class DGS(SummarizationAlgorithm):
    """
    The Document Graph Summarizer (DGS) is an algorithm that minimizes redundancy by splitting documents into communities.
    The algorithm receives documents and builds a summary from the largest communities to capture all facets.

    The DGS is parameter-less, and therefore maintains no state except for a :class:`~nlp.tokenizer.Tokenizer`.
    This :class:`~nlp.tokenizer.Tokenizer` is used to calculate the brevity score when selecting which :class:`~nlp.document.Document` to add to the :class:`~summarization.summary.Summary`.

    :ivar ~.tokenizer: The tokenizer used to calculate the brevity score.
    :vartype ~.tokenizer: :class:`~nlp.tokenizer.Tokenizer`
    """

    def __init__(self):
        """
        Create the DGS summarization algorithm with a tokenizer.
        """

        self.tokenizer = Tokenizer(min_length=1)

    def summarize(self, documents, length, query=None, *args, **kwargs):
        """
        Summarize the given documents.

        This function constructs the document graph using the given list of :class:`~nlp.document.Document`.
        Then, the method splits the graph into communities, representing facets of discussion, using the Girvan-Newman (:func:`networkx.algorithms.community.centrality.girvan_newman`) algorithm.
        From each community, this function chooses the one document that maximize similarity with the query, centrality and length.

        :param documents: The list of documents to summarize.
        :type documents: list of :class:`~nlp.document.Document`
        :param length: The maximum length of the summary in characters.
        :type length: float
        :param query: The query around which to build the summary.
                      If no query is given, the summary is built around the centroid of documents.
        :type query: :class:`~vsm.vector.Vector` or None

        :return: The summary of the documents.
        :rtype: :class:`~summarization.summary.Summary`

        :raises ValueError: When the summary length is not positive.
        """

        summary = Summary()

        """
        Validate the inputs.
        """
        if length <= 0:
            raise ValueError(f"Invalid summary length {length}")

        """
        Compute the query if need be.
        """
        query = query or self._compute_query(documents)

        """
        Convert the documents into a graph.
        From this graph extract all communities.
        """
        graph = self._to_graph(documents)
        communities = self._extract_communities(graph)

        """
        Repeatedly go over the communities.
        Larger communities are preferred to reflect popular facets.
        From each community, the top-scoring, valid document is added to the summary.
        """
        while communities:
            subset = self._largest_communities(communities)
            scores = self._score_documents(graph, subset, query)

            """
            Visit the largest communities first.
            Filter out documents that are too long to be in the summary.
            """
            for community, scores in zip(subset, scores):
                """
                Only one document can be chosen from each community.
                Therefore remove the community immediately.
                """
                communities.remove(community)
                documents = self._filter_documents(community, summary, length - len(str(summary)))
                scores = { document: score for document, score in scores.items()
                           if document in documents }

                """
                If no document is valid, move on to the next community.
                Otherwise, add the highest-scoring document to the summary.
                """
                if not documents:
                    continue
                else:
                    summary.documents.append(max(scores, key=scores.get))

        return summary

    def _compute_query(self, documents):
        """
        Create the query from the given documents.
        The query is equivalent to the centroid of the documents.

        :param documents: The list of documents to summarize.
        :type documents: list of :class:`~nlp.document.Document`

        :return: The centroid of the documents.
        :rtype: `~vsm.vector.Vector`
        """

        return Cluster(vectors=documents).centroid

    def _to_graph(self, documents):
        """
        Convert the given documents to a networkx graph.
        The documents are converted to nodes.
        Weighted edges between them are created if their similarity exceeds 0.

        .. note::

            The weight of edges is `1 - similarity`.
            The higher the similarity, the less weight.
            Therefore more paths go through that edge.

        :param documents: The list of documents to convert into a graph.
        :type documents: list of :class:`~nlp.document.Document`

        :return: A graph with nodes representind documents and weighted edges between them.
        :rtype: :class:`~networkx.Graph`
        """

        graph = nx.Graph()

        """
        First add the nodes to the graph.
        """
        for document in documents:
            graph.add_node(document, document=document)

        """
        Add the weighted edges between the documents.
        """
        for i, source in enumerate(documents):
            for target in documents[:i]:
                similarity = vector_math.cosine(source, target)
                if similarity > 0:
                    graph.add_edge(source, target, weight=(1 - similarity))

        return graph

    def _extract_communities(self, graph):
        """
        Extract the communities from the given graph.

        :param graph: The document graph.
        :type graph: :class:`~networkx.Graph`

        :return: List of parititions.
                 Each partition is a set of nodes.
        :rtype: list of set
        """

        """
        If the graph is already split into enough communities, do not split it further.
        """
        connected = list(nx.connected_components(graph))
        if len(connected) >= math.sqrt(len(graph.nodes)):
            return connected

        communities = community.girvan_newman(graph, most_valuable_edge=self._most_central_edge)
        partitions = list(next(communities))
        while len(partitions) < math.sqrt(len(graph.nodes)):
            partitions = list(next(communities))

        return partitions

    def _most_central_edge(self, graph):
        """
        Find the most central edge in the given graph.
        The algorithm uses NetworkX's betweenness centrality, but it is based on weight.
        The lower the weight, the more shortest paths could go through it.

        :param graph: The graph on which the algorithm operates.
        :type graph: :class:`~networkx.Graph`

        :return: The most central edge, made up of the source and edge nodes.
        :rtype: tuple
        """

        centrality = edge_betweenness_centrality(graph, weight='weight')
        edge = max(centrality, key=centrality.get)
        return edge

    def _largest_communities(self, communities):
        """
        Extract the largest communities.

        :param communities: The list of partitions.
                            Each partition is a set of nodes.
        :type communities: list of set

        :return: A list of the largest communities.
                 Each community is a set of nodes.
        :rtype: list of set
        """

        if communities:
            size = max(len(community) for community in communities)
            return list(filter(lambda community: len(community) == size, communities))

        return [ ]

    def _score_documents(self, graph, communities, query):
        """
        Score the documents in the given communities.
        The score is based on centrality and query similarity.

        :param graph: The graph in which the communities were found.
        :type graph: :class:`~networkx.Graph`
        :param communities: The list of partitions.
                            Each partition is a set of nodes.
        :type communities: list of set
        :param query: The query to which to compare the documents.
        :type query: :class:`vsm.vector.Vector`

        :return: A list of documents scored and separated by their community.
                 The outer list corresponds to the communities.
                 Internally, the documents are a dictionary.
                 The keys are the documents and the values their scores.
        :rtype: list of dict
        """

        scores = []

        """
        Go through each community and represent it as a subgraph.
        Calculate the eigenvector centrality for each node in the subgraph.
        Then, calculate the similarity between each node, or document, and the query.
        """
        for community in communities:
            subgraph = graph.subgraph(community)
            centrality_scores = centrality.eigenvector_centrality(subgraph)
            brevity_scores = { document: self._brevity_score(document.text)
                                 for document in subgraph.nodes }
            relevance = { document: vector_math.cosine(document, query)
                          for document in subgraph.nodes }
            scores.append({ document: brevity_scores[document] * centrality_scores[document] * relevance[document]
                             for document in subgraph.nodes })

        return scores

    def _brevity_score(self, text, r=10, *args, **kwargs):
        """
        Calculate the brevity score, bounded between 0 and 1.
        This score is based on `BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni et al. (2002) <https://dl.acm.org/doi/10.3115/1073083.1073135>`:

        .. math::

            score = max(1, e^{1 - \\frac{r}{c}})

        where :math:`c` is the number of tokens in the text, and :math:`r` is the ideal number of tokens.
        The score is based on the default tokenizezr.

        The score is 1 even when the tweet is longer than the desired length.
        In this way, the brevity score is more akin to a brevity penalty.

        :param text: The text to score.
                     The text is tokanized by the function.
        :type text: str
        :param r: The ideal number of tokens in the text.
        :type r: str

        :return: The brevity score, bounded between 0 and 1.
        :rtype: float
        """

        """
        The tokens are extracted using the same method as in the consumer.
        """
        tokens = self.tokenizer.tokenize(text)

        """
        If the text has no tokens, then the score is 0.
        """
        if not len(tokens):
            return 0

        """
        If there are tokens in the text, the score is calculated using the formula.
        If there are more tokens than the desired length, the score is capped at 1.
        """
        return min(math.exp(1 - r/len(tokens)), 1)

    def _filter_documents(self, documents, summary, length):
        """
        Get the documents that can be added to the summary.
        These include:

            #. Documents that are not already in the summary;

            #. Documents that are shorter than the length.

        :param documents: The list of available documents.
        :type documents: list of :class:`~nlp.document.Document`
        :param summary: The summary constructed so far.
        :type summary: :class:`~summarization.summary.Summary`
        :param length: The maximum length of the document.
                       The length is inclusive.
        :type length: float

        :return: A list of documents that can be added to the summary.
        :rtype: list of :class:`~nlp.document.Document`
        """

        documents = set(documents).difference(set(summary.documents))
        documents = [ document for document in documents if len(str(document)) <= length ]
        return documents
