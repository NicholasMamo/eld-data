"""
Resolution is the fourth step of the APD process.
This step assigns maps a candidate to an alternate, preferably more semantic, representation.
For example, the :class:`~apd.resolvers.external.wikipedia_name_resolver.WikipediaNameResolver` resolves candidates to Wikipedia concepts.

Resolution does not have to map every candidate participant to an alternate representation.
Resolvers may have failing conditions.
If a candidate participant cannot be resolved, then it is usually rejected.
Those that can be resolved are accepted as valid participants.
Both resolved and unresolved candidates are returned as a tuple.

The functionality revolves around the :class:`~apd.resolvers.resolver.Resolver`'s :func:`~apd.resolvers.resolver.Resolver.resolve` method.
"""

class Resolver(object):
    """
    The simplest resolver returns all candidates as resolved participants.
    This can be thought of as a mapping from a candidate to the same candidate.

    However, the resolver, as a class, represents the kind of functionality that all sub-classes should be able to offer.
    This functionality is namely the ability to resolve candidates.

    The functionality revolves around one the :func:`~apd.resolvers.resolver.Resolver.resolve` method
    The input candidates should be the product of a :class:`~apd.scorers.scorer.Scorer` process.
    In other words, they should be a dictionary, with the keys being the candidates and the values being the score.
    """

    def resolve(self, candidates, *args, **kwargs):
        """
        The resolution function returns the same candidates as they were given, but as a list.
        They are sorted according to their score.

        :param candidates: The candidates to resolve.
                           The candidates should be in the form of a dictionary.
                           The keys should be the candidates, and the values the scores.
        :type candidates: dict

        :return: A tuple containing the resolved and unresolved candidates respectively.
                 The base resolver resolves all candidates to the same candidates.
                 Therefore unresolved candidates are empty.
        :rtype: tuple of lists
        """

        return (list(sorted(candidates.keys(), key=lambda candidate: candidates.get(candidate), reverse=True)), [ ])
