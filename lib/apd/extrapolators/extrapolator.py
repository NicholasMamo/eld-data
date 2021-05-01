"""
Extrapolation is the fifth step of the APD process.
This step is analogous to entity set expansion: it looks for similar participants to those that were resolved previously.

The functionality revolves around the :class:`~apd.extrapolators.extrapolator.Extrapolator`'s :func:`~apd.extrapolators.extrapolator.Extrapolator.extrapolate` method.
The main function returns the participants in descending order of their relevance to the event's corpus.
The participants should be ranked in descending order of their relevance.
"""

class Extrapolator(object):
    """
    The simplest extrapolator returns no new participants.
    However, it defines the functionality that is common to all other extrapolators.

    The functionality revolves around the :func:`~apd.extrapolators.extrapolator.Extrapolator.extrapolate` method.
    The input participants should be the product of a :class:`~apd.resolvers.resolver.Resolver` process: a list of strings representing participants.
    The output, then, is a list of new participants that the extrapolator considers valid for the event.
    """

    def extrapolate(self, participants, *args, **kwargs):
        """
        Extrapolate from the given participants.
        This extrapolator returns no new participants.

        :param participants: The participants found by the resolver.
        :type participants: list of str

        :return: The new participants identified as relevant by the extrapolator.
        :rtype: list of str
        """

        return [ ]
