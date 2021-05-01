"""
Post-processing is the sixth and last step in the APD process.
This step modifies the final participants to make them more useful to how they will be used.

The functionality revolves around one method: the :func:`~apd.postprocessors.postprocessor.Postprocessor.postprocess` method.
This function returns all participants, but with some changes made to them.
"""

class Postprocessor(object):
    """
    The simplest post-processor returns the participants without any changes.
    All other post-processors may add functionality to change participants.
    However, all post-processors must return all participants and in the same order.
    """

    def postprocess(self, participants, *args, **kwargs):
        """
        Return the same participants as those received.

        :param participants: The participants to postprocess.
        :type participants: list of str

        :return: The postprocessed participants.
        :rtype: list of str
        """

        return participants
