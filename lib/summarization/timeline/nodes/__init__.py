"""
Nodes are the basic component of a :class:`~summarization.timeline.Timeline`.
Each :class:`~summarization.timeline.nodes.Node` stores information that can be summarized later on.

The most crucial components of the :class:`~summarization.timeline.nodes.Node` class are the :func:`~summarization.timeline.nodes.Node.add` and :func:`~summarization.timeline.nodes.Node.similarity` functions.
These two functions are facility functions for the :class:`~summarization.timeline.Timeline` to decide where to add information.
Both functions accept the same input, regardless if they do not use all of it.
"""

from abc import ABC, abstractmethod
import os
import sys
import time

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from objects.attributable import Attributable
from objects.exportable import Exportable

class Node(Exportable, Attributable):
    """
    The :class:`~summarization.timeline.nodes.Node` base class is a general representation of the nodes that make up timelines.
    Any :class:`~summarization.timeline.nodes.Node` maintains, at least, the timestamp when it was created as its state.
    However, all classes that inherit the :class:`~summarization.timeline.nodes.Node` need to maintain their own information, whatever it may be.

    All inherited classes must also implement the following functionality:

    - :func:`~summarization.timeline.nodes.Node.add`: used by the :class:`~summarization.timeline.Timeline` to add information to a node.
    - :func:`~summarization.timeline.nodes.Node.similarity`: used by the :class:`~summarization.timeline.Timeline` to assess whether the new information is relevant to a node.
    - :func:`~summarization.timeline.nodes.Node.get_all_documents`: used when summarizing to get all the :class:`~nlp.document.Document` instances stored in the node.
    - :func:`~summarization.timeline.nodes.Node.merge`: used to combine several nodes (of the same type) into one node.

    .. note::

        The inputs for the :func:`~summrization.timeline.nodes.node.Node.add` and :func:`~summrization.timeline.nodes.node.Node.similarity` function should be the same.
        This is necessary even if an input is used in :func:`~summrization.timeline.nodes.node.Node.add`, but not in :func:`~summrization.timeline.nodes.node.Node.similarity`, for example.
        This is because the :func:`~summarization.timeline.Timeline.add` function passes all arguments and keyword arguments to both functions in the same way.

    :ivar created_at: The timestamp when the node was created.
    :vartype created_at: float
    """

    def __init__(self, created_at):
        """
        Create the node with the given timestamp.

        :param created_at: The timestamp when the node was created.
        :type created_at: float
        """

        super(Node, self).__init__()
        self.created_at = created_at

    @abstractmethod
    def add(self, *args, **kwargs):
        """
        Add information to the node.
        The information is passed on as arguments and keyword arguments.
        """

        pass

    @abstractmethod
    def similarity(self, *args, **kwargs):
        """
        Compute the similarity between this node and the given information.
        The information is passed on as arguments and keyword arguments.

        :return: The similarity between this node and the given information.
        :rtype: float
        """

        pass

    @abstractmethod
    def get_all_documents(self, *args, **kwargs):
        """
        Get all the :class:`~nlp.document.Document` instances in this node.

        The implementation differs according to how the node stores information.
        However, they must all have functionality to return a list of :class:`~nlp.document.Document` instances.

        :return: A list of documents in the node.
        :rtype: list of :class:`~nlp.document.Document`
        """

        pass

    def expired(self, expiry, timestamp):
        """
        Check whether the node has expired.
        In reality, this function checks whether a number of seconds, equivalent to ``expiry``, have passed since the node was created.

        :param expiry: The lifetime of a node before it is said to expire.
                       It is measured in seconds.
        :type expiry: float
        :param timestamp: The current timestamp.
        :type timestamp: float

        :raises ValueError: When the expiry is negative.
        """

        if expiry < 0:
            raise ValueError(f"The expiry cannot be negative: received {expiry}")

        return timestamp - self.created_at >= expiry

    @staticmethod
    @abstractmethod
    def merge(created_at, *args):
        """
        Create a new :class:`~summarization.timeline.nodes.Node` by combining the data in all of the given nodes.
        The new :class:`~summarization.timeline.nodes.Node` will have the given timestamp, but it will inherit all of the data in the nodes.

        All the nodes are provided as additional arguments (using ``*args``).
        If none are given, this function creates an empty :class:`~summarization.timeline.nodes.Node`.

        :param created_at: The timestamp when the node was created.
        :type created_at: float

        :return: A new node with all of the data stored in the given nodes.
        :rtype: :class:`~summarization.timeline.nodes.Node`
        """

        pass

from .cluster_node import ClusterNode
from .document_node import DocumentNode
from .topical_cluster_node import TopicalClusterNode
