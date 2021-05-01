"""
Timelines are not summaries, but they are closely associated.
The :class:`~summarization.timeline.Timeline` is the bridge between :ref:`TDT algorithms <tdt_algorithms>` and :ref:`summarization algorithms <summarization_algorithms>`.
In fact, :ref:`most consumers <consumers_algorithms>` use a :ref:`TDT algorithm <tdt_algorithms>` to produce a :class:`~summarization.timeline.Timeline` that can then be summarized by a :ref:`summarization algorithm <summarization_algorithms>`.

A timeline is structured as a list of :class:`~summarization.timeline.nodes.Node`.
Each :class:`~summarization.timeline.nodes.Node` stores information about what happened in that period of time.
The :class:`~summarization.timeline.Timeline` is usually managed by :ref:`a consumer <consumers_algorithms>`, which also adds :class:`~summarization.timeline.nodes.Node` instances to it.
Later, the nodes can be summarized with any :ref:`summarization algorithm <summarization_algorithms>`.
"""

import importlib
import os
import sys
import time

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from objects.exportable import Exportable

class Timeline(Exportable):
    """
    The :class:`~summarization.timeline.Timeline` is a rather simple class that stores a list of :class:`~summarization.timeline.nodes.Node` instances.
    In spite of its simplicity, the :class:`~summarization.timeline.Timeline` class hides a lot of functionality.

    Generally, only the constructor (:func:`~summarization.timeline.Timeline.__init__`) and the :func:`~summarization.timeline.Timeline.add` function are used.
    The latter is responsible for identifying whether there is a need to add a new :class:`~summarization.timeline.nodes.Node` or whether the latest information should go into the latest :class:`~summarization.timeline.nodes.Node`.
    The decision is based on parameters set when instantiating the :class:`~summarization.timeline.Timeline`:

    - The ``expiry`` parameter controls the time that must elapse before a :class:`~summarization.timeline.nodes.Node` is no longer active.
      Inactive nodes do not automatically absorb new information.
    - The ``min_similarity`` parameter is the minimum similarity between the new information and an old :class:`~summarization.timeline.nodes.Node` for the :class:`~summarization.timeline.nodes.Node` to absorb that information.
    - The ``max_time`` parameter is the time that must elapse before an old :class:`~summarization.timeline.nodes.Node` is definitively retired.
      Retired :class:`~summarization.timeline.nodes.Node` instances cannot change.

    At any time, the timeline only ever has one active :class:`~summarization.timeline.nodes.Node`.
    If there is not an active :class:`~summarization.timeline.nodes.Node` instances, the incoming information can either be absorbed by an expired :class:`~summarization.timeline.nodes.Node`, or go into a new :class:`~summarization.timeline.nodes.Node`.

    Note that what information means is vague.
    The :class:`~summarization.timeline.Timeline` always has one type of :class:`~summarization.timeline.nodes.Node`, and each :class:`~summarization.timeline.nodes.Node` accepts different kind of information.
    For example, the :class:`~summarization.timeline.nodes.document_node.DocumentNode` is made up of :class:`~nlp.document.Document` instances.
    The type of :class:`~summarization.timeline.nodes.Node` is set using the ``node_type`` parameter when instantiating the :class:`~summarization.timeline.Timeline`.
    This class automatically creates the appropriate :class:`~summarization.timeline.nodes.Node` for you.

    :ivar nodes: The list of :class:`~summarization.timeline.nodes.Node` instances in the timeline.
    :vartype nodes: :class:`~summarization.timeline.nodes.Node`
    :ivar node_type: The type of :class:`~summarization.timeline.nodes.Node` to create in the timeline.
    :vartype node_type: :class:`~summarization.timeline.nodes.Node`
    :ivar expiry: The time in seconds that it takes for a :class:`~summarization.timeline.nodes.Node` to expire.
                  Expired :class:`~summarization.timeline.nodes.Node` instances do not automatically absorb documents.
                  If the expiry is 0, new documents immediately join a new :class:`~summarization.timeline.nodes.Node` unless they are absorbed.
    :vartype expiry: float
    :ivar min_similarity: The minimum similarity between incoming documents and a :class:`~summarization.timeline.nodes.Node` to be absorbed by it.
                          This value is inclusive.
    :vartype min_similarity: float
    :ivar max_time: The maximum time in seconds to look back when deciding whether a :class:`~summarization.timeline.nodes.Node` should absorb a new topic.
                    The comparison is made with the :class:`~summarization.timeline.nodes.Node`'s `created_at` instance variable.
                    This value is inclusive.
    :vartype max_time: float
    """

    def __init__(self, node_type, expiry, min_similarity, max_time=600, nodes=None):
        """
        Create the timeline with an empty set of nodes.

        :param node_type: The type of :class:`~summarization.timeline.nodes.Node` to create in the timeline.
        :type node_type: :class:`~summarization.timeline.nodes.Node`
        :param expiry: The time in seconds that it takes for a :class:`~summarization.timeline.nodes.Node` to expire.
                       Expired :class:`~summarization.timeline.nodes.Node` instances do not automatically absorb documents.
                       If the expiry is 0, new documents immediately join a new :class:`~summarization.timeline.nodes.Node` unless they are absorbed.
        :type expiry: float
        :param min_similarity: The minimum similarity between incoming documents and a :class:`~summarization.timeline.nodes.Node` to be absorbed by it.
                              This value is inclusive.
        :type min_similarity: float
        :param max_time: The maximum time in seconds to look back when deciding whether a :class:`~summarization.timeline.nodes.Node` should absorb a new topic.
                         The comparison is made with the :class:`~summarization.timeline.nodes.Node`'s `created_at` instance variable.
                         This value is inclusive.
        :type max_time: float
        :param nodes: The initial list of :class:`~summarization.timeline.nodes.Node` instances in the timeline.
        :type nodes: list of :class:`~summarization.timeline.nodes.Node`

        :raises ValueError: When the expiry is negative.
        :raises ValueError: When the minimum similarity is not between 0 and 1.
        """

        """
        Validate the parameters.
        """

        if expiry < 0:
            raise ValueError(f"The node expiry cannot be negative: received {expiry}")

        if not 0 <= min_similarity <= 1:
            raise ValueError(f"The minimum similarity must be between 0 and 1: received {min_similarity}")

        self.nodes = nodes or [ ]
        self.node_type = node_type
        self.expiry = expiry
        self.min_similarity = min_similarity
        self.max_time = max_time

    def add(self, timestamp=None, *args, **kwargs):
        """
        Add information to a :class:`~summarization.timeline.nodes.Node` on the timeline.
        This function does not necessarily add a new :class:`~summarization.timeline.nodes.Node`.
        Instead, there are three cases:

        1. First, the timeline tries to find a :class:`~summarization.timeline.nodes.Node` that the timeline created very recently.
           If there is an active :class:`~summarization.timeline.nodes.Node`, add the information to it.
        2. If there isn't, it goes over the :class:`~summarization.timeline.nodes.Node` instances backwards to find one that is very similar to the information.
           This function only goes as far back as the ``max_time`` allows.
           If there is one, add the information to it.
        3. If all else fails, create a new :class:`~summarization.timeline.nodes.Node` with the given information.

        All arguments and keyword arguments are passed on to the :class:`~summarization.timeline.nodes.Node`'s :func:`~summarization.timeline.nodes.Node.add` and :func:`~summarization.timeline.nodes.Node.similarity` methods.
        That means the input to this function depends on the type of :class:`~summarization.timeline.nodes.Node` that the timeline accepts.
        Only the ``timestamp`` parameter belongs to this function.
        If no ``timestamp`` is given, the current time is used instead.

        :param timestamp: The current timestamp.
                          If the timestamp is not given, the current time is used.
        :type timestamp: float
        """

        timestamp = time.time() if timestamp is None else timestamp

        """
        If there are nodes and the latest one is still active—it hasn't expired—add the documents to it.
        """
        if self.nodes and not self.nodes[-1].expired(self.expiry, timestamp):
            self.nodes[-1].add(*args, **kwargs)
            return

        """
        Go through the nodes backwards and see if any node absorbs the documents.
        """
        for node in self.nodes[::-1]:
            if timestamp - node.created_at <= self.max_time and node.similarity(*args, **kwargs) >= self.min_similarity:
                node.add(*args, **kwargs)
                return

        """
        If no node absorbs the documents, create a new node and add them to it.
        """
        node = self._create(created_at=timestamp)
        node.add(*args, **kwargs)
        self.nodes.append(node)

    def _create(self, created_at, *args, **kwargs):
        """
        Create a new :class:`~summarization.timeline.nodes.Node` on the timeline.
        Any arguments and keyword arguments are passed on to the :func:`~summarization.timeline.nodes.Node.__init__` method.

        :param created_at: The timestamp when the node was created.
        :type created_at: float

        :return: The created node.
        :rtype: :class:`~summarization.timeline.nodes.Node`
        """

        return self.node_type(created_at=created_at, *args, **kwargs)

    def to_array(self):
        """
        Export the :class:`~summarization.timeline.Timeline` as an associative array.

        :return: The :class:`~summarization.timeline.Timeline` as an associative array.
        :rtype: dict
        """

        return {
            'class': str(Timeline),
            'node_type': str(self.node_type),
            'expiry': self.expiry,
            'min_similarity': self.min_similarity,
            'nodes': [ node.to_array() for node in self.nodes ],
        }

    @staticmethod
    def from_array(array):
        """
        Create a :class:`~summarization.timeline.Timeline` instance from the given associative array.

        :param array: The associative array with the attributes to create the :class:`~summarization.timeline.Timeline`.
        :type array: dict

        :return: A new instance of the timeline with the same attributes stored in the object.
        :rtype: :class:`~summarization.timeline.nodes_node.ClusterNode`
        """

        nodes = [ ]
        for node in array.get('nodes'):
            module = importlib.import_module(Exportable.get_module(node.get('class')))
            cls = getattr(module, Exportable.get_class(node.get('class')))
            nodes.append(cls.from_array(node))

        module = importlib.import_module(Exportable.get_module(array.get('node_type')))
        node_type = getattr(module, Exportable.get_class(array.get('node_type')))

        return Timeline(node_type=node_type, expiry=array.get('expiry'),
                        min_similarity=array.get('min_similarity'), nodes=nodes)
