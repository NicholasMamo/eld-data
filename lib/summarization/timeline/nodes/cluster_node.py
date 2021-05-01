"""
The :class:`~summarization.timeline.nodes.cluster_node.ClusterNode` stores :class:`~vsm.clustering.cluster.Cluster` instances made up of :class:`~nlp.document.Document` instances instead of :class:`~nlp.document.Document` instances.
This affords the node more flexibility when computing :func:`~summarization.timeline.nodes.cluster_node.ClusterNode.similarity`.

Each :class:`~vsm.clustering.cluster.Cluster` conceptually represents a topic.
When a new :class:`~vsm.clustering.cluster.Cluster` arrives, the :class:`~summarization.timeline.nodes.cluster_node.ClusterNode` matches it separately with these previous 'topics'.
This is different from the :class:`~summarization.timeline.nodes.document_node.DocumentNode`, which compares new :class:`~nlp.document.Document` instances with all of the information in the node.
Therefore the :class:`~summarization.timeline.nodes.cluster_node.ClusterNode` can overcome fragmentation.
"""

import importlib
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from objects.exportable import Exportable
from summarization.timeline.nodes import Node
from vsm import vector_math
from vsm.clustering.cluster import Cluster

class ClusterNode(Node):
    """
    The :class:`~summarization.timeline.nodes.cluster_node.ClusterNode` stores :class:`~nlp.document.Document` instances in :class:`~vsm.clustering.cluster.Cluster` instances.
    When comparing a new :class:`~vsm.clustering.cluster.Cluster`, the :func:`~summarization.timeline.nodes.cluster_node.ClusterNode.similarity` function compares it separately with each :class:`~vsm.clustering.cluster.Cluster` in the node.

    :ivar clusters: The list of clusters in this node.
    :vartype clusters: list of :class:`~vsm.clustering.cluster.Cluster`
    """

    def __init__(self, created_at, clusters=None):
        """
        Create the node with an optional initial list of :class:`~vsm.clustering.cluster.Cluster` instances.

        :param created_at: The timestamp when the node was created.
        :type created_at: float
        :param clusters: The initial list of :class:`~vsm.clustering.cluster.Cluster` instances in this node.
        :type clusters: None or list of :class:`~vsm.clustering.cluster.Cluster`
        """

        super(ClusterNode, self).__init__(created_at)
        self.clusters = clusters or [ ]

    def add(self, cluster, *args, **kwargs):
        """
        Add a new :class:`~vsm.clustering.cluster.Cluster` to the node.

        :param cluster: The :class:`~vsm.clustering.cluster.Cluster` to add to the node.
        :type cluster: :class:`~vsm.clustering.cluster.Cluster`
        """

        self.clusters.append(cluster)

    def get_all_documents(self, *args, **kwargs):
        """
        Get all the :class:`~nlp.document.Document` instances in this node.
        This function aggregates all of the :class:`~nlp.document.Document` instances in its :class:`~vsm.clustering.cluster.Cluster` instances.

        :return: A list of :class:`~nlp.document.Document` instances in the node.
        :rtype: list of :class:`~nlp.document.Document`
        """

        return [ document for cluster in self.clusters for document in cluster.vectors ]

    def similarity(self, cluster, *args, **kwargs):
        """
        Compute the similarity between this node and the given :class:`~vsm.clustering.cluster.Cluster`.
        Since :class:`~vsm.clustering.cluster.Cluster` instances represent topics, this function tries to match the new :class:`~vsm.clustering.cluster.Cluster` with any :class:`~vsm.clustering.cluster.Cluster` already in the node.

        The similarity measure is :func:`~vsm.vector_math.cosine` and always compares the :class:`~vsm.clustering.cluster.Cluster` instances' centroids with each other.
        The returned similarity is the highest pairwise similarity.

        :param cluster: The cluster with which to compute similarity.
        :type cluster: :class:`~vsm.clustering.cluster.Cluster`

        :return: The similarity between this node and the given :class:`~vsm.clustering.cluster.Cluster`.
        :rtype: float
        """

        if self.clusters:
            return max(vector_math.cosine(cluster.centroid, other.centroid) for other in self.clusters)

        return 0

    def to_array(self):
        """
        Export the cluster node as an associative array.

        :return: The cluster node as an associative array.
        :rtype: dict
        """

        return {
            'class': str(ClusterNode),
            'created_at': self.created_at,
            'clusters': [ cluster.to_array() for cluster in self.clusters ],
        }

    @staticmethod
    def from_array(array):
        """
        Create an instance of the :class:`~summarization.timeline.nodes.cluster_node.ClusterNode` from the given associative array.

        :param array: The associative array with the attributes to create the :class:`~summarization.timeline.nodes.cluster_node.ClusterNode`.
        :type array: dict

        :return: A new instance of the :class:`~summarization.timeline.nodes.cluster_node.ClusterNode` with the same attributes stored in the object.
        :rtype: :class:`~summarization.timeline.nodes.cluster_node.ClusterNode`
        """

        clusters = [ ]
        for cluster in array.get('clusters'):
            module = importlib.import_module(Exportable.get_module(cluster.get('class')))
            cls = getattr(module, Exportable.get_class(cluster.get('class')))
            clusters.append(cls.from_array(cluster))

        return ClusterNode(created_at=array.get('created_at'), clusters=clusters)

    @staticmethod
    def merge(created_at, *args):
        """
        Create a new :class:`~summarization.timeline.nodes.cluster_node.ClusterNode` by combining the documents in all of the given nodes.
        The new :class:`~summarization.timeline.nodes.cluster_node.ClusterNode` will have the given timestamp, but it will inherit all of the documents in the nodes.

        All the nodes are provided as additional arguments (using ``*args``).
        If none are given, this function creates an empty :class:`~summarization.timeline.nodes.cluster_node.ClusterNode`.

        :param created_at: The timestamp when the node was created.
        :type created_at: float

        :return: A new node with all of the data stored in the given nodes.
        :rtype: :class:`~summarization.timeline.nodes.cluster_node.ClusterNode`
        """

        node = ClusterNode(created_at)

        for _node in args:
            for cluster in _node.clusters:
                node.add(cluster)

        return node
