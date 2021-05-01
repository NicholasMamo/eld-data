"""
Although you can create a :class:`~vsm.clustering.cluster.Cluster` yourself, it is more likely you will be receiving clusters from a clustering algorithm.
Clustering algorithms take in :class:`~vsm.vector.Vector` instances, or other instances derived from them, and cluster them automatically.

The :class:`~vsm.clustering.algorithms.ClusteringAlgorithm` exists so that all algorithms have a uniform interface.
For example, you can cluster :class:`~vsm.vector.Vector` instances using the :func:`~vsm.clustering.algorithms.ClusteringAlgorithm.cluster` method.
"""

from abc import ABC, abstractmethod

class ClusteringAlgorithm(ABC):
    """
    In EvenTDT, clustering algorithms maintain a state.
    That state contains, at least, a list of :class:`~vsm.clustering.cluster.Cluster` instances.
    This state is not always needed, although approaches like :class:`~vsm.clustering.algorithms.no_k_means.NoKMeans` do use it.
    If the algorithm does not need to use the state, it can store the latest generated clusters there.

    Aside from the state, all clustering algorithms must, at least, provide the :func:`~vsm.clustering.algorithms.ClusteringAlgorithm.cluster` functionality.
    This function receives a list of :class:`~vsm.vector.Vector` instances and clusters them.

    :ivar clusters: A list of clusters.
    :vartype clusters: list of :class:`~vsm.clustering.cluster.Cluster`
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the state of the clustering algorithm with an empty list of clusters.
        """

        self.clusters = [ ]

    @abstractmethod
    def cluster(self, vectors, *args, **kwargs):
        """
        Cluster the given vectors.
        The function returns the list of clusters produced so far, or a selection of it.

        :param vectors: The vectors to cluster.
        :type vectors: list of :class:`~vsm.vector.Vector`

        :return: The clusters in the algorithm state.
        :rtpye: list of :class:`~vsm.clustering.cluster.Cluster`
        """

        pass

from .no_k_means import NoKMeans
from .temporal_no_k_means import TemporalNoKMeans
