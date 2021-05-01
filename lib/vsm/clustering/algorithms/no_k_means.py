"""
The No-K-Means algorithm is a one-pass incremental clustering algorithm.
Breaking that down, it means that the algorithm:

1. Needs to read vectors only once to assign them to a cluster, and
2. Adds vectors to clusters, but does not remove them.

The algorithm compares incoming vectors to existing clusters.
The technique adds the vector to the most similar cluster if the similarity exceeds a certain threshold.
Otherwise, it creates a new cluster with the vector.

Therefore the complexity of the algorithm is :math:`O(nm)`, where :math:`n` is the number of vectors and :math:`m` is the number of clusters.
Since incremental approaches are often used in online settings, where timeliness is just as important as performance, the algorithm also has a freeze period.

The algorithm retires clusters that have not been updated after seeing a number of vectors, equivalent to the freeze period.
That means that if the freeze period is :math:`f`, and the cluster does not receive any vectors from any of the last :math:`f` vectors, the algorithm freezes it.
Frozen clusters cannot receive new vectors, and thus the algorithm can skip many checks, improving the efficiency.

.. note::

    This implementation is based on the algorithm outlined in `this paper by Azzopardi et al. in 2016 <https://www.researchgate.net/profile/Colin_Layfield/publication/303893387_Extended_No-K-Means_for_Search_Results_Clustering/links/575acd4208ae9a9c95518dfd.pdf>`_.
    The concept of one-pass incremental algorithms is very general, however, so you will find approaches like this one elsewhere.
"""

import os
import sys

path = os.path.join(os.path.dirname(__file__), '..')
if path not in sys.path:
    sys.path.append(path)

from algorithms import ClusteringAlgorithm
from cluster import Cluster

class NoKMeans(ClusteringAlgorithm):
    """
    The No-K-Means algorithm class maintains two variables in its state:

    1. The clusters that are active, or which have not been frozen yet; and
    2. The frozen clusters, which may not receive new vectors.

    .. warning::

        By default, the algorithm does not store frozen clusters because they hog memory over time.
        You can specify that the algorithm should store frozen clusters by initializing the algorithm with the ``store_frozen`` parameter set to ``True``.
        Normally, algorithms (such as the :class:`~queues.consumers.eld_consumer.ELDConsumer`) can retain the clusters that they need themselves, and the algorithm removes the unused, frozen clusters.

    :ivar frozen_clusters: A list of clusters that have been retired.
    :vartype frozen_clusters: list of :class:`~vsm.clustering.cluster.Cluster` instances
    :ivar threshold: The minimum similarity between a vector and a cluster's centroid.
                     If any cluster exceeds this threshold, the vector is added to that cluster.
    :vartype threshold: float
    :ivar freeze_period: The number of vectors that arrive without being added to a cluster before it is frozen.
    :vartype freeze_period: int
    :ivar store_frozen: A boolean indicating whether frozen clusters should be retained.
    :vartype store_frozen: bool
    """

    def __init__(self, threshold, freeze_period, store_frozen=False):
        """
        Initialize the algorithm with an empty set of active and frozen clusters.
        The No-K-Means takes in two algorithm-related parameters:

        1. The threshold, or the minimum similarity to add a vector to a cluster; and
        2. The freeze period, or how long a cluster can go without receiving new vectors before being frozen.

        By default, the algorithm does not retain clusters once they are frozen, but you can change this behavior by setting the ``store_frozen`` parameter to ``True``.

        :param threshold: The minimum similarity between a vector and a cluster's centroid.
                          If any cluster exceeds this threshold, the vector is added to that cluster.
        :type threshold: float
        :param freeze_period: The number of vectors that arrive without being added to a cluster before it is frozen.
        :type freeze_period: int
        :param store_frozen: A boolean indicating whether frozen clusters should be retained.
        :type store_frozen: bool
        """

        super(NoKMeans, self).__init__()
        self.frozen_clusters = [ ]

        self.threshold = threshold
        self.freeze_period = freeze_period
        self.store_frozen = store_frozen

    def cluster(self, vectors, *args, **kwargs):
        """
        Cluster the given vectors.
        This function uses the ``clusters`` property as a list of active clusters.
        It automatically freezes inactive clusters as it iterates through the vectors.

        Any additional arguments and keyword arguments are passed on to the :func:`~vsm.cluster.cluster.Cluster.similarity` function.

        .. note::

            This function only returns the clusters that received new vectors during the function call.
            To get a list of _all_ active clusters, use the ``clusters`` property.

        :param vectors: The list of vectors to cluster.
        :type vectors: list of :class:`~vsm.vector.Vector`

        :return: The clusters that received vectors during this function call.
        :rtpye: list of :class:`~vsm.clustering.cluster.Cluster` instances
        """

        updated_clusters = [ ]

        for vector in vectors:
            """
            Freeze inactive clusters first.
            In this way, nothing gets added to them, thereby resetting their age.
            """
            for cluster in self.clusters:
                self._update_age(cluster)
                if self._to_freeze(cluster):
                    self._freeze(cluster)

            """
            If there are active clusters, get the closest cluster.
            If the vector's similarity with the cluster exceeds the threshold, add the vector to the cluster.
            The cluster's age is resetted.
            """
            if self.clusters:
                cluster, similarity = self._closest_cluster(vector, *args, **kwargs)
                if similarity >= self.threshold:
                    cluster.vectors.append(vector)
                    self._reset_age(cluster)
                    updated_clusters.append(cluster)
                    continue

            """
            If there was no similar cluster, create a new cluster with just that vector.
            """
            cluster = Cluster([ vector ])
            self.clusters.append(cluster)
            updated_clusters.append(cluster)

        return list(set(updated_clusters))

    def _update_age(self, cluster, increment=1):
        """
        Update the age of the given cluster.

        :param cluster: The cluster whose age will be incremented.
        :type cluster: :class:`~vsm.clustering.cluster.Cluster`
        :param increment: The amount by which to increment the age.
        :type increment: int
        """

        cluster.attributes['age'] = cluster.attributes.get('age', 0) + increment

    def _to_freeze(self, cluster):
        """
        Check if the cluster's age is high enough that the cluster should be frozen.
        A cluster should be frozen if its age is longer than the freeze period.

        :param cluster: The cluster whose age will be incremented.
        :type cluster: :class:`~vsm.clustering.cluster.Cluster`

        :return: A boolean indicating whether the cluster should be frozen.
        :rtype: bool
        """

        return cluster.attributes.get('age') > self.freeze_period

    def _freeze(self, cluster):
        """
        Freeze the given cluster.
        This removes the cluster from the list of active clusters.
        Simultaneously, it is added to the list of frozen clusters.

        :param cluster: The cluster whose age will be incremented.
        :type cluster: :class:`~vsm.clustering.cluster.Cluster`

        :raises ValueError: When the cluster is not active.
        """

        self.clusters.remove(cluster)

        if self.store_frozen:
            self.frozen_clusters.append(cluster)

    def _closest_cluster(self, vector, *args, **kwargs):
        """
        Get the closest cluster to the vector and return the similarity score.
        This function only compares the vector with active clusters.
        Any additional arguments and keyword arguments are passed on to the :func:`~vsm.cluster.cluster.Cluster.similarity` function.

        :param vector: The vector to compare with all clusters.
        :type vector: :class:`~vsm.vector.Vector`

        :return: A tuple consisting of the closest cluster and the similarity between it and the vector.
                 If there are no active clusters, ``None`` is returned instead.
        :rtype: tuple or None
        """

        if not self.clusters:
            return None

        similarities = [ (cluster, cluster.similarity(vector, *args, **kwargs)) for cluster in self.clusters ]
        closest_cluster = sorted(similarities, key=lambda tuple: tuple[1], reverse=True)[0]
        return closest_cluster

    def _reset_age(self, cluster):
        """
        Reset the cluster age to zero.
        This function is called when an existing cluster receives a new vector.

        :param cluster: The cluster whose age will be incremented.
        :type cluster: :class:`~vsm.clustering.cluster.Cluster`
        """

        cluster.attributes['age'] = 0
