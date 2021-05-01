"""
The temporal No-K-Means is a variant of the :class:`~vsm.clustering.algorithms.no_k_means.NoKMeans` clustering algorithm.
The main change in the :class:`~vsm.clustering.algorithms.temporal_no_k_means.TemporalNoKMeans` is that the freeze period is not based on the number of observed vectors since one was added to a cluster.
Instead, it is based on the time that has elapsed since a cluster last received a vector.
In other words, it's possible that the algorithm freezes a cluster because a lot of time has passed since it was active, even if it has received no new vectors!
"""

import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from .no_k_means import NoKMeans
from vsm.clustering.cluster import Cluster

class TemporalNoKMeans(NoKMeans):
    """
    The temporal No-K-Means maintains the same state as the :class:`~vsm.clustering.algorithms.no_k_means.NoKMeans`.
    Differently from it, however, the temporal No-K-Means' freeze period is in terms of seconds, not in terms of vectors.

    :ivar threshold: The minimum similarity between a vector and a cluster's centroid.
                     If any cluster exceeds this threshold, the vector is added to that cluster.
    :vartype threshold: float
    :ivar freeze_period: The number of seconds of inactivity of a cluster before it is frozen.
    :vartype freeze_period: int
    :ivar store_frozen: A boolean indicating whether frozen clusters should be retained.
    :vartype store_frozen: bool
    """

    def __init__(self, threshold, freeze_period, store_frozen=False):
        """
        Initialize the algorithm with the clusters.

        :param threshold: The minimum similarity between a vector and a cluster's centroid.
                          If any cluster exceeds this threshold, the vector is added to that cluster.
        :type threshold: float
        :param freeze_period: The number of seconds of inactivity of a cluster before it is frozen.
        :type freeze_period: int
        :param store_frozen: A boolean indicating whether frozen clusters should be retained.
        :type store_frozen: bool
        """

        super(TemporalNoKMeans, self).__init__(threshold, freeze_period, store_frozen)

    def cluster(self, vectors, time="timestamp", *args, **kwargs):
        """
        Cluster the given vectors.
        To freeze clusters, this method needs to know the timestamp of vectors.
        By default, it looks for a ``timestamp`` attribute, but you can specify a different attribute.

        Any additional arguments and keyword arguments are passed on to the :class:`~vsm.clustering.cluster.Cluster`'s :func:`~vsm.clustering.cluster.Cluster.similarity` function.

        :param vectors: The list of vectors to cluster.
        :type vectors: list of :class:`~vsm.vector.Vector`
        :param time: The name of the vector attribute used to get the timestamp value.
                     The time value is expected to be a float or integer.
        :type time: str

        :return: The clusters that received vectors.
        :rtpye: list of :class:`~vsm.clustering.cluster.Cluster`
        """

        updated_clusters = [ ]

        """
        Vectors are clustered chronologically.
        """
        latest = -1
        vectors = sorted(vectors, key=lambda vector: vector.attributes.get(time))
        for vector in vectors:
            timestamp = vector.attributes.get(time)

            """
            Freeze inactive clusters first.
            In this way, nothing gets added to them, thereby resetting their age.
            Skip this step if the next vector is not newer than the previous vector.
            """
            if latest < timestamp:
                for cluster in self.clusters:
                    self._update_age(cluster, timestamp, time)
                    if self._to_freeze(cluster):
                        self._freeze(cluster)
                latest = timestamp

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

    def _update_age(self, cluster, timestamp, time):
        """
        Update the age of the given cluster.
        The age is calculated by subtracting from the current timestamp the time of the last vector in the cluster.
        It is assumed that the last vector in the cluster is the most recently-published.

        :param cluster: The cluster whose age will be incremented.
        :type cluster: :class:`~vsm.clustering.cluster.Cluster`
        :param timestamp: The current timestamp of the clustering algorithm.
                          This is equivalent to the last received vector.
        :type timestamp: int
        :param time: The name of the vector attribute used to get the timestamp value.
                     The time value is expected to be a float or integer.
        :type time: str

        :raises IndexError: When the cluster has no vectors.
        :raises TypeError: When the vector has no time attribute.
        """

        vector = cluster.vectors[-1]
        cluster.attributes['age'] = timestamp - vector.attributes.get(time)
