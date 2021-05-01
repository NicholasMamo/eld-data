"""
A cluster is a group of :class:`~vsm.vector.Vector` instances, or other classes that inherit from the :class:`~vsm.vector.Vector` class.
The purpose of clusters is that they represent a single topic.
To achieve this goal, good clusters:

1. Have a high distance between them if they (or, their instances) represent different topics, and
2. Have a small distance between their own :class:`~vsm.vector.Vector` instances.

Although you can create a :class:`~vsm.clustering.cluster.Cluster` instance yourself, it is more common to generate clusters automatically using a :class:`~vsm.clustering.algorithms.clustering.ClusteringAlgorithm`.
"""

import importlib
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from objects.attributable import Attributable
from objects.exportable import Exportable

from vsm import Vector, vector_math

class Cluster(Attributable, Exportable):
    """
    The :class:`~vsm.clustering.cluster.Cluster` class is a collection of :class:`~vsm.vector.Vector` instances, or inherited classes, like :class:`~nlp.document.Document`.
    You can add :class:`~vsm.vector.Vector` instances to the :class:`~vsm.clustering.cluster.Cluster` using the ``vectors`` property.
    This property wraps a ``list``, which means that you can also use it to retrieve the :class:`~vsm.vector.Vector` instances.

    The :class:`~vsm.clustering.cluster.Cluster` also has a special property: ``centroid``.
    The centroid is a :class:`~vsm.vector.Vector` that represents the cluster's general direction.
    It is calculated as an average over all the dimensions of the :class:`~vsm.vector.Vector` instances making up the cluster.

    Clusters are based on the :class:`~objects.attributable.Attributable` class so they may have additional properties.

    :ivar vectors: The list of vectors that make up the cluster.
    :vartype vectors: list of :class:`~vsm.vector.Vector`
    :ivar centroid: The centroid of the cluster, representing the average vector in the cluster.
    :vartype centroid: :class:`~vsm.vector.Vector`
    :ivar _last: A snapshot of the cluster the last time that its centroid was re-calculated.
                 This variable is used so that the centroid is not re-calculated needlessly if the cluster's vector has not changed.
    :vartype _last: dict
    """

    def __init__(self, vectors=None, *args, **kwargs):
        """
        Initialize the cluster with an empty centroid and a list of vectors.

        :param vectors: An initial list of vectors, or a single vector.
                        If ``None`` is given, an empty list is initialized instead.
        :type vectors: list of :class:`~vsm.vector.Vector` or :class:`~vsm.vector.Vector` or ``None``
        """

        super(Cluster, self).__init__(*args, **kwargs)
        self.vectors = vectors
        self.centroid = Vector()
        self._last = None

    def similarity(self, vector, similarity_measure=vector_math.cosine):
        """
        Calculate the similarity between the given vector and this cluster's centroid.
        The similarity and its bounds depends on the similarity measure that is used.
        The similarity measure defaults to :func:`~vsm.vector_math.cosine`, but a different measure can be provided.

        :param vector: The vector that will be compared with the centroid.
        :type vector: :class:`~vsm.vector.Vector`
        :param similarity_measure: The similarity function to use to compare the likeliness of the vector with the cluster.
        :type similarity_measure: func

        :return: The similarity between the cluster and the vector.
        :rtype: float
        """

        return similarity_measure(self.centroid, vector)

    def recalculate_centroid(self):
        """
        Recalculate the centroid.
        The centroid is the average value of all dimensions across all vectors.
        For example, the centroid's magnitude :math:`c_f` of feature :math:`f` is calculated as:

        .. math::

            c_f = \\frac{1}{|V|} \\sum^{v \\in V} v_f

        where :math:`V` is the list of vectors in the cluster and :math:`v_f` is the magnitude of feature :math:`f` in vector :math:`V`.
        After calculating the centroid, the function automatically normalizes it so that the centroid's magnitude is 1.

        .. note::

            Normally, you wouldn't need to call this function.
            In fact, this function does not return the centroid.
            This function is invoked automatically when you fetch the centroid.
        """

        centroid = { }
        for vector in self.vectors:
            for dimension, magnitude in vector.dimensions.items():
                centroid[dimension] = centroid.get(dimension, 0) + magnitude / len(self.vectors)

        self.__centroid = Vector(centroid)
        self.__centroid.normalize()

    @property
    def centroid(self):
        """
        Get the cluster's centroid.
        Before returning the centroid, this function automatically re-calculates it.

        :return: The cluster's centroid.
        :rtype: :class:`~vsm.vector.Vector`
        """

        current = self.to_array()
        if not self._last or self._last != current:
            self._last = current
            self.recalculate_centroid()

        return self.__centroid

    @centroid.setter
    def centroid(self, centroid):
        """
        Override the centroid.

        :param centroid: The new centroid.
        :type centroid: :class:`~vsm.vector.Vector`
        """

        self.__centroid = centroid

    @property
    def vectors(self):
        """
        Get the list of vectors in the cluster.

        :return: The list of vectors in the cluster.
        :rtype: list of :class:`~vsm.vector.Vector`
        """

        return self.__vectors

    @vectors.setter
    def vectors(self, vectors=None):
        """
        Override the vectors.

        :param vectors: The new vectors.
        :type vectors: list of :class:`~vsm.vector.Vector` or :class:`~vsm.vector.Vector` or None
        """

        if vectors is None:
            self.__vectors = [ ]
        elif type(vectors) is list:
            self.__vectors = list(vectors)
        else:
            self.__vectors = [ vectors ]

    def get_representative_vectors(self, vectors=1, similarity_measure=vector_math.cosine):
        """
        Get the vectors that are closest to the centroid.
        By default, :func:`~vsm.vector_math.cosine` is used to calculate the similarity between a vector and the centroid.
        However, a different similarity measure can be provided.

        :param vectors: The number of vectors the fetch.
        :type vectors: int
        :param similarity_measure: The similarity function to use to compare the likeliness of the vector with the cluster.
        :type similarity_measure: func

        :return: The representative vectors.
                 If the number of vectors that is sought is one, only the closest vector is returned.
                  Otherwise, a list of vectors is returned.
        :rtype: :class:`~vsm.vector.Vector` or list of :class:`~vsm.vector.Vector`
        """

        """
        First calculate all the similarities between the centroid and each vector in the cluster.
        Then, rank all vectors by their similarity score.
        """

        similarities = [ self.similarity(vector, similarity_measure) for vector in self.vectors ]
        similarities = zip(self.vectors, similarities)
        similarities = sorted(similarities, key=lambda x:x[1])[::-1]

        """
        If only one vector is needed, just return the vector, not a list of vectors.
        Otherwise return a list.
        """
        if vectors == 1:
            return similarities[0][0] if len(similarities) else None
        else:
            similarities = similarities[:vectors]
            return [ similarity[0] for similarity in similarities ]

    def get_intra_similarity(self, similarity_measure=vector_math.cosine):
        """
        Get the average similarity between vectors and the cluster.
        This is calculated by comparing each vector with the cluster's centroid individually.
        At the end, the function calculates the average similarity.

        By default, :func:`~vsm.vector_math.cosine` is used to calculate the similarity between vectors in the cluster.
        However, a different similarity measure can be provided.

        :param similarity_measure: The similarity function to use to compare the likeliness of the vector with the cluster.
        :type similarity_measure: func

        :return: The average intra-similarity of the cluster.
        :rtype: float
        """

        if self.vectors:
            centroid = self.centroid
            similarities = [ similarity_measure(centroid, vector) for vector in self.vectors ]
            return sum(similarities)/len(similarities)

        return 0

    def size(self):
        """
        Get the number of vectors in the cluster.

        :return: The number of vectors in the cluster.
        :rtype: int
        """

        return len(self.vectors)

    def to_array(self):
        """
        Export the cluster as an associative array.
        The centroid is not included as it is calculated every time it is requested.

        :return: The cluster as an associative array.
        :rtype: dict
        """

        return {
            'class': str(Cluster),
            'attributes': self.attributes,
            'vectors': [ vector.to_array() for vector in self.vectors ]
        }

    @staticmethod
    def from_array(array):
        """
        Create a :class:`~vsm.clustering.cluster.Cluster` instance from the given associative array.

        :param array: The associative array with the attributes to create the cluster.
        :type array: dict

        :return: A new instance of an object with the same attributes stored in the object.
        :rtype: :class:`~vsm.clustering.cluster.Cluster`
        """

        vectors = [ ]
        for vector in array.get('vectors'):
            module = importlib.import_module(Exportable.get_module(vector.get('class')))
            cls = getattr(module, Exportable.get_class(vector.get('class')))
            vectors.append(cls.from_array(vector))

        return Cluster(vectors=vectors, attributes=array.get('attributes'))
