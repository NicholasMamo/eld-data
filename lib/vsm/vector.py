"""
The basis for vectors is the :class:`~VectorSpace`, or the space of all vector dimensions.
A :class:`~Vector` is made up of dimensions in the :class:`~VectorSpace` which make up its direction.
Therefore the :class:`~VectorSpace` is one of the most important classes in EvenTDT because it is the basis for a lot of the vector-related functionality.
"""

import os
import sys

path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)

from objects.attributable import Attributable
from objects.exportable import Exportable
from vsm import vector_math

class VectorSpace(dict):
    """
    The :class:`~VectorSpace` represents the space of all :class:`~Vector` dimensions.
    This class is based on a normal Python ``dict``:

        - The keys of this dictionary represent the feature or dimension name.
        - The corresponding values represent the magnitude of the :class:`~Vector` along that dimension.

    The only change from the normal dictionary is that the value of an unspecified dimension is not undefined or ``None``.
    Instead, the :class:`~VectorSpace` returns 0: a :class:`~Vector` is made up of all dimensions, but some (or many) of them have a magnitude of 0.
    This functionality is implemented in the :func:`~VectorSpace.__getitem__` function.
    """

    def __getitem__(self, key):
        """
        Get the value, or magnitude, of the dimension having the given key as name.
        If the :class:`~Vector` has no dimension with that name, the function returns a magnitude of 0.

        :param key: The name of the dimension whose magnitude will be fetched.
        :type key: str

        :return: The magnitude of the dimension, or 0 if the :class:`~Vector` does not have a value for the dimension.
        :rtype: float
        """

        return self.get(key, 0)

class Vector(Attributable, Exportable):
    """
    The :class:`~Vector` class is a manifestation of a vector in the :class:`~VectorSpace`.
    In this library, it is the basic building block for many other classes, such as the :class:`~nlp.document.Document` and the :class:`~vsm.clustering.cluster.Cluster`.

    The :class:`~Vector` is based on two other classes that add some functionality to the basics:

        - :class:`~objects.attributable.Attributable`: allows :class:`~Vector` instances to accept additional attributes.
        - :class:`~objects.exportable.Exportable`: allows :class:`~Vector` instances to be exported as an associative array and imported back again.

    The :class:`~Vector` stores its make-up information in the :class:`~VectorSpace`.
    This can be accessed through the ``dimensions`` property.

    :ivar dimensions: The dimensions of the :class:`~Vector` in the :class:`~VectorSpace`.
                      The keys are the dimension names.
                      The corresponding values are the magnitude of the :class:`~Vector` in that direction.
    :vartype dimensions: :class:`~VectorSpace`
    """

    def __init__(self, dimensions=None, *args, **kwargs):
        """
        Create the :class:`~Vector`.
        By default, it has no dimensions, but you can provide initial values as a ``dict``.

        :param dimensions: The initial dimensions of the :class:`~Vector`.
                           If ``None`` is given, the class initializes the dimensions as an empty ``dict``.
        :type dimensions: dict or :class:`~VectorSpace` or None
        """

        super(Vector, self).__init__(*args, **kwargs)
        self.dimensions = dimensions

    @property
    def dimensions(self):
        """
        Get the dimensions of the :class:`~Vector`.
        The dimensions are returned as a :class:`~VectorSpace`, which is based on a ``dict``.

        :return: The dimensions of the :class:`~Vector`.
        :rtype: :class:`~VectorSpace`
        """

        return self.__dimensions

    @dimensions.setter
    def dimensions(self, dimensions=None):
        """
        Set the dimensions to the given dictionary or :class:`~VectorSpace`.
        If you provide ``None`` as the dimensions, the function resets the :class:`~Vector`'s :class:`~VectorSpace`.

        :param dimensions: The new dimensions as a dictionary.
                           If ``None`` is given, the class initializes the dimensions with an empty :class:`~VectorSpace` instead.
        :type dimensions: dict or :class:`~VectorSpace` or ``None``
        """

        self.__dimensions = VectorSpace() if dimensions is None else VectorSpace(dimensions)

    def normalize(self):
        """
        Normalize the :class:`~Vector` so that its magnitude is 1.

        .. note::

            You can read more about :class:`~Vector` normalization :func:`here <vsm.vector_math.normalize>`.
        """

        self.dimensions = vector_math.normalize(self).dimensions

    def copy(self):
        """
        Create a copy of the :class:`~Vector`.
        The copy has the same :class:`~VectorSpace` dimensions and attributes.

        :return: A copy of this :class:`~Vector` instance.
        :rtype: :class:`~Vector`
        """

        return Vector(self.dimensions.copy(), self.attributes.copy())

    def to_array(self):
        """
        Export the :class:`~Vector` as ``dict``.
        This ``dict`` has three keys:

            1. The class name, used when re-creating the :class:`~Vector`;
            2. The :class:`~Vector`'s attributes as a ``dict``; and
            3. The :class:`~Vector`'s dimensions as a ``dict``.

        :return: The :class:`~Vector` as ``dict``.
        :rtype: dict
        """

        return {
            'class': str(Vector),
            'attributes': self.attributes,
            'dimensions': self.dimensions,
        }

    @staticmethod
    def from_array(array):
        """
        Create an instance of the :class:`~Vector` from the given ``dict``.
        This function expects the array to have been generated by the :func:`~Vector.to_array`, and must have these keys:

            1. The class name,
            2. The :class:`~Vector`'s attributes as a ``dict``, and
            3. The :class:`~Vector`'s dimensions as a ``dict``.

        :param array: The ``dict`` with the attributes to create the :class:`~Vector`.
        :type array: dict

        :return: A new instance of the :class:`~Vector` with the same attributes stored in the object.
        :rtype: :class:`~Vector`
        """

        return Vector(dimensions=array.get('dimensions'), attributes=array.get('attributes'))
