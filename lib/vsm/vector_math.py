"""
The advantage of :class:`~vsm.vector.Vector` is that as mathematical constructs, we can also manipulate and compare them with mathematics.
This script contains a list of mathematical functions that can be used to change and compare :class:`~vsm.vector.Vector` instances.
"""

import math
import os
import sys

path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.append(path)

from vsm import vector

def magnitude(v):
    """
    Get the magnitude of the given :class:`~vsm.vector.Vector`.
    The magnitude is computed as:

    .. math::

        ||v|| = \\sqrt{\\sum_{n=1}^{V} {v_n^2}}

    where :math:`v` is a :class:`~vsm.vector.Vector` having :math:`V` dimensions.

    :param v: The vector whose magnitude will be calculated.
    :type v: :class:`~vsm.vector.Vector`

    :return: The magnitude of the :class:`~vsm.vector.Vector`.
    :rtype: float
    """

    return math.sqrt(sum([value ** 2 for value in v.dimensions.values()]))

def normalize(v):
    """
    Normalize the given :class:`~vsm.vector.Vector`.
    Normalization is computed as:

    .. math::

        f = \\frac{f}{||v||}

    where :math:`f` is a feature in vector :math:`v`.
    After normalizing the :class:`~vsm.vector.Vector`, its magnitude will become 1.

    .. warning::

        This is different from the :func:`~augmented_normalize` function.
        The new :class:`~vsm.vector.Vector` will have a magnitude of 1 because the denominator is the original :class:`~vsm.vector.Vector`'s magnitude.

    :param v: The :class:`~vsm.vector.Vector` that will be normalized.
    :type v: :class:`~vsm.vector.Vector`

    :return: A new :class:`~vsm.vector.Vector` with a magnitude of 1.
    :rtype: :class:`~vsm.vector.Vector`
    """

    n = v.copy()

    m = magnitude(n)
    if m > 0:
        dimensions = { dimension: float(value)/m for dimension, value in n.dimensions.items() }
        return vector.Vector(dimensions)
    else:
        return v

def augmented_normalize(v, a=0.5):
    """
    Normalize the given :class:`~vsm.vector.Vector` using the formula:

    .. math::

        f = a + (1 - a) \\frac{f}{x}

    where :math:`x` is the magnitude of the highest dimension :math:`f` in the :class:`~vsm.vector.Vector`.
    :math:`a` is the augmentation, between 0 and 1, inclusive.

    This function normalizes the :class:`~vsm.vector.Vector`, but with a magnitude that is not 1.
    Instead, all of its dimensions will have a magnitude of at least :math:`a`.

    .. warning::

        This is different from the :func:`~normalize` function.
        The new :class:`~vsm.vector.Vector` will not have a magnitude of 1.
        This is because the denominator is not the magnitude, but the highest dimension,
        You can read more about this type of normalization `here <https://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html>`__.

    :param v: The :class:`~vsm.vector.Vector` that will be normalized
    :type v: :class:`~vsm.vector.Vector`
    :param a: The minimum magnitude of each dimension.
    :type a: float

    :return: A new :class:`~vsm.vector.Vector`, normalized with all dimensions having a magnitude of at least :math:`a`.
    :rtype: float

    :raises ValueError: When the augmentation is not between 0 and 1
    """

    if not 0 <= a <= 1:
        raise ValueError(f"The augmentation must be between 0 and 1 inclusive, {a} received")

    n = v.copy()

    dimensions = n.dimensions
    x = max(dimensions.values()) if len(dimensions) > 0 else 1
    dimensions = { dimension: a + (1 - a) * value / x for dimension, value in dimensions.items() }
    n.dimensions = dimensions
    return n

def concatenate(vectors):
    """
    Concatenate a list of vectors and return a new vector.
    This simply means adding the dimensions together:

    .. math::

        f = \\sum_{i=1}^{|V|}{f_i}

    where :math:`f` is the weight of the concatenated vector.
    :math:`f_i` is the weight of the same feature `f` in each vector in the set :math:`V`.

    :param vectors: A list of vectors.
    :type vectors: list of :class:`~vsm.vector.Vector` instances

    :return: A new :class:`~vsm.vector.Vector` whose dimensions are the summations of the given :class:`~vsm.vector.Vector` instances.
    :rtype: :class:`~vsm.vector.Vector`
    """

    concatenated = { }
    for v in vectors:
        for dimension in v.dimensions:
            concatenated[dimension] = concatenated.get(dimension, 0) + v.dimensions[dimension]

    return vector.Vector(concatenated)

def euclidean(v1, v2):
    """
    Compute the Euclidean distance between the two :class:`~vsm.vector.Vector` instances.
    The Euclidean distance :math:`e_{p, q}` is computed as:

    .. math::

        e_{p, q} = \\sqrt{ \\sum_{i=1}^{n}{ (q_i - p_i)^2 } }

    Where :math:`q_i` is feature :math:`i` in :class:`~vsm.vector.Vector` :math:`q`, and :math:`p_i` is the same feature :math:`i` in :class:`~vsm.vector.Vector` :math:`p`.
    :math:`n` is the union of features in :class:`~vsm.vector.Vector` :math:`q` and :class:`~vsm.vector.Vector` :math:`p`.

    :param v1: The first :class:`~vsm.vector.Vector`.
    :type v1: :class:`~vsm.vector.Vector`
    :param v2: The second :class:`~vsm.vector.Vector`.
    :type v2: :class:`~vsm.vector.Vector`

    :return: The Euclidean distance between the two :class:`~vsm.vector.Vector` instances.
             This distance has a lower-bound of 0.
    :rtype: float
    """

    dimensions = list(set(v1.dimensions.keys()).union(v2.dimensions.keys()))
    differences = [ (v1.dimensions[dimension] - v2.dimensions[dimension]) ** 2 for dimension in dimensions ]
    return math.sqrt(sum(differences))

def manhattan(v1, v2):
    """
    Compute the Manhattan distance between the two :class:`~vsm.vector.Vector` instances.
    The Manhattan distance :math:`m_{p, q}` is computed as:

    .. math::

        m_{p, q} = \\sum_{i=1}^{n}{ |q_i - p_i| }

    Where :math:`q_i` is feature :math:`i` in :class:`~vsm.vector.Vector` :math:`q`, and :math:`p_i` is the same feature :math:`i` in :class:`~vsm.vector.Vector` :math:`p`.
    :math:`n` is the union of features in :class:`~vsm.vector.Vector` :math:`q` and :class:`~vsm.vector.Vector` :math:`p`.

    :param v1: The first :class:`~vsm.vector.Vector`.
    :type v1: :class:`~vsm.vector.Vector`
    :param v2: The second :class:`~vsm.vector.Vector`.
    :type v2: :class:`~vsm.vector.Vector`

    :return: The Manhattan distance between the two :class:`~vsm.vector.Vector` instances.
             This distance has a lower-bound of 0.
    :rtype: float
    """

    dimensions = list(set(v1.dimensions.keys()).union(v2.dimensions.keys()))
    differences = [ abs(v1.dimensions[dimension] - v2.dimensions[dimension]) for dimension in dimensions ]
    return sum(differences)

def cosine(v1, v2):
    """
    Compute the cosine similarity between the two :class:`~vsm.vector.Vector` instances.
    The cosine similarity :math:`cos_{p, q}` is computed as:

    .. math::

        cos_{p, q} = \\frac{\\sum_{i=1}^{n}{ q_i \\cdot p_i }}{ ||p|| + ||q|| }

    Where :math:`q_i` is feature :math:`i` in :class:`~vsm.vector.Vector` :math:`q`, and :math:`p_i` is the same feature :math:`i` in :class:`~vsm.vector.Vector` :math:`p`.
    :math:`n` is the intersection of features in :class:`~vsm.vector.Vector` :math:`q` and :class:`~vsm.vector.Vector` :math:`p`.

    :param v1: The first :class:`~vsm.vector.Vector`.
    :type v1: :class:`~vsm.vector.Vector`
    :param v2: The second :class:`~vsm.vector.Vector`.
    :type v2: :class:`~vsm.vector.Vector`

    :return: The cosine similarity between the two :class:`~vsm.vector.Vector` instances.
             This similarity is bound between 0 and 1.
    :rtype: float
    """

    m1, m2 = magnitude(v1), magnitude(v2)
    if (m1 > 0 and m2 > 0):
        dimensions = set(list(v1.dimensions.keys()) + list(v2.dimensions.keys()))
        products = [ v1.dimensions[dimension] * v2.dimensions[dimension] for dimension in dimensions ]
        return sum(products) / (m1 * m2)
    else:
        return 0

def cosine_distance(v1, v2):
    """
    Compute the cosine distance between the two :class:`~vsm.vector.Vector` instances.
    The cosine distance :math:`cosd_{p, q}` is computed as:

    .. math::

        cosd_{p, q} = 1 - cos_{p, q}

    .. warning::

        The cosine distance is not a real distance metric as it does not have the triangle inequality property.
        You can read more about why that is `here <https://en.wikipedia.org/wiki/Cosine_similarity>`__.

    :param v1: The first :class:`~vsm.vector.Vector`.
    :type v1: :class:`~vsm.vector.Vector`
    :param v2: The second :class:`~vsm.vector.Vector`.
    :type v2: :class:`~vsm.vector.Vector`

    :return: The cosine distance between the two :class:`~vsm.vector.Vector` instances.
             This distance is bound between 0 and 1.
    :rtype: float
    """

    return 1 - cosine(v1, v2)
