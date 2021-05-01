"""
Document representation in EvenTDT uses the Vector Space Model (VSM).
The library represents documents as vectors, where each feature can mean a different word.

EvenTDT builds the VSM from scratch.
First, there is the :class:`~vsm.vector.VectorSpace` class, which is a special dictionary that represents the VSM.
Second, there is the :class:`~vsm.vector.Vector` class, which is a vector in the :class:`~vsm.vector.VectorSpace`.

.. note::

    Documents are special vectors that have text.
    You can read more about documents in the :ref:`NLP chapter <nlp>`.
"""

from .vector import *
