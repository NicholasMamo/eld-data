"""
The term frequency weighting scheme is used when there is no need for a global scheme.
The term frequency :math:`tf_{t,d}` of a feature :math:`t` is equivalent to its frequency :math:`f_{t,d}` in document :math:`d`:

.. math::

    tf_{t,d} = f_{t,d}
"""

import os
import sys

path = os.path.join(os.path.dirname(__file__), '..')
if path not in sys.path:
    sys.path.append(path)

from weighting import TermWeightingScheme
from weighting.local_schemes import tf
from weighting.global_schemes.filler import Filler

class TF(TermWeightingScheme):
    """
    The term frequency weighting scheme is used when there is no need for a global scheme.
    It is an instance of a :class:`~nlp.weighting.TermWeightingScheme` with:

    1. :class:`~nlp.weighting.local_schemes.tf.TF` as a local term-weighting scheme, and
    2. :class:`~nlp.weighting.global_schemes.filler.Filler` as a global term-weighting scheme (that does not change the score at all).
    """

    def __init__(self):
        """
        Initialize the term-weighting scheme by supplying the TF and filler schemes.
        """

        super(TF, self).__init__(tf.TF(), Filler())
