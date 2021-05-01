"""
TDT algorithms are split broadly into document-pivot and feature-pivot techniques:

- Document-pivot approaches use clustering to identify `what` is being discussed (read more `here <https://nyphoon.com/2020/07/27/document-pivot-methods-whats-happening/>`__), and
- Feature-pivot approaches identify `how` people are talking (read more `here <https://nyphoon.com/2020/08/06/feature-pivot-methods-did-something-happen/>`__).

Since TDT approaches are so broad by nature, it is difficult to find a general pattern.
However, all algorithms must have a mechanism to detect topics from a specific type input.
This functionality is encapsulated in the :class:`~tdt.algorithms.TDTAlgorithm`, which specifies only a :func:`~tdt.algorithms.TDTAlgorithm.detect` method.

.. note::

    Most TDT algorithms have a detection stage and a summarization stage.
    EvenTDT separates the two into:

    - :class:`~tdt.algorithms.TDTAlgorithm` and
    - :class:`~summarization.algorithms.SummarizationAlgorithm`.

    The reason why EvenTDT enforces this separation is that the two approaches tackle vastly different problems.
    Well-designed approaches delineate the separation between the two.
    If the :class:`~tdt.algorithms.TDTAlgorithm` is meant to work closely together with the :class:`~summarization.algorithms.SummarizationAlgorithm`, then its output needs to be compatible with the summarization algorithm.
"""

from abc import ABC, abstractmethod

class TDTAlgorithm(ABC):
    """
    Since TDT algorithms vary greatly, there is no general state.
    The :class:`~tdt.algorithms.TDTAlgorithm`'s purpose is to create a standard interface to TDT algorithms.
    Although the state and the parameters change from one algorithm to the other, all of them must implement the :func:`~tdt.algorithms.TDTAlgorithm.detect` method.
    """

    @abstractmethod
    def detect(self, *args, **kwargs):
        """
        Detect breaking topics.
        The parameters accepted by this function as well as the return value change according to the algorithm.
        """

        pass

from .eld import ELD, SlidingELD
from .zhao import Zhao
