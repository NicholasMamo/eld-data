"""
The concept of nutrition was used predominantly by `Cataldi et al. <https://dl.acm.org/doi/abs/10.1145/2542182.2542189>`_.
Later, it was adopted by in several papers, such as in `FIRE <https://link.springer.com/chapter/10.1007/978-3-319-74497-1_3>`_ and `ELD <https://dl.acm.org/doi/abs/10.1145/3342220.3344921>`_.

Nutrition is most common in feature-pivot techniques as it is a general way of measuring the importance of features.
For example, `Cataldi et al. <https://dl.acm.org/doi/abs/10.1145/2542182.2542189>`_ used it to measure the popularity of a term.
Even when the term `nutrition` is not used, the importance of terms can be thought of as nutrition.

Nutrition is important because it often fuels another TDT metrid: burst.
Burst, introduced by `Kleinberg <https://link.springer.com/article/10.1023/A:1024940629314>`_, measures the change in nutrition to find spikes in the use of terms, for example.
More generally, burst tries to identify a change in behavior, which may also be a sharp increase in tweeting volume.

The :class:`~tdt.nutrition.NutritionStore` is meant to be an interface for any data structure that stores nutrition.
The interface contains the methods that all stores must implement.
For example, implementations can store data in a database or in memory.

All nutrition stores separate nutrition based on timestamps, which can represent, among others:

- The nutrition data at that particular timestamp, or
- The nutrition data during a period of time represented by that timestamp.
"""

from abc import ABC, abstractmethod

class NutritionStore(ABC):
    """
    The :class:`~tdt.nutrition.NutritionStore` base class does not make any assumptions about where the nutrition data is stored.
    However, it defines a set of functions that need to be implemented.
    In that way, all derived classes share a similar interface.

    The nutrition store provides functions that cover not just storage, but also retrieval.
    A common theme across all nutrition store instances is that they separate nutrition data into timestamps.
    Timestamps can represent either a particular instance of time, or a period of time.
    This depends on the application, and is not a feature of the nutrition store.
    """

    @abstractmethod
    def __init__(self):
        """
        Create the nutrition store.
        All nutrition stores use this function to initialize the storage or their connections to it.
        """

        pass

    @abstractmethod
    def add(self, timestamp, nutrition):
        """
        Add a nutrition data to the store at the given timestamp.

        .. warning::

            This function overwrites any data at the given timestamp.

        :param timestamp: The timestamp of the nutrition data.
        :type timestamp: float or int
        :param nutrition: The nutrition data to add.
                          The nutrition data can be any value.
        :type nutrition: any
        """

        pass

    @abstractmethod
    def get(self, timestmap):
        """
        Get the nutrition data at the given timestamp.

        :param timestamp: The timestamp whose nutrition is to be returned.
        :type timestamp: float or int

        :return: The nutrition at the given timestamp.
        :rtype: any
        """

        pass

    @abstractmethod
    def all(self):
        """
        Get all the nutrition data.

        :return: All the nutrition data in the nutrition store as a dictionary.
                 The keys are the timestamps, and the values are the nutrition data at those timestamps.
        :rtype: dict
        """

        pass

    @abstractmethod
    def between(self, start, end):
        """
        Get the nutrition data between the given timestamps.

        .. note::

            The start timestamp is inclusive, the end timestamp is exclusive.

        :param start: The first timestamp that should be included in the returned nutrition data.
                      If no time window with the given timestamp exists, all returned time windows succeed it.
        :type start: float or int
        :param end: All the nutrition data from the beginning until the given timestamp.
                    Any nutrition data at the end timestamp is not returned.
        :type end: float or int

        :return: All the nutrition data between the given timestamps.
                 The start timestamp is inclusive, the end timestamp is exclusive.
        :rtype: dict
        """

        pass

    def since(self, start):
        """
        Get the nutrition data since the given timestamp.

        .. note::

            The start timestamp is inclusive.

        :param start: The first timestamp that should be included in the returned nutrition data.
                      If no time window with the given timestamp exists, all returned time windows succeed it.
        :type start: float or int

        :return: All the nutrition data from the given timestamp onward.
        :rtype: dict
        """

        if self.all():
            timestamps = [ float(timestamp) for timestamp in self.all().keys() ]
            last = max(timestamps)
            return self.between(start, float(last) + 1)

        return { }

    def until(self, end):
        """
        Get a list of nutrition sets that came before the given timestamp.

        .. note::

            The end timestamp is exclusive.

        :param end: The timestamp before which nutrition data should be returned.
        :type end: float or int

        :return: All the nutrition data from the beginning until the given timestamp.
                 Any nutrition data at the end timestamp is not returned.
        :rtype: dict
        """

        if self.all():
            timestamps = [ float(timestamp) for timestamp in self.all().keys() ]
            last = min(timestamps)
            return self.between(0, str(end))

        return { }

    @abstractmethod
    def remove(self, *args):
        """
        Remove nutrition data from the given list of timestamps.
        The timestamps should be given as arguments.
        """

        pass

    @abstractmethod
    def copy(self):
        """
        Create a copy of the nutrition store.
        The copy is a deep copy.
        This means that all changes to the copy do not affect the original.

        :return: A copy of the nutrition store.
        :rtype: :class:`~tdt.nutrition.store.NutritionStore`
        """

        pass

from .memory import MemoryNutritionStore
