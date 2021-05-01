"""
The :class:`~tdt.nutrition.memory.MemoryNutritionStore` stores data in a Python dictionary in memory.
This store is simple and efficient because there is little overhead, unlike when connecting to a database, for example.
However, since all of the nutrition data is stored in memory, you need to pay attention to the :class:`~tdt.nutrition.memory.MemoryNutritionStore`'s blueprint.
Like all :class:`~tdt.nutrition.NutritionStore` instances, you can clear out old data to keep the memory requirements in check.
"""

import copy
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from tdt.nutrition import NutritionStore

class MemoryNutritionStore(NutritionStore):
    """
    The :class:`~tdt.nutrition.memory.MemoryNutritionStore` stores data in a Python dictionary.
    The keys are the timestamps, and the value is the nutrition data at that timestamp.

    The nutrition data can be several things:

    - If nutrition represents the overall volume, the nutrition data can be an integer representing the number of documents observed at that timestamp.
    - If nutrition represents the number of times a term appears, the nutrition data can be a dictionary, with terms as keys and their frequency as the corresponding values.

    In addition, the :class:`~tdt.nutrition.memory.MemoryNutritionStore` does not impose any restrictions on the nutrition data.
    It can represent objects and different timestamps can have different data types.

    :ivar store: The nutrition store as a dictionary.
                 The keys are the timestamps, and the values are the nutrition data.
                 The nutrition data can be any value.
    :vartype store: dict
    """

    def __init__(self):
        """
        Create the nutrition store as a dictionary.
        """

        self.store = { }

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

        self.store[float(timestamp)] = nutrition

    def get(self, timestamp):
        """
        Get the nutrition data at the given timestamp.

        :param timestamp: The timestamp whose nutrition is to be returned.
        :type timestamp: float or int

        :return: The nutrition at the given timestamp.
        :rtype: any
        """

        return self.store.get(float(timestamp))

    def all(self):
        """
        Get all the nutrition data.

        :return: All the nutrition data in the nutrition store as a dictionary.
                 The keys are the timestamps, and the values are the nutrition data at those timestamps.
        :rtype: dict
        """

        return self.store

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

        :raises ValueError: When the start timestamp is on or after the end timestamp.
        """

        if float(start) >= float(end):
            raise ValueError(f"The start timestamp must be before the end timestamp: {start} >= {end}")

        return { timestamp: self.get(timestamp) for timestamp in self.store if float(start) <= float(timestamp) < float(end) }

    def remove(self, *args):
        """
        Remove nutrition data from the given list of timestamps.
        The timestamps should be given as arguments.
        """

        timestamps = [ float(timestamp) for timestamp in args ]
        self.store = { timestamp: self.store.get(timestamp) for timestamp in self.store if timestamp not in timestamps }

    def copy(self):
        """
        Create a copy of the nutrition store.
        The copy is a deep copy.
        This means that all changes to the copy do not affect the original.

        :return: A copy of the nutrition store.
        :rtype: :class:`~tdt.nutrition.memory.MemoryNutritionStore`
        """

        store = MemoryNutritionStore()

        copy_data = copy.deepcopy(self.all())
        for timestamp, data in copy_data.items():
            store.add(timestamp, data)

        return store
