"""
Zhao et al.'s algorithm is the first algorithm to be deployed over specified events on Twitter.
This approach looks for spikes in the overall tweeting volume in the most recent time window.

To identify these spikes, the approach splits time windows of increasing length into two.
If the second half has a marked increase in volume the algorithm identifies the second half as a topic.
The decision of whether something happened is based on a ratio, taken to be 1.7 in the original paper.
In practice, this means that if the second half of a time-window has 70% more tweets than the first half, then it represents a topic.

The time window starts at 10 seconds and changes dynamically.
If the increase is not significant, then the time window is progressively increased to 20 seconds, 30 seconds and, finally, 60 seconds.
If none of these time windows report a large enough increase, then the algorithm detects no topic.

The algorithm is very efficient and is suitable to run in real-time.
However, since it works only using the overall tweeting volume, it can only detect whether something happened.
It cannot explain what happened, or what the most important features are.

.. note::

    This implementation is based on the algorithm presented in `Human as Real-Time Sensors of Social and Physical Events: A Case Study of Twitter and Sports Games by Zhao et al. (2011) <https://arxiv.org/abs/1106.4300>`_.
"""

import math
import os
import time
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from tdt.algorithms import TDTAlgorithm

class Zhao(TDTAlgorithm):
    """
    In order to detect topics, Zhao et al.'s algorithm looks at the increase between two halves of a time window.
    The original paper set this increase to 70%, but this implementation supports other values.

    In addition to the ratio, since this approach is a feature-pivot technique also stores a :class:`~tdt.nutrition.NutritionStore`.
    This implementation uses a sliding time window.
    Therefore the keys of the :class:`~tdt.nutrition.NutritionStore` should be timestamps.
    The values at each timestamp should be the number of documents observed at that timestamp.
    The algorithm automatically separates the nutrition according to the varying sizes of the time window.

    :ivar store: The store contraining historical nutrition data.
                 The algorithm expects the nutrition values to represent the stream volume.
                 Therefore the keys should be the timestamps, and the values should integers representing the number of documents observed at that timestamp.
    :vartype store: :class:`~tdt.nutrition.store.NutritionStore`
    :ivar post_rate: The minimum increase between the two halves of the sliding time window to represent a burst.
    :vartype post_rate: float
    """

    def __init__(self, store, post_rate=1.7):
        """
        :param store: The store contraining historical nutrition data.
                      The algorithm expects the nutrition values to represent the stream volume.
                      Therefore the keys should be the timestamps, and the values should integers representing the number of documents observed at that timestamp.
        :type store: :class:`~tdt.nutrition.store.NutritionStore`
        :param post_rate: The minimum increase between the two halves of the sliding time window to represent a burst.
        :type post_rate: float
        """

        self.store = store
        self.post_rate = post_rate

    def detect(self, timestamp=None):
        """
        Detect topics using historical data from the nutrition store.
        This function receives the timestamp and creates time windows of varying sizes that end at that timestamp.

        :param timestamp: The timestamp at which to try to identify emerging topics.
                          If it is not given, the current timestamp is used.
                          This value is exclusive.
        :type timestamp: float or None

        :return: A tuple with the start and end timestamp of the time window when there was a burst.
                 If there was no burst, `False` is returned.
        :rtype: tuple or bool
        """

        """
        If no time was given, default to the current timestamp.
        """
        timestamp = timestamp or time.time()

        """
        Go through each time window and check whether there as a breaking development.
        """
        time_windows = [ 10, 20, 30, 60 ]
        for window in time_windows:
            """
            Split the time window in two and get the volume in both.
            """
            half_window = window / 2.
            first_half = self.store.between(timestamp - window, timestamp - half_window)
            second_half = self.store.between(timestamp - half_window, timestamp)

            """
            If the first half has no tweets, skip the time window.
            """
            if sum(first_half.values()) == 0:
                continue

            """
            Calculate the increase in post rate.
            If the ratio is greater than or equal to the post rate, the time window is breaking.
            Therefore return the emerging period: the second half of the time window.
            """
            ratio = sum(second_half.values()) / sum(first_half.values())
            if ratio >= self.post_rate:
                return (float(min(second_half)), float(max(second_half)))

        """
        Return `False` if none of the time windows were deemed to be emerging.
        """
        return False
