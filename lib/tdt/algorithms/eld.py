"""
Event TimeLine Detection (ELD) is a feature-pivot TDT approach designed to create interpretable results.

.. note::

    The original implementation of ELD is a combined document-pivot and feature-pivot TDT approach.
    First, it clusters documents and then it applies the feature-pivot technique on large clusters.
    However, the algorithm in this module is the feature-pivot technique.
    The full implementation of ELD is in the :class:`~queues.consumers.algorithms.eld_consumer.ELDConsumer`.

ELD computes burst for each individual term.
The interpretation is in the form of a burst value that lies between -1 and 1:

- -1 indicates that a term has lost all popularity since the past checkpoints,
- 0 indicates that a term's popularity has not changed over the past checkpoints, and
- 1 that the term has gone from completely unpopular to maximum popularity in the most recent checkpoint.

Negative burst can be used to check when a topic is over.
This is because after the peak, when the topic slows down, the burst becomes negative or close to zero.

The checkpoints work like checkpoints.
The complete system routinely creates checkpoints that represent the importance of terms in a particular checkpoint.
For example, at timestamp 100, the checkpoint can represent the importance of terms between timestamps 90 and 100.

To calculate the burst, this algorithm compares the local context with the global context.
The local context refers to the importance of a term in a cluster.
The global context refers to the importance of a term in previous checkpoints, which consider all documents, not just those in a cluster.
Burst is calculated as:

.. math::

    burst_k^t = \\frac{\\sum_{c=t-s}^{t-1}((nutr_{k,l} - nutr_{k,c}) \\cdot \\frac{1}{\\sqrt{e^{t - c}}})}{\\sum_{c=1}^s\\frac{1}{\\sqrt{e^c}}}

where :math:`k` is the term for which burst is to be calculated.
:math:`t` is the current checkpoint and :math:`s` is the number of checkpoints to consider.
:math:`nutr_{k,l}` is the nutrition of the term in the local context.
:math:`nutr_{k,c}` is the nutrition of the term in the checkpoint :math:`c`.

The square root in the burst calculation is the decay rate and is one of the parameters of this algorithm.
The higher the decay rate, the less importance old checkpoints receive.

.. note::

    ELD borrows the terminology from Cataldi et al.'s previous work and algorithm: :class:`~tdt.algorithms.cataldi.Cataldi`.
    The importance of terms is called nutrition.

.. note::
    In reality, the bounds are not really between -1 and 1, but between :math:`-x` and :math:`x`, where :math:`x` is the maximum nutrition of a term.
    To get the burst bounds between -1 and 1, term nutritions need to be bound between 0 and 1 (that is, :math:`x = 1`).
    This is the original implementation in the :class:`~queues.consumers.algorithms.eld_consumer.ELDConsumer`.

.. note::

    The implementation of :class:`~tdt.algorithms.eld.ELD` is based on the algorithm outlined in `ELD: Event TimeLine Detection -- A Participant-Based Approach to Tracking Events by Mamo et al. (2019) <https://dl.acm.org/doi/abs/10.1145/3342220.3344921>`_.
"""

import math
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from tdt.algorithms import TDTAlgorithm

class ELD(TDTAlgorithm):
    """
    Mamo et al.'s ELD is a feature-pivot TDT algorithm to detect breaking terms.
    The algorithm returns not only terms, but also the degree to which they are breaking.

    The algorithm receives one parameter apart from the :class:`~tdt.nutrition.NutritionStore`: the decay rate.
    The decay rate is used to penalize old checkpoints and give recent checkpoints more importance in the burst calculation.

    The keys of the :class:`~tdt.nutrition.NutritionStore` should be timestamps that represent checkpoints, or time windows.
    The checkpoint size depends on the application and how fast you expect the stream to change.
    Each timestamp should have a dictionary with the nutritions of terms in it; the keys are the terms and the values are the corresponding nutrition values.

    :ivar store: The store containing historical nutrition data.
                 The algorithm expects the timestamps to represent checkpoints.
                 Therefore the nutrition store should have dictionaries with timestamps as keys, and the nutrition of terms in a dictionary as values.
                 In other words, the timestamps should represent an entire checkpoint, not just a particular timestamp.
    :vartype store: :class:`~tdt.nutrition.store.NutritionStore`
    :ivar decay_rate: The decay rate used by the algorithm.
                      The larger the decay rate, the less importance far-off windows have in the burst calculation.
    :vartype decay_rate: float
    """

    def __init__(self, store, decay_rate=(1./2.)):
        """
        Instantiate the TDT algorithm with the :class:`~tdt.nutrition.NutritionStore` that will be used to detect topics and the decay rate.

        :param store: The store containing historical nutrition data.
                      The algorithm expects the timestamps to represent checkpoints.
                      Therefore the nutrition store should have dictionaries with timestamps as keys, and the nutrition of terms in a dictionary as values.
                      In other words, the timestamps should represent an entire checkpoint, not just a particular timestamp.
        :type store: :class:`~tdt.nutrition.store.NutritionStore`
        :param decay_rate: The decay rate used by the algorithm.
                           The larger the decay rate, the less importance far-off windows have in the burst calculation.
        :type decay_rate: float
        """

        self.store = store
        self.decay_rate = decay_rate

    def detect(self, nutrition, since=None, until=None, min_burst=0):
        """
        Detect topics using historical data from the given nutrition store.

        This function compares the nutrition of terms in the local context (the ``nutrition`` parameter) with the global context (the checkpoints in the class' :class:`~tdt.nutrition.NutritionStore`).
        By default, this function uses all checkpoints until the given time window.
        If no end timestamp (the ``until`` parameter) is given, the current timestamp is taken.

        Fewer checkpoints can be used by providing the ``since`` and ``until`` parameters.
        Old checkpoints have a smaller effect on the result than recent checkpoints so they can be removed with little effect.

        .. note::

            This function assumes that nutrition is always zero or positive.
            As a result, the burst can be calculated only for terms that have a nutrition equal to or greater than the minimum burst.
            The function makes an exception if the minimum burst is negative.
            In this case, all terms have to be considered in the calculation.

        .. note::

            The minimum burst is exclusive.
            This is so that terms with a burst of 0 (no change from previous checkpoints) are excluded.

        :param nutrition: The nutrition values of the local context.
                          The keys should be the terms, and the values the respective nutrition.
        :type nutrition: dict
        :param since: The timestamp since when nutrition should be considered.
                      If it is not given, all of the nutrition that is available until the ``until`` is used.
        :type since: float or None
        :param until: The timestamp until when nutrition should be considered.
                      If it is not given, all of the nutrition that is available since the ``since`` parameter is used.
                      If the algorithm is being used retrospectively, this parameter can represent the current timestamp to get only past nutrition.
        :type until: float or None
        :param min_burst: The minimum burst of a term to be considered emerging and returned.
                          This value is exclusive so that terms with an unchanging nutrition (a burst of 0) are not returned.
                          By default, only terms thet have a non-zero positive burst are returned.
                          These terms have seen their popularity increase.
        :type min_burst: float

        :return: The breaking terms and their burst as a dictionary.
                 The keys are the terms and the values are the respective burst values.
        :rtype: dict
        """

        """
        If no timestamp to begin with is provided, all nutrition from the earliest possible timestamp is used.
        If no end timestamp is provided, all nutrition is used.
        """
        since = since or 0

        """
        Load the historic nutrition data.
        The timestamp being evaluated is not used.
        """
        if until:
            historic = self.store.between(since, until)
        else:
            historic = self.store.since(since)

        terms = self._get_terms(min_burst, nutrition, historic)

        """
        Compute the burst of all the terms.
        Filter those with a low burst.
        """
        burst = { term: self._compute_burst(term, nutrition, historic) for term in terms }
        burst = { term: burst for term, burst in burst.items() if burst > min_burst }
        return burst

    def _get_terms(self, min_burst, nutrition, historic):
        """
        Get the terms for which to calculate the burst.
        If the minimum burst is not negative, only calculate the burst for the terms in the current time window.
        Furthermore, the burst can be calculated only for terms whose nutrition is equal to or greater than the minimum burst.

        Otherwise, get a list of all the terms in the historic data to find drops in burst.
        These terms are added to the new nutrition data.
        Burst is computed for all these terms.

        :param min_burst: The minimum burst of a term to be considered emerging and returned.
                          This value is exclusive.
                          By default, only terms thet have a non-zero positive burst are returned.
                          These terms have seen their popularity increase.
        :type min_burst: float
        :param historic: The historic nutrition values from the past time windows.
                         The keys should be the terms, and the values the respective nutrition.
        :type historic: dict
        :param nutrition: The nutrition values from the current (sliding) time window.
                          The keys should be the terms, and the values the respective nutrition.
        :type nutrition: dict
        """

        if min_burst >= 0:
            return [ term for term in nutrition if nutrition.get(term) >= min_burst ]
        else:
            terms = [ term for data in historic.values()
                            for term in data ]
            return set(list(nutrition.keys()) + terms)

    def _compute_burst(self, term, nutrition, historic):
        """
        Calculate the burst for the given term using the historical data.
        The equation used is:

        .. math::

            burst_k^t = \\frac{\\sum_{c=t-s}^{t-1}((nutr_{k,l} - nutr_{k,c}) \\cdot \\frac{1}{\\sqrt{e^{t - c}}})}{\\sum_{c=1}^s\\frac{1}{\\sqrt{e^c}}}

        where :math:`k` is the term for which burst is to be calculated.
        :math:`t` is the current time window and :math:`s` is the number of time windows to consider.
        :math:`nutr_{k,l}` is the nutrition of the term in the local context.
        This local context refers to a cluster since the broader ELD system combines document-pivot and feature-pivot techniques.
        :math:`nutr_{k,c}` is the nutrition of the term in the checkpoint :math:`c`.

        The denominator is the component that is responsible for binding the burst between 1 and -1.

        .. note::

            The time windows are between :math:`t-s` and :math:`t-1`
            The most recent time window is :math:`x = t-1`.
            The exponential decay's denominator would thus be 2.
            At :math:`x = t-2`, the denominator would be 3.
            Thus, the older time windows get less importance.

        :param term: The term whose burst is being calculated.
        :type term: str
        :param nutrition: The nutrition in the current time window.
                          The keys are the terms and the values are their nutritions.
        :type nutrition: dict
        :param historic: The historic data.
                         The keys are the timestamps of each time window.
                         The values are the nutritions of the time window—another dictionary.
                         The keys in the inner dictionary are the terms and the values are their nutritions.
        :type historic: dict

        :return: The term's burst.
        :rtype: float
        """

        """
        First calculate the numerator.
        The historic data is sorted in descending order.
        """
        historic = sorted(historic.items(), key=lambda data: data[0], reverse=True)
        historic = [ nutrition for timestamp, nutrition in historic ]
        burst = [ (nutrition.get(term, 0) - historic[c].get(term, 0)) * self._compute_decay(c + 1)
                  for c in range(len(historic)) ]

        """
        Calculate the denominator.
        """
        coefficient = self._compute_coefficient(len(historic))
        return sum(burst) / coefficient

    def _compute_decay(self, c):
        """
        Compute the decay with an exponential formula:

        .. math::

            x = \\frac{1}{(e^c)^d}

        where :math:`c` is the number of time windows being considered and :math:`d` is the decay rate.
        By default, the decay rate is :math:`\\frac{1}{2}`:

        .. math::

            x = \\frac{1}{\\sqrt{e^c}}

        :param c: The current time window.
        :type c: int

        :return: The exponential decay factor, or how much weight the burst of a term in a time window has.
        :rtype: float
        """

        return(1 / math.exp(c) ** self.decay_rate)

    def _compute_coefficient(self, s):
        """
        Get the denominator of the burst calculation.
        This denominator is used to rescale the function, for example with bounds between -1 and 1.

        :param s: The number of time windows being considered.
        :type s: int

        :return: The denominator of the burst calculation.
        :rtype: float

        :raises ValueError: When there is a negative number of time windows.
        """

        if s < 0:
            raise ValueError(f"The number of time windows cannot be negative: received {s}")

        """
        If there are no time windows, the co-efficient should be 1.
        """
        if not s:
            return 1
        else:
            return sum([ self._compute_decay(s + 1) for s in range(s) ])

class SlidingELD(ELD):
    """
    :class:`~tdt.algorithms.eld.SlidingELD` is a variation of :class:`~tdt.algorithms.eld.ELD` that uses a sliding window instead of checkpoints.
    This class creates the checkpoints every time the :class:`~tdt.algorithms.eld.SlidingELD` is called.

    In addition to the :class:`~tdt.nutrition.NutritionStore`, :class:`~tdt.algorithms.eld.SlidingELD` maintains in its state:

    - The window size in seconds, used to partition the historic nutrition into time windows.
    - The number of windows to use when detecting bursty terms.
      Since :class:`~tdt.algorithms.eld.SlidingELD` uses a decay mechanism, old time windows have little effect on the burst.
      Therefore the number of windows can be restrained.
    - A boolean indicating whether to normalize nutrition.
      Unlike :class:`~tdt.algorithms.eld.ELD`, since the algorithm uses a sliding time window, the nutrition cannot be normalized in advance.
      If nutrition is not normalized, or rescaled between 0 and 1, for each time window, burst cannot be interpreted because it is no longer bound between -1 and 1.
      If this flag is set to ``True``, the algorithm normalizes each time window so that burst is interpretable.

    :ivar window_size: The length in seconds of time windows.
    :vartype window_size: int
    :ivar windows: The number of windows to use when detecting bursty terms.
    :vartype windows: int
    :ivar normalized: A boolean indicating whether to normalize the nutrition in each time window when calculating the burst.
                       When nutrition is normalized, it is bound between 0 and 1.
                       This means that burst is then normalized between -1 and 1.
    :vartype normalized: bool
    """

    def __init__(self, store, decay_rate=(1./2.), window_size=60, windows=10, normalized=True):
        """
        Instantiate the TDT algorithm with the :class:`~tdt.nutrition.NutritionStore` that will be used to detect topics and the decay rate.

        :param store: The store containing historical nutrition data.
                      The algorithm expects the timestamps to represent checkpoints.
                      Therefore the nutrition store should have dictionaries with timestamps as keys, and the nutrition of terms in a dictionary as values.
                      In other words, the timestamps should represent an entire checkpoint, not just a particular timestamp.
        :type store: :class:`~tdt.nutrition.store.NutritionStore`
        :param decay_rate: The decay rate used by the algorithm.
                           The larger the decay rate, the less importance far-off windows have in the burst calculation.
        :type decay_rate: float
        :param window_size: The length in seconds of time windows.
        :type window_size: int
        :param windows: The number of windows to use when detecting bursty terms.
        :type windows: int
        :param normalized: A boolean indicating whether to normalize the nutrition in each time window when calculating the burst.
                           When nutrition is normalized, it is bound between 0 and 1.
                           This means that burst is then normalized between -1 and 1.
        :type normalized: bool

        :raises ValueError: When the window size is not a positive number.
        :raises ValueError: When the number of windows is not an integer.
        :raises ValueError: When the number of windows is not a positive number.
        """

        super(SlidingELD, self).__init__(store, decay_rate)

        if window_size < 1:
            raise ValueError(f"The window size should be positive; received { window_size }")

        if type(windows) is not int:
            raise ValueError(f"The number of windows should be an integer; received { type(windows) }")

        if windows < 1:
            raise ValueError(f"The number of windows should be a positive integer; received { windows }")

        self.window_size = window_size
        self.windows = windows
        self.normalized = normalized

    def detect(self, timestamp=None, min_burst=0):
        """
        Detect topics using historical data from the nutrition store.

        This function partitions the nutrition in the store into time windows.
        The most recent time window ends at the given ``timestamp``.
        The previous time windows are used as historic windows.

        .. note::

            This function assumes that nutrition is always zero or positive.
            As a result, the burst can be calculated only for terms that have a nutrition equal to or greater than the minimum burst.
            The function makes an exception if the minimum burst is negative.
            In this case, all terms have to be considered in the calculation.

        .. note::

            The minimum burst is exclusive.
            This is so that terms with a burst of 0 (no change from previous checkpoints) are excluded.

        .. warning::

            If there is no historic data yet, the function returns an empty set of terms.
            Otherwise, the function would be trivial and all terms would be considered to be bursty.

        :param timestamp: The timestamp at which to calculate the burst of terms.
                          If it is not given, the latest timestamp in the nutrition store is used.
        :type timestamp: float
        :param min_burst: The minimum burst of a term to be considered emerging and returned.
                          This value is exclusive so that terms with an unchanging nutrition (a burst of 0) are not returned.
                          By default, only terms thet have a non-zero positive burst are returned.
                          These terms have seen their popularity increase.
        :type min_burst: float

        :return: The breaking terms and their burst as a dictionary.
                 The keys are the terms and the values are the respective burst values.
        :rtype: dict
        """

        """
        If there is nutrition data in the nutrition store, return immediately.
        """
        if not self.store.all():
            return { }

        """
        Partition the nutrition data into time windows.
        """
        timestamp = timestamp or max(self.store.all().keys())
        nutrition, historic = self._partition(timestamp)

        """
        If there is no historic data, return immediately.
        Otherwise, all terms will be breaking.
        """
        if all(not( data ) for data in historic.values()):
            return { }

        """
        Normalize the time windows if need be.
        """
        if self.normalized:
            nutrition = self._normalize(nutrition)
            historic = { window: self._normalize(nutrition) for window, nutrition in historic.items() }

        """
        Get the terms for which to calculate the minimum burst.
        """
        terms = self._get_terms(min_burst, nutrition, historic)

        """
        Compute the burst of all the terms.
        Filter those with a low burst.
        """
        burst = { term: self._compute_burst(term, nutrition, historic) for term in terms }
        burst = { term: burst for term, burst in burst.items() if burst > min_burst }
        return burst

    def _partition(self, timestamp):
        """
        Partition the nutrition in the store into time windows.
        This function returns a tuple:

        1. The nutrition at the latest time window.
        2. The nutrition at the time windows preceding the latest one.

        The number of time windows, including the latest one, is at most equivalent to the number of time windows defined during instantiation.

        The historic nutrition is a dictionary, with the timestamps as keys and the nutrition data as the values.
        The timestamps indicate the end of the time window, not the start.
        Moreover, the end value is inclusive.

        :param timestamp: The timestamp at which to create the time windows.
        :type timestamp: float

        :return: A tuple, containing:

                  - The nutrition at the latest time window, and
                 - The nutrition at the previous time windows.
        :rtype: tuple of dict
        """

        # NOTE: In this function, the ``since`` is exclusive, and the ``until`` is inclusive.

        """
        Calculate the nutrition.
        """
        nutrition = self.store.between(timestamp - self.window_size + 1, timestamp + 1)
        nutrition = { t: values for t, values in nutrition.items()
                                 if t <= timestamp }
        nutrition = self._merge(*nutrition.values())

        """
        Calculate the historic nutrition.
        """
        historic = { }
        for window in range(1, self.windows):
            since = max(timestamp - self.window_size * (window + 1) + 1, 0)
            until = timestamp - self.window_size * window
            if until > 0:
                data = self.store.between(since, until + 1)
                data = { t: values for t, values in data.items()
                                    if t <= until }
                historic[until] = self._merge(*data.values())

        return (nutrition, historic)

    def _merge(self, *args):
        """
        Merge the given nutrition values.

        :return: One dictionary with the nutrition values added per term.
        :rtype: dict
        """

        nutrition = { }

        for data in args:
            for term, value in data.items():
                nutrition[term] = nutrition.get(term, 0) + value

        return nutrition

    def _normalize(self, window):
        """
        Normalize the given time window.
        This operation rescales the nutrition values of the time window to be between 0 and 1.

        :param window: A dictionary with the terms as keys and their nutrition value as values.
        :type window: dict

        :return: The normalized time window, with the maximum value set to 1.
        :rtype: dict
        """

        """
        Return immediately if the window is empty.
        """
        if not window or not max(window.values()):
            return { }

        max_nutrition = max(window.values())
        return { term: nutrition/max_nutrition for term, nutrition in window.items() }
