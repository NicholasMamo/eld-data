"""
Run unit tests on Mamo et al. (2019)'s ELD algorithm.
"""

import math
import os
import random
import string
import sys
import time
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from algorithms import ELD, SlidingELD
from nutrition.memory import MemoryNutritionStore

class TestELD(unittest.TestCase):
    """
    Test Mamo et al. (2019)'s ELD algorithm.
    """

    def test_detect(self):
        """
        Test detecting bursty terms.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 0.67, 'b': 1, 'c': 1 })
        store.add(50, { 'a': 0.67, 'b': 0.33, 'c': 0.3 })
        store.add(40, { 'a': 1, 'b': 0.67, 'c': 0.67 })
        self.assertEqual(set([ 'c', 'b' ]), set(algo.detect(store.get(60), until=60)))

    def test_detect_recency(self):
        """
        Test that when detecting bursty terms, recent time windows have more weight.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1, 'b': 1, 'c': 1 })
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.67 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        terms = algo.detect(store.get(60), until=60)
        self.assertGreater(terms.get('b'), terms.get('c'))

    def test_detect_since_inclusive(self):
        """
        Test that when detecting bursty terms with a selection of time windows, only those time windows are used.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1, 'b': 1, 'c': 1 })
        store.add(50, { 'a': 0.67, 'b': 1, 'c': 1 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 1 })
        self.assertEqual(set([ 'a' ]), set(algo.detect(store.get(60), since=50, until=60)))
        self.assertEqual(set([ 'a', 'b' ]), set(algo.detect(store.get(60), since=40, until=60)))

    def test_detect_nutrition_store_unchanged(self):
        """
        Test that when detecting bursty terms, the store itself is unchanged.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1, 'b': 1, 'c': 1 })
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        store_copy = dict(store.all())
        algo.detect(store.get(60))
        self.assertEqual(store_copy, store.all())

    def test_detect_empty_nutrition(self):
        """
        Test that when detecting bursty terms with an empty nutrition, no terms are returned.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1, 'b': 1, 'c': 1 })
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        self.assertFalse(algo.detect({ }))

    def test_detect_empty_historic(self):
        """
        Test that when detecting bursty terms with empty historic data, no terms are returned.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 0.95, 'b': 0.75, 'c': 1 })
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        self.assertEqual(set([ ]), set(algo.detect(store.get(60), until=40)))

    def test_detect_all_terms(self):
        """
        Test that when detecting bursty terms, the data is taken from both the historic data and the nutrition.
        For this test, the minimum burst is set such that it includes all burst values.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        store.add(30, { 'a': 1.00, 'b': 0.75 })

        self.assertEqual(set([ 'a', 'b', 'd' ]), set(algo.detect({ 'd': 1.00 }, until=40, min_burst=-1.1)))
        self.assertEqual(set([ 'a', 'b', 'c', 'd' ]), set(algo.detect({ 'd': 1.00 }, until=60, min_burst=-1.1)))
        self.assertEqual(set([ 'a', 'b', 'c', 'd' ]), set(algo.detect({ 'd': 1.00 }, until=60, min_burst=-1.)))
        self.assertEqual(set([ 'd' ]), set(algo.detect({ 'd': 1.00 }, until=60)))

    def test_get_terms_negative_minimum_burst(self):
        """
        Test that when detecting bursty terms, all terms are returned when the minimum burst is negative.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        store.add(30, { 'a': 1.00, 'b': 0.75 })

        nutrition = { 'd': 1.00 }
        historic = store.until(60)
        self.assertEqual(set(algo._get_terms(-0.1, nutrition, historic)), { 'a', 'b', 'c', 'd' })

    def test_get_terms_zero_minimum_burst(self):
        """
        Test that when detecting bursty terms, the data is only taken from the present nutrition when the minimum burst is zero.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        store.add(30, { 'a': 1.00, 'b': 0.75 })

        nutrition = { 'd': 1.00 }
        historic = store.until(60)
        self.assertEqual(algo._get_terms(0, nutrition, historic), [ 'd' ])

    def test_get_terms_positive_minimum_burst(self):
        """
        Test that when detecting bursty terms, the data is only taken from the present nutrition when the minimum burst is positive.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        store.add(30, { 'a': 1.00, 'b': 0.75 })

        nutrition = { 'd': 1.00 }
        historic = store.until(60)
        self.assertEqual(algo._get_terms(0.5, nutrition, historic), [ 'd' ])

    def test_get_terms_positive_minimum_burst_filter(self):
        """
        Test that when detecting bursty terms, the data is only taken from the present nutrition when the minimum burst is positive.
        Moreover, the terms should be filtered based on the minimum burst.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(50, { 'a': 0.67, 'b': 0.3, 'c': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67, 'c': 0.3 })
        store.add(30, { 'a': 1.00, 'b': 0.75 })

        nutrition = { 'd': 1.00, 'e': 0.5 }
        historic = store.until(60)
        self.assertEqual(algo._get_terms(0.5, nutrition, historic), [ 'd', 'e' ])
        self.assertEqual(algo._get_terms(0.6, nutrition, historic), [ 'd' ])

    def test_compute_burst_non_existent_term(self):
        """
        Test that when computing the burst of a term that does not exist, 0 is returned.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 0 })
        store.add(50, { 'a': 0 })
        store.add(40, { 'a': 0 })
        self.assertEqual(0, algo._compute_burst('d', store.get(60), store.until(60)))

    def test_compute_burst_term_zero_nutrition(self):
        """
        Test that when computing the burst of a term that has a nutrition of 0, 0 is returned.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 0 })
        store.add(50, { 'a': 0 })
        store.add(40, { 'a': 0 })
        self.assertEqual(0, algo._compute_burst('a', store.get(60), store.until(60)))

    def test_compute_burst_empty_historic(self):
        """
        Test that when computing the burst when the historic data is empty, 0 is returned.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 0 })
        self.assertEqual(0, algo._compute_burst('a', store.get(60), store.until(60)))

    def test_compute_burst_recency(self):
        """
        Test that when computing burst, recent historical data has more importance.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1, 'b': 1 })
        store.add(50, { 'a': 0.67, 'b': 0.33 })
        store.add(40, { 'a': 0.33, 'b': 0.67 })
        self.assertGreater(algo._compute_burst('b', store.get(60), store.until(60)),
                           algo._compute_burst('a', store.get(60), store.until(60)))

    def test_compute_burst(self):
        """
        Test the burst computation.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1 })
        store.add(50, { 'a': 0.67 })
        store.add(40, { 'a': 0.33 })

        """
        Formula: ((1 - 0.67) 1/(sqrt(e^1)) + (1 - 0.33) 1/(sqrt(e^2)))/(1/sqrt(e^1) + 1/sqrt(e^2))
        """
        self.assertEqual(round(0.458363827391369, 10),
                         round(algo._compute_burst('a', store.get(60), store.until(60)), 10))

    def test_compute_burst_unchanged_nutrition(self):
        """
        Test that when giving a dictionary of nutrition to compute the drops, the dictionary is unchanged.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1 })
        store.add(50, { 'a': 0.67 })
        store.add(40, { 'a': 0.33 })
        nutrition = store.get(60)
        historic = store.until(60)
        nutrition_copy = dict(nutrition)
        algo._compute_burst('a', nutrition, historic)
        self.assertEqual(nutrition_copy, nutrition)

    def test_compute_burst_unchanged_historical(self):
        """
        Test that when giving a dictionary of historic nutrition to compute the drops, the dictionary is unchanged.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1 })
        store.add(50, { 'a': 0.67 })
        store.add(40, { 'a': 0.33 })
        nutrition = store.get(60)
        historic = store.until(60)
        historic_copy = dict(historic)
        algo._compute_burst('a', nutrition, historic)
        self.assertEqual(historic_copy, historic)

    def test_compute_burst_upper_bound(self):
        """
        Test that the upper bound of the burst is 1 when the maximum nutrition is 1.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1 })
        store.add(50, { 'a': 0 })
        store.add(40, { 'a': 0 })
        self.assertEqual(1, algo._compute_burst('a', store.get(60), store.until(60)))

    def test_compute_burst_lower_bound(self):
        """
        Test that the lower bound of the burst is -1 when the maximum nutrition is 1.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 0 })
        store.add(50, { 'a': 1 })
        store.add(40, { 'a': 1 })
        self.assertEqual(-1, algo._compute_burst('a', store.get(60), store.until(60)))

    def test_compute_burst_unchanged(self):
        """
        Test that when the nutrition is unchanged, the burst is 0.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        store.add(60, { 'a': 1 })
        store.add(50, { 'a': 1 })
        store.add(40, { 'a': 1 })
        self.assertEqual(0, algo._compute_burst('a', store.get(60), store.until(60)))

    def test_compute_decay(self):
        """
        Test the decay computation with multiple time windows.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        self.assertEqual(round(0.2231301601484298, 10),
            round(algo._compute_decay(3), 10))

    def test_compute_decay_custom_rate(self):
        """
        Test that when the decay rate is not default, it is used.
        """

        store = MemoryNutritionStore()
        algo = ELD(store, 1./3.)
        self.assertEqual(round(0.3678794411714424, 10),
            round(algo._compute_decay(3), 10))

    def test_compute_coefficient_negative(self):
        """
        Test that the coefficient computation with negative time windows raises a ValueError.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        self.assertRaises(ValueError, algo._compute_coefficient, -1)

    def test_compute_coefficient_zero(self):
        """
        Test that the coefficient computation with no time windows equals 1.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        self.assertEqual(1, algo._compute_coefficient(0))

    def test_compute_coefficient_one(self):
        """
        Test the coefficient computation with one time window.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        self.assertEqual(round(0.6065306597126334, 10),
                         round(algo._compute_coefficient(1), 10))

    def test_compute_coefficient_multiple(self):
        """
        Test the coefficient computation with multiple time windows.
        """

        store = MemoryNutritionStore()
        algo = ELD(store)
        self.assertEqual(round(1.1975402610325056, 10),
                         round(algo._compute_coefficient(3), 10))

class TestSlidingELD(unittest.TestCase):
    """
    Test the sliding window variant of Mamo et al. (2019)'s ELD algorithm.
    """

    def test_init_store(self):
        """
        Test that when initializing the class, the nutrition store is saved properly.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store)
        self.assertEqual(store, algo.store)

    def test_init_decay_rate(self):
        """
        Test that when initializing the class, the decay rate is saved properly.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, decay_rate=0.2)
        self.assertEqual(0.2, algo.decay_rate)

    def test_init_window_size(self):
        """
        Test that when initializing the class, the window size is saved properly.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, window_size=120)
        self.assertEqual(120, algo.window_size)

    def test_init_window_size_negative(self):
        """
        Test that when initializing the class with a negative window size, the class raises a ValueError.
        """

        store = MemoryNutritionStore()
        self.assertRaises(ValueError, SlidingELD, store, window_size=-1)

    def test_init_window_size_zero(self):
        """
        Test that when initializing the class with a window size of zero, the class raises a ValueError.
        """

        store = MemoryNutritionStore()
        self.assertRaises(ValueError, SlidingELD, store, window_size=0)

    def test_init_window_size_positive(self):
        """
        Test that when initializing the class with a positive window size, the class accepts it.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, window_size=1)
        self.assertTrue(algo)

    def test_init_windows(self):
        """
        Test that when initializing the class, the number of windows is saved properly.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=5)
        self.assertEqual(5, algo.windows)

    def test_init_windows_float(self):
        """
        Test that when initializing the class with a float as the number of windows, the class raises a ValueError.
        """

        store = MemoryNutritionStore()
        self.assertRaises(ValueError, SlidingELD, store, windows=1.2)

    def test_init_windows_str(self):
        """
        Test that when initializing the class with a string as the number of windows, the class raises a ValueError.
        """

        store = MemoryNutritionStore()
        self.assertRaises(ValueError, SlidingELD, store, windows='5')

    def test_init_windows_negative(self):
        """
        Test that when initializing the class with a negative number as the number of windows, the class raises a ValueError.
        """

        store = MemoryNutritionStore()
        self.assertRaises(ValueError, SlidingELD, store, windows=-1)

    def test_init_windows_zero(self):
        """
        Test that when initializing the class with zero as the number of windows, the class raises a ValueError.
        """

        store = MemoryNutritionStore()
        self.assertRaises(ValueError, SlidingELD, store, windows=0)

    def test_init_windows_one(self):
        """
        Test that when initializing the class with 1 as the number of windows, the class accepts it.
        """

        store = MemoryNutritionStore()
        self.assertTrue(SlidingELD(store, windows=1))

    def test_init_normalized(self):
        """
        Test that when initializing the class, the normalization boolean is saved properly.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, normalized=False)
        self.assertEqual(False, algo.normalized)

    def test_detect(self):
        """
        Test detecting bursty terms.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=6, window_size=10)
        store.add(60, { 'a': 30, 'b': 15 })
        store.add(55, { 'a': 20, 'b': 20 })
        store.add(50, { 'a': 5, 'b': 30 })
        store.add(45, { 'a': 10, 'b': 30 })
        store.add(40, { 'a': 20, 'b': 5 })
        store.add(35, { 'a': 5, 'b': 5 })
        store.add(30, { 'a': 5, 'b': 5 })
        store.add(25, { 'a': 5, 'b': 5 })
        store.add(20, { 'a': 5, 'b': 5 })
        store.add(15, { 'a': 5, 'b': 5 })
        store.add(10, { 'a': 5, 'b': 5 })
        store.add(5, { 'a': 5, 'b': 5 })
        store.add(0, { 'a': 5, 'b': 5 })
        self.assertEqual({ 'a' }, set(algo.detect(timestamp=60)))
        self.assertEqual({ 'b' }, set(algo.detect(timestamp=50)))

    def test_detect_timestamp_inclusive(self):
        """
        Test that the given timestamp is inclusive.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=6, window_size=10)
        store.add(60, { 'a': 30, 'b': 15 })
        store.add(55, { 'a': 20, 'b': 20 })
        store.add(50, { 'a': 20, 'b': 30 })
        store.add(45, { 'a': 30, 'b': 30 })
        store.add(40, { 'a': 20, 'b': 5 })
        store.add(35, { 'a': 5, 'b': 5 })
        store.add(30, { 'a': 5, 'b': 5 })
        store.add(25, { 'a': 5, 'b': 5 })
        store.add(20, { 'a': 5, 'b': 5 })
        store.add(15, { 'a': 5, 'b': 5 })
        store.add(10, { 'a': 5, 'b': 5 })
        store.add(5, { 'a': 5, 'b': 5 })
        store.add(0, { 'a': 5, 'b': 5 })

        """
        There is a burst at 60, but not one second earlier.
        """
        bursty = algo.detect(timestamp=60)
        self.assertTrue('a' in bursty)
        bursty = algo.detect(timestamp=59)
        self.assertFalse('a' in bursty)

    def test_detect_recency(self):
        """
        Test that when detecting bursty terms, recent time windows have more weight.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=6, window_size=10)
        store.add(60, { 'a': 30, 'b': 30 })
        store.add(55, { 'a': 30, 'b': 30 })
        store.add(50, { 'a': 20, 'b': 10 })
        store.add(45, { 'a': 20, 'b': 10 })
        store.add(40, { 'a': 10, 'b': 20 })
        store.add(35, { 'a': 10, 'b': 20 })
        bursty = algo.detect(timestamp=60)
        self.assertGreater(bursty.get('b'), bursty.get('a'))

    def test_detect_nutrition_store_unchanged(self):
        """
        Test that when detecting bursty terms, the store itself is unchanged.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=6, window_size=10)
        store.add(60, { 'a': 30, 'b': 30 })
        store.add(55, { 'a': 30, 'b': 30 })
        store.add(50, { 'a': 20, 'b': 10 })
        store.add(45, { 'a': 20, 'b': 10 })
        store.add(40, { 'a': 10, 'b': 20 })
        store.add(35, { 'a': 10, 'b': 20 })
        store_copy = dict(store.all())
        algo.detect(timestamp=60)
        self.assertEqual(store_copy, store.all())

    def test_detect_empty_nutrition(self):
        """
        Test that when detecting bursty terms with no empty nutrition, no terms are returned.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=6, window_size=10)
        store.add(60, { 'a': 30, 'b': 30 })
        store.add(55, { 'a': 30, 'b': 30 })
        store.add(50, { 'a': 20, 'b': 10 })
        store.add(45, { 'a': 20, 'b': 10 })
        store.add(40, { 'a': 10, 'b': 20 })
        store.add(35, { 'a': 10, 'b': 20 })
        nutrition, historic = algo._partition(timestamp=70)
        self.assertEqual({ }, nutrition)
        self.assertFalse(algo.detect(timestamp=70))

    def test_detect_empty_historic(self):
        """
        Test that when detecting bursty terms with empty historic data, no terms are returned.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=6, window_size=15)
        store.add(60, { 'a': 30, 'b': 30 })
        store.add(55, { 'a': 30, 'b': 30 })
        store.add(50, { 'a': 20, 'b': 10 })
        nutrition, historic = algo._partition(timestamp=60)
        self.assertTrue(all( { } == values for values in historic.values() ))
        self.assertFalse(algo.detect(timestamp=60))

        algo = SlidingELD(store, windows=6, window_size=10)
        nutrition, historic = algo._partition(timestamp=60)
        self.assertTrue(any( not { } == values for values in historic.values() ))
        self.assertEqual({ 'a', 'b' }, set(algo.detect(timestamp=60)))

    def test_detect_all_terms(self):
        """
        Test that when detecting bursty terms, the data is taken from both the historic data and the nutrition.
        For this test, the minimum burst is set such that it includes all burst values.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=3, window_size=5)
        store.add(60, { 'a': 50, 'b': 0, 'c': 50 })
        store.add(55, { 'a': 50, 'b': 50, 'c': 50 })
        store.add(50, { 'a': 25, 'b': 50, 'c': 50 })

        self.assertEqual({ 'a' }, set(algo.detect(timestamp=60)))
        burst = algo.detect(timestamp=60, min_burst=-1.1)
        self.assertEqual({ 'a', 'b', 'c' }, set(burst))
        self.assertTrue(burst['a'] > 0)
        self.assertTrue(burst['b'] < 0)
        self.assertTrue(burst['c'] == 0)

    def test_detect_lower_bound(self):
        """
        Test that when detecting bursty terms, the lower bound is -1.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=2, window_size=5)
        store.add(60, { 'a': 1, 'b': 0, 'c': 1 })
        store.add(55, { 'a': 0, 'b': 1, 'c': 1 })
        store.add(50, { 'a': 0, 'b': 1, 'c': 1 })

        terms = algo.detect(timestamp=60, min_burst=-1.1)
        self.assertEqual(-1, min(terms.values()))
        self.assertTrue(all( burst >= -1 for burst in terms.values() ))

    def test_detect_upper_bound(self):
        """
        Test that when detecting bursty terms, the upper bound is 1.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=3, window_size=5)
        store.add(60, { 'a': 1, 'b': 0, 'c': 1 })
        store.add(55, { 'a': 0, 'b': 1, 'c': 1 })
        store.add(50, { 'a': 0, 'b': 1, 'c': 1 })

        terms = algo.detect(timestamp=60, min_burst=-1.1)
        self.assertEqual(1, max(terms.values()))
        self.assertTrue(all( burst <= 1 for burst in terms.values() ))

    def test_detect_bounds_not_normalized(self):
        """
        Test that when detecting bursty terms without normalization, the bounds change.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=3, window_size=5, normalized=False)
        store.add(60, { 'a': 50, 'b': 0, 'c': 50 })
        store.add(55, { 'a': 0, 'b': 50, 'c': 50 })
        store.add(50, { 'a': 0, 'b': 50, 'c': 50 })

        terms = algo.detect(timestamp=60, min_burst=-51)
        self.assertEqual(50, terms['a'])
        self.assertEqual(-50, terms['b'])
        self.assertEqual(0, terms['c'])
        self.assertTrue(all( burst <= 50 for burst in terms.values() ))
        self.assertTrue(all( burst >= -50 for burst in terms.values() ))

    def test_partition(self):
        """
        Test partitioning with an example.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=4, window_size=5)
        store.add(60, { 'a': 50, 'b': 0 })
        store.add(55, { 'a': 50, 'b': 50 })
        store.add(50, { 'a': 25, 'b': 0 })

        """
        Use the latest data.
        """
        nutrition, historic = algo._partition(timestamp=60)
        self.assertEqual({ 'a': 50, 'b': 0 }, nutrition)
        self.assertEqual({ 55, 50, 45 }, set(historic.keys()))
        self.assertEqual({ 'a': 50, 'b': 50 }, historic[55])
        self.assertEqual({ 'a': 25, 'b': 0 }, historic[50])
        self.assertEqual({ }, historic[45])

        """
        Move one time window back.
        """
        nutrition, historic = algo._partition(timestamp=55)
        self.assertEqual({ 'a': 50, 'b': 50 }, nutrition)
        self.assertEqual({ 50, 45, 40 }, set(historic.keys()))
        self.assertEqual({ 'a': 25, 'b': 0 }, historic[50])
        self.assertEqual({ }, historic[45])
        self.assertEqual({ }, historic[40])

        """
        Use a larger time window.
        """
        algo = SlidingELD(store, windows=4, window_size=10)
        nutrition, historic = algo._partition(timestamp=60)
        self.assertEqual({ 'a': 100, 'b': 50 }, nutrition)
        self.assertEqual({ 50, 40, 30 }, set(historic.keys()))
        self.assertEqual({ 'a': 25, 'b': 0 }, historic[50])
        self.assertEqual({ }, historic[40])
        self.assertEqual({ }, historic[30])

    def test_partition_empty(self):
        """
        Test that when partitioning an empty nutrition store, the correct keys are set, but all nutrition values are empty.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, windows=10, window_size=60)
        nutrition, historic = algo._partition(600)
        self.assertEqual({ }, nutrition)
        self.assertEqual(set(range(60, 600, 60)), set(historic.keys()))
        self.assertTrue(all( { } == window for window in historic.values() ))

    def test_partition_nutrition_include_timestamp(self):
        """
        Test that when partitioning, the end timestamp of the nutrition is included.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, window_size=2)
        store.add(10, { 'a': 4 })
        store.add(9, { 'a': 2 })
        nutrition, _ = algo._partition(10)
        self.assertEqual({ 'a': 6 }, nutrition)

    def test_partition_nutrition_exclude_timestamp(self):
        """
        Test that when partitioning, the start of the time window of the nutrition for the given timestamp is excluded.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, window_size=2)
        store.add(10, { 'a': 4 })
        store.add(9, { 'a': 2 })
        store.add(8, { 'a': 1 })
        nutrition, _ = algo._partition(10)
        self.assertEqual({ 'a': 6 }, nutrition)

    def test_partition_historic_until_inclusive(self):
        """
        Test that when partitioning, the ``until`` timestamp is inclusive in the nutrition and historic data.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, window_size=2, windows=3)
        store.add(10, { 'a': 4 })
        store.add(9, { 'a': 2 })
        store.add(8, { 'a': 1 })
        store.add(7, { 'b': 1 })
        store.add(6, { 'b': 3 })
        store.add(5, { 'b': 5 })
        store.add(4, { 'b': 7 })
        store.add(3, { 'a': 2, 'b': 3 })
        nutrition, historic = algo._partition(10)
        self.assertEqual({ 'a': 6 }, nutrition)
        self.assertEqual({ 8, 6 }, set(historic.keys()))
        self.assertEqual({ 'a': 1, 'b': 1 }, historic[8])
        self.assertEqual({ 'b': 8 }, historic[6])

    def test_partition_historic_excludes_negative_windows(self):
        """
        Test that when partitioning, the negative-timestamped time windows are excluded.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store, window_size=4, windows=3)
        store.add(10, { 'a': 4 })
        store.add(9, { 'a': 2 })
        store.add(8, { 'a': 1 })
        store.add(7, { 'b': 1 })
        store.add(6, { 'b': 3 })
        store.add(5, { 'b': 5 })
        store.add(4, { 'b': 7 })
        store.add(3, { 'a': 2, 'b': 3 })
        store.add(2, { 'a': 2, 'b': 3 })
        nutrition, historic = algo._partition(10)
        self.assertEqual({ 6, 2 }, set(historic.keys()))
        self.assertEqual({ 'a': 2, 'b': 18 }, historic[6])
        self.assertEqual({ 'a': 2, 'b': 3 }, historic[2])

    def test_partition_time_windows(self):
        """
        Test that when partitioning, the correct number of time windows is used.
        """

        windows = 3

        store = MemoryNutritionStore()
        algo = SlidingELD(store, window_size=2, windows=windows)
        store.add(10, { 'a': 4 })
        store.add(9, { 'a': 2 })
        store.add(8, { 'a': 1 })
        store.add(7, { 'b': 1 })
        store.add(6, { 'b': 3 })
        store.add(5, { 'b': 5 })
        store.add(4, { 'b': 7 })
        store.add(3, { 'a': 2, 'b': 3 })
        nutrition, historic = algo._partition(10)
        self.assertEqual(2, len(historic))

    def test_merge_empty(self):
        """
        Test that when merging an empty list of nutrition data, the function returns an empty set.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store)
        self.assertEqual({ }, algo._merge())

    def test_merge_empty(self):
        """
        Test that when merging an empty list of nutrition data, the function returns an empty set.
        """

        store = MemoryNutritionStore()
        store.add(1, { 'a': 1, 'b': 2 })
        store.add(2, { 'a': 2 })
        store.add(3, { 'b': 0.5 })
        algo = SlidingELD(store)
        self.assertEqual({ 'a': 3, 'b': 2.5 }, algo._merge(*store.all().values()))

    def test_normalize_empty(self):
        """
        Test that when normalizing an empty dictionary, the function returns an empty dictionary.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store)
        self.assertEqual({ }, algo._normalize(store.get(10)))

    def test_normalize_max_zero(self):
        """
        Test that when normalizing a dictionary that is not empty, but where all terms have a nutrition of 0, the function returns an empty dictionary.
        """

        store = MemoryNutritionStore()
        algo = SlidingELD(store)
        store.add(10, { 'a': 0, 'b': 0 })
        self.assertEqual({ }, algo._normalize(store.get(10)))

    def test_normalize_lower_bound(self):
        """
        Test that when normalizing nutrition, all new values have a value greater than zero.
        """

        store = MemoryNutritionStore()
        store.add(10, { 'a': 1, 'b': 2, 'c': 3 })
        algo = SlidingELD(store)
        self.assertTrue(all( 0 < nutrition for nutrition in algo._normalize(store.get(10)).values() ))

    def test_normalize_upper_bound(self):
        """
        Test that when normalizing nutrition, all new values have a value less than or equal to one.
        """

        store = MemoryNutritionStore()
        store.add(10, { 'a': 1, 'b': 2, 'c': 3 })
        algo = SlidingELD(store)
        self.assertTrue(all( 1 >= nutrition for nutrition in algo._normalize(store.get(10)).values() ))

    def test_normalize_maximum_exists(self):
        """
        Test that when normalizing nutrition, there is at least one value with a value of one.
        """

        store = MemoryNutritionStore()
        store.add(10, { 'a': 1, 'b': 2, 'c': 3 })
        algo = SlidingELD(store)
        self.assertTrue(1 in algo._normalize(store.get(10)).values())

    def test_normalize_intermediate_values(self):
        """
        Test that when normalizing nutrition, all values are normalized properly.
        """

        store = MemoryNutritionStore()
        store.add(10, { 'a': 1, 'b': 2, 'c': 3 })
        algo = SlidingELD(store)
        normalized = algo._normalize(store.get(10))
        self.assertEqual(1/3, normalized['a'])
        self.assertEqual(2/3, normalized['b'])
        self.assertEqual(3/3, normalized['c'])

    def test_normalize_all_terms(self):
        """
        Test that when normalizing nutrition, all terms are returned.
        """

        store = MemoryNutritionStore()
        store.add(10, { 'a': 1, 'b': 2, 'c': 3 })
        algo = SlidingELD(store)
        original = store.get(10)
        normalized = algo._normalize(original)
        self.assertEqual(set(original.keys()), set(normalized.keys()))
