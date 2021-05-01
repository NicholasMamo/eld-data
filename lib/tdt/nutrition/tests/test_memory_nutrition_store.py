"""
Test the memory nutrition store.
"""

import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..')
if path not in sys.path:
    sys.path.append(path)

from memory import MemoryNutritionStore

class TestMemoryNutritionStore(unittest.TestCase):
    """
    Test the memory nutrition store.
    """

    def test_create_nutrition_store(self):
        """
        Test that when creating the nutrition store, it is an empty dictionary.
        """

        self.assertEqual({ }, MemoryNutritionStore().store)

    def test_add_nutrition_string(self):
        """
        Test that when adding nutrition data in a timestamp given as a string, it is stored.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.store)
        nutrition.add('10', { 'a': 1 })
        self.assertEqual({ 10: { 'a': 1 } }, nutrition.store)

    def test_add_nutrition_int(self):
        """
        Test that when adding nutrition data in a timestamp given as an integer, it is type-cast properly.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.store)
        nutrition.add(10, { 'a': 1 })
        self.assertEqual({ 10: { 'a': 1 } }, nutrition.store)

    def test_add_nutrition_float(self):
        """
        Test that when adding nutrition data in a timestamp given as a float, it is type-cast properly.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.store)
        nutrition.add(10.5, { 'a': 1 })
        self.assertEqual({ 10.5: { 'a': 1 } }, nutrition.store)

    def test_add_nutrition_arbitrary(self):
        """
        Test that the nutrition store handles any type of nutrition data.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.store)
        nutrition.add(123, 10)
        self.assertEqual({ 123: 10 }, nutrition.store)

    def test_add_multiple_nutrition(self):
        """
        Test that when adding nutrition data at multiple timestamps, all of them are stored.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.store)
        nutrition.add(10, { 'a': 1 })
        self.assertEqual({ 10: { 'a': 1 } }, nutrition.store)
        nutrition.add(20, { 'b': 2 })
        self.assertEqual({ 10: { 'a': 1 }, 20: { 'b': 2 } }, nutrition.store)

    def test_add_overwrite_string(self):
        """
        Test that when adding nutrition to an already-occupied timestamp with a string as timestamp, the old data is overwritten.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.store)
        nutrition.add(10, { 'a': 1 })
        self.assertEqual({ 10: { 'a': 1 } }, nutrition.store)
        nutrition.add('10', { 'b': 2 })
        self.assertEqual({ 10: { 'b': 2 } }, nutrition.store)

    def test_add_overwrite_int(self):
        """
        Test that when adding nutrition to an already-occupied timestamp with an integer as timestamp, the old data is overwritten.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.store)
        nutrition.add('10', { 'a': 1 })
        self.assertEqual({ 10: { 'a': 1 } }, nutrition.store)
        nutrition.add(10, { 'b': 2 })
        self.assertEqual({ 10: { 'b': 2 } }, nutrition.store)

    def test_add_overwrite_float(self):
        """
        Test that when adding nutrition to an already-occupied timestamp with a float as timestamp, the old data is overwritten.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.store)
        nutrition.add(10.5, { 'a': 1 })
        self.assertEqual({ 10.5: { 'a': 1 } }, nutrition.store)
        nutrition.add(10.5, { 'b': 2 })
        self.assertEqual({ 10.5: { 'b': 2 } }, nutrition.store)

    def test_get_nutrition_string(self):
        """
        Test getting nutrition data with a string as timestamp.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(10, { 'a': 1 })
        self.assertEqual({ 'a': 1 }, nutrition.get('10'))

    def test_get_nutrition_int(self):
        """
        Test getting nutrition data with an integer as timestamp.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(10, { 'a': 1 })
        self.assertEqual({ 'a': 1 }, nutrition.get(10))

    def test_get_nutrition_float(self):
        """
        Test getting nutrition data with a float as a timestamp.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(10.5, { 'a': 1 })
        self.assertEqual({ 'a': 1 }, nutrition.get(10.5))

    def test_get_missing_nutrition(self):
        """
        Test that when getting nutrition for a missing timestamp, ``None`` is returned.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual(None, nutrition.get(10))

    def test_all_nutrition(self):
        """
        Test that when getting all the nutrition data, all of it is returned.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 10: 1, 20: 2 }, nutrition.all())

    def test_all_dict(self):
        """
        Test that the return type when getting all nutrition data is a dictionary.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual(dict, type(nutrition.all()))

    def test_all_empty(self):
        """
        Test that an empty dictionary is returned when there is no data in the nutrition store.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.all())

    def test_between_start_after_than_end_string(self):
        """
        Test that when getting nutrition with the start timestamp being less than the end timestamp, a ValueError is raised.
        This test uses strings.
        """

        nutrition = MemoryNutritionStore()
        self.assertRaises(ValueError, nutrition.between, '10', '0')

    def test_between_start_same_as_end_string(self):
        """
        Test that when getting nutrition with the start timestamp being equivalent to the end timestamp, a ValueError is raised.
        This test uses strings.
        """

        nutrition = MemoryNutritionStore()
        self.assertRaises(ValueError, nutrition.between, '10', '10')

    def test_between_start_after_than_end_int(self):
        """
        Test that when getting nutrition with the start timestamp being less than the end timestamp, a ValueError is raised.
        This test uses integers.
        """

        nutrition = MemoryNutritionStore()
        self.assertRaises(ValueError, nutrition.between, 10, 0)

    def test_between_start_same_as_end_int(self):
        """
        Test that when getting nutrition with the start timestamp being equivalent to the end timestamp, a ValueError is raised.
        This test uses integers.
        """

        nutrition = MemoryNutritionStore()
        self.assertRaises(ValueError, nutrition.between, 10, 10)

    def test_between_start_after_than_end_float(self):
        """
        Test that when getting nutrition with the start timestamp being less than the end timestamp, a ValueError is raised.
        This test uses floats.
        """

        nutrition = MemoryNutritionStore()
        self.assertRaises(ValueError, nutrition.between, 10.5, 0.0)

    def test_between_start_same_as_end_float(self):
        """
        Test that when getting nutrition with the start timestamp being equivalent to the end timestamp, a ValueError is raised.
        This test uses floats.
        """

        nutrition = MemoryNutritionStore()
        self.assertRaises(ValueError, nutrition.between, 10.5, 10.5)

    def test_between_start_inclusive(self):
        """
        Test that the start timestamp is inclusive when getting nutrition data between two timestamps.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 10: 1}, nutrition.between(10, 15))

    def test_between_end_exclusive(self):
        """
        Test that the end timestamp is exclusive when getting nutrition data between two timestamps.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 10: 1}, nutrition.between(9, 20))

    def test_between_empty_result_dict(self):
        """
        Test that when getting the nutrition data matches nothing, an empty dictionary is returned.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.between(0, 10))

    def test_between(self):
        """
        Test getting nutrition data between two timestamps.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 10: 1}, nutrition.between(1, 19))

    def test_since_empty(self):
        """
        Test that when getting nutrition data since a timestamp from an empty store, an empty dictionary is returned.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.since(0))

    def test_since_inclusive(self):
        """
        Test that when getting nutrition data since a timestamp, the start timestamp is included.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertTrue(10 in nutrition.since(10))

    def test_since_int(self):
        """
        Test that when getting nutrition data since a timestamp with an integer, the correct data is returned.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 10: 1, 20: 2 }, nutrition.since(10))

    def test_since_float(self):
        """
        Test that when getting nutrition data since a timestamp with a float, the correct data is returned.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 10: 1, 20: 2 }, nutrition.since(10.0))

    def test_since_string(self):
        """
        Test that when getting nutrition data since a timestamp with a float, the correct data is returned.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 10: 1, 20: 2 }, nutrition.since('9'))
        self.assertTrue(all(float(timestamp) >= 9 for timestamp in nutrition.since(9)))

    def test_until_empty(self):
        """
        Test that when getting nutrition data until a timestamp from an empty store, an empty dictionary is returned.
        """

        nutrition = MemoryNutritionStore()
        self.assertEqual({ }, nutrition.until(10))

    def test_until_exclusive(self):
        """
        Test that when getting nutrition data until a timestamp, the end timestamp is excluded.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertFalse(10 in nutrition.until(10))

    def test_until_int(self):
        """
        Test that when getting nutrition data until a timestamp with an integer, the correct data is returned.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 0: 0 }, nutrition.until(10))

    def test_until_float(self):
        """
        Test that when getting nutrition data until a timestamp with a float, the correct data is returned.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 0: 0 }, nutrition.until(10.0))

    def test_until_string(self):
        """
        Test that when getting nutrition data until a timestamp with a string, the correct data is returned.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 0: 0, 10: 1 }, nutrition.until(20))
        self.assertTrue(all(float(timestamp) < 20 for timestamp in nutrition.until('20')))

    def test_remove_nothing(self):
        """
        Test that when no timestamps are provided, the nutrition data is unchanged.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 0: 0, 10: 1, 20: 2 }, nutrition.all())
        nutrition.remove()
        self.assertEqual({ 0: 0, 10: 1, 20: 2 }, nutrition.all())

    def test_remove_string(self):
        """
        Test that when removing nutrition from a single string timestamp, only that data is removed.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 0: 0, 10: 1, 20: 2 }, nutrition.all())
        nutrition.remove('10')
        self.assertEqual({ 0: 0, 20: 2 }, nutrition.all())

    def test_remove_int(self):
        """
        Test that when removing nutrition from a single integer timestamp, only that data is removed.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 0: 0, 10: 1, 20: 2 }, nutrition.all())
        nutrition.remove(10)
        self.assertEqual({ 0: 0, 20: 2 }, nutrition.all())

    def test_remove_float(self):
        """
        Test that when removing nutrition from a single float timestamp, that data is only remove if the values match.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10.0, 1)
        nutrition.add(20, 2)
        self.assertEqual({ 0: 0, 10.0: 1, 20: 2 }, nutrition.all())
        nutrition.remove(10.0)
        self.assertEqual({ 0: 0, 20: 2 }, nutrition.all())

    def test_remove_multiple(self):
        """
        Test removing multiple timestamps.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        nutrition.add(30, 3)
        self.assertEqual({ 0: 0, 10: 1, 20: 2, 30: 3 }, nutrition.all())
        nutrition.remove(10, 20)
        self.assertEqual({ 0: 0, 30: 3 }, nutrition.all())

    def test_remove_multiple_mixed(self):
        """
        Test removing multiple timestamps with mixed types.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        nutrition.add(30.0, 3)
        self.assertEqual({ 0: 0, 10: 1, 20: 2, 30.0: 3 }, nutrition.all())
        nutrition.remove('10', 20, 30.0)
        self.assertEqual({ 0: 0 }, nutrition.all())

    def test_remove_until(self):
        """
        Test removing timestamps that come until the given timestamp.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        nutrition.add(30, 3)
        self.assertEqual({ 0: 0, 10: 1, 20: 2, 30: 3 }, nutrition.all())
        nutrition.remove(*nutrition.until(20))
        self.assertEqual({ 20: 2, 30: 3}, nutrition.all())

    def test_remove_since(self):
        """
        Test removing timestamps that come since the given timestamp.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        nutrition.add(30, 3)
        self.assertEqual({ 0: 0, 10: 1, 20: 2, 30: 3 }, nutrition.all())
        nutrition.remove(*nutrition.since(20))
        self.assertEqual({ 0: 0, 10: 1}, nutrition.all())

    def test_copy(self):
        """
        Test that when creating a copy of the nutrition store, all the timestamps and data are the same.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        nutrition.add(30, 3)

        copy = nutrition.copy()
        self.assertEqual(0, copy.get(0))
        self.assertEqual(1, copy.get(10))
        self.assertEqual(2, copy.get(20))
        self.assertEqual(3, copy.get(30))

    def test_copy_edit(self):
        """
        Test that when editing a copy of the nutrition store, the original data is the same.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, 0)
        nutrition.add(10, 1)
        nutrition.add(20, 2)
        nutrition.add(30, 3)

        copy = nutrition.copy()
        copy.add(0, 1)
        copy.add(10, 2)
        copy.add(20, 3)
        copy.add(30, 4)

        self.assertEqual(0, nutrition.get(0))
        self.assertEqual(1, nutrition.get(10))
        self.assertEqual(2, nutrition.get(20))
        self.assertEqual(3, nutrition.get(30))

        self.assertEqual(1, copy.get(0))
        self.assertEqual(2, copy.get(10))
        self.assertEqual(3, copy.get(20))
        self.assertEqual(4, copy.get(30))

    def test_copy_deep(self):
        """
        Test that the copy of the nutrition store is deep.
        """

        nutrition = MemoryNutritionStore()
        nutrition.add(0, { 'value': 0 })
        nutrition.add(10, [ 'value', 1 ])

        copy = nutrition.copy()
        copy.add(0, { 'value': 1 })
        copy.add(10, [ 'value', 2 ])

        self.assertEqual({ 'value': 0 }, nutrition.get(0))
        self.assertEqual([ 'value', 1 ], nutrition.get(10))

        self.assertEqual({ 'value': 1 }, copy.get(0))
        self.assertEqual([ 'value', 2 ], copy.get(10))
