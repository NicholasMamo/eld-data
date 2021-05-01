"""
Run unit tests on the :class:`~objects.attributable.Attributable` class.
"""

import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..')
if path not in sys.path:
    sys.path.append(path)

from attributable import Attributable

class TestAttributable(unittest.TestCase):
    """
    Test the :class:`~objects.attributable.Attributable` class.
    """

    def test_create_empty(self):
        """
        Test that the empty attributable object has an empty dictionary.
        """

        self.assertEqual({ }, Attributable().attributes)

    def test_create_with_data(self):
        """
        Test that an attributable object accepts attributes in the constructor.
        """

        self.assertEqual({ 'a': 1 }, Attributable({ 'a': 1 }).attributes)
