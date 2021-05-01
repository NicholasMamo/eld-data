"""
Test the functionality of the queue.
"""

import os
import random
import re
import string
import sys
import unittest
import warnings

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from queues import Queue

class TestQueue(unittest.TestCase):
    """
    Test the implementation of the queue.
    """

    def test_empty_init(self):
        """
        Test that when creating a queue without any data, an empty list is generated instead.
        """

        queue = Queue()
        self.assertEqual([ ], queue.queue)

    def test_init_with_data(self):
        """
        Test that when a queue is created with data, all of it is saved.
        """

        queue = Queue(1)
        self.assertEqual([ 1 ], queue.queue)

    def test_init_with_multiple_data(self):
        """
        Test that when a queue is created with data, all of it is saved.
        """

        queue = Queue(1, True, 'a')
        self.assertEqual([ 1, True, 'a' ], queue.queue)

    def test_init_with_list_data(self):
        """
        Test that when creating a queue with a list, the list is saved as a single element.
        """

        queue = Queue([ 1, True, 'a' ])
        self.assertEqual([[ 1, True, 'a' ]], queue.queue)

    def test_enqueue_no_data(self):
        """
        Test that when enqueuing no data, the queue remains as it was in the beginning.
        """

        data = list(range(0, 10))
        queue = Queue(*data)
        self.assertEqual(data, queue.queue)
        queue.enqueue()
        self.assertEqual(data, queue.queue)

    def test_enqueue_data(self):
        """
        Test that when enqueuing a single element, it is enqueued as an element.
        """

        data = list(range(0, 10))
        queue = Queue(*data)
        self.assertEqual(data, queue.queue)
        queue.enqueue(11)
        self.assertEqual(data + [ 11 ], queue.queue)

    def test_enqueue_multiple_data(self):
        """
        Test that when enqueuing multiple elements, all of them are enqueued.
        """

        data = list(range(0, 10))
        queue = Queue(*data)
        self.assertEqual(data, queue.queue)
        queue.enqueue(*list(range(10, 12)))
        self.assertEqual(data + list(range(10, 12)), queue.queue)

    def test_enqueue_list(self):
        """
        Test that when enqueuing a list, the list itself is enqueued.
        """

        data = list(range(0, 10))
        queue = Queue(*data)
        self.assertEqual(data, queue.queue)
        queue.enqueue(list(range(10, 12)))
        self.assertEqual(data + [list(range(10, 12))], queue.queue)

    def test_dequeue_empty_queue(self):
        """
        Test that when dequeuing an empty queue, ``None`` is returned.
        """

        queue = Queue()
        self.assertEqual(None, queue.dequeue())

    def test_dequeue_only_element(self):
        """
        Test that when dequeuing the only element from a queue, the element is returned and the queue becomes empty.
        """

        queue = Queue(1)
        self.assertEqual(1, queue.dequeue())
        self.assertFalse(queue.length())

    def test_dequeue(self):
        """
        Test that when dequeuing an element, it is returned and removed.
        """

        queue = Queue(*range(0, 10))
        self.assertEqual(0, queue.dequeue())
        self.assertEqual(list(range(1, 10)), queue.queue)

    def test_dequeue_all_empty_queue(self):
        """
        Test when dequeuing an empty queue, an empty list is returned.
        """

        queue = Queue()
        self.assertEqual([ ], queue.dequeue_all())
        self.assertFalse(queue.length())

    def test_dequeue_all(self):
        """
        Test when dequeuing all of the queue's elements, they are returned and the queue is emptied.
        """

        queue = Queue(*list(range(0, 10)))
        self.assertEqual(10, queue.length())
        self.assertEqual(list(range(0, 10)), queue.dequeue_all())
        self.assertFalse(queue.length())

    def test_empty_empty_queue(self):
        """
        Test that emptying an empty queue changes nothing.
        """

        queue = Queue()
        self.assertFalse(queue.length())
        queue.empty()
        self.assertFalse(queue.length())
        self.assertEqual(list, type(queue.queue))

    def test_empty_queue(self):
        """
        Test that when emptying a queue, all of the elements are removed.
        """

        queue = Queue(*list(range(0, 10)))
        self.assertEqual(10, queue.length())
        queue.empty()
        self.assertFalse(queue.length())
        self.assertEqual([ ], queue.queue)

    def test_length_empty_queue(self):
        """
        Test that the length of an empty queue is 0.
        """

        queue = Queue()
        self.assertFalse(queue.length())

    def test_length_queue(self):
        """
        Test that the length of a queue is equal to the number of elements in it.
        """

        queue = Queue()
        self.assertFalse(queue.length())
        queue.enqueue(*list(range(0, 10)))
        self.assertEqual(10, queue.length())

    def test_length_queue_with_list(self):
        """
        Test that a queue with just a list as an element has a length of 1.
        """

        queue = Queue(list(range(0, 10)))
        self.assertEqual(1, queue.length())

    def test_head_empty_queue(self):
        """
        Test that when getting the head element of an empty queue, ``None`` is returned instead.
        """

        queue = Queue()
        self.assertEqual(None, queue.head())

    def test_head(self):
        """
        Test that getting the head element of a queue does not remove it.
        """

        queue = Queue(*list(range(0, 10)))
        self.assertEqual(10, queue.length())
        self.assertEqual(0, queue.head())
        self.assertEqual(10, queue.length())
        self.assertEqual(0, queue.head())
        self.assertEqual(10, queue.length())

    def test_tail_empty_queue(self):
        """
        Test that when getting the tail element of an empty queue, ``None`` is returned instead.
        """

        queue = Queue()
        self.assertEqual(None, queue.tail())

    def test_tail(self):
        """
        Test that getting the tail element of a queue does not remove it.
        """

        queue = Queue(*list(range(0, 10)))
        self.assertEqual(10, queue.length())
        self.assertEqual(9, queue.tail())
        self.assertEqual(10, queue.length())
        self.assertEqual(9, queue.tail())
        self.assertEqual(10, queue.length())
