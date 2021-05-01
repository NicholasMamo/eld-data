"""
Test the functionality of the simulated file reader.
"""

import asyncio
import json
import os
import sys
import time
import unittest

from datetime import datetime
from tweepy import OAuthHandler, Stream

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from queues import Queue
from twitter import *
from twitter.file.simulated_reader import SimulatedFileReader

class TestSimulatedFileReader(unittest.IsolatedAsyncioTestCase):
    """
    Test the functionality of the simulated file reader.
    """

    def test_positive_speed(self):
        """
        Test that when creating a simulated file reader with a positive speed, no ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertTrue(SimulatedFileReader(Queue(), f, speed=1))

    def test_zero_speed(self):
        """
        Test that when creating a simulated file reader with a speed of zero, a ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertRaises(ValueError, SimulatedFileReader, Queue(), f, speed=0)

    def test_negative_speed(self):
        """
        Test that when creating a simulated file reader with a negative speed, a ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertRaises(ValueError, SimulatedFileReader, Queue(), f, speed=-1)

    def test_floating_point_skip_lines(self):
        """
        Test that when creating a simulated file reader with a floating point number of lines to skip, a ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertRaises(ValueError, SimulatedFileReader, Queue(), f, skip_lines=0.1)

    def test_float_skip_lines(self):
        """
        Test that when creating a simulated file reader with a rounded float number of lines to skip, no ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertTrue(SimulatedFileReader(Queue(), f, skip_lines=1.0))

    def test_integer_skip_lines(self):
        """
        Test that when creating a simulated file reader with an integer number of lines to skip, no ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertTrue(SimulatedFileReader(Queue(), f, skip_lines=1))

    def test_negative_skip_lines(self):
        """
        Test that when creating a simulated file reader with a negative number of lines to skip, a ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertRaises(ValueError, SimulatedFileReader, Queue(), f, skip_lines=-1)

    def test_zero_skip_lines(self):
        """
        Test that when creating a simulated file reader that skips no lines, no ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertTrue(SimulatedFileReader(Queue(), f, skip_lines=0))

    def test_positive_skip_lines(self):
        """
        Test that when creating a simulated file reader that skips a positive number of lines, no ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertTrue(SimulatedFileReader(Queue(), f, skip_lines=1))

    def test_negative_skip_time(self):
        """
        Test that when creating a simulated file reader with a negative number of seconds to skip, a ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertRaises(ValueError, SimulatedFileReader, Queue(), f, skip_time=-1)

    def test_zero_skip_time(self):
        """
        Test that when creating a simulated file reader that skips no time, no ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertTrue(SimulatedFileReader(Queue(), f, skip_time=0))

    def test_positive_skip_time(self):
        """
        Test that when creating a simulated file reader that skips a positive number of seconds, no ValueError is raised.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            self.assertTrue(SimulatedFileReader(Queue(), f, skip_time=1))

    async def test_read(self):
        """
        Test reading the corpus without skipping anything.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10)
            self.assertEqual(0, queue.length())
            await reader.read()
            self.assertEqual(600, queue.length())

    async def test_read_skip_no_lines(self):
        """
        Test that when reading the corpus after skipping no lines, all tweets are loaded.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_lines=0)
            await reader.read()
            self.assertEqual(600, queue.length())

    async def test_read_skip_lines(self):
        """
        Test reading the corpus after skipping a number of lines.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_lines=100)
            await reader.read()
            self.assertEqual(500, queue.length())

    async def test_read_skip_all_lines(self):
        """
        Test that when all lines are skipped, the queue is empty.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_lines=600)
            await reader.read()
            self.assertEqual(0, queue.length())

    async def test_read_skip_excess_lines(self):
        """
        Test that when excess lines are skipped, the queue is empty.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_lines=601)
            await reader.read()
            self.assertEqual(0, queue.length())

    async def test_read_skip_no_time(self):
        """
        Test that when reading the corpus after skipping no time, all tweets are loaded.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_time=0)
            await reader.read()
            self.assertEqual(600, queue.length())

    async def test_read_skip_lines(self):
        """
        Test reading the corpus after skipping some time.
        """

        """
        Calculate the number of lines that should be skipped.
        """
        skipped = 0
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            for line in lines:
                if extract_timestamp(json.loads(line)) == start:
                    skipped += 1
                else:
                    break

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_time=1)
            await reader.read()
            self.assertEqual(600 - skipped, queue.length())

    async def test_read_skip_all_time(self):
        """
        Test reading the corpus after skipping all time.
        """

        """
        Calculate the number of lines that should be skipped.
        """
        skip = 0
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            end = extract_timestamp(json.loads(lines[-1]))
            skip = end - start

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_time=skip)
            await reader.read()
            self.assertEqual(50, queue.length())

    async def test_read_skip_excess_time(self):
        """
        Test reading the corpus after excess time.
        """

        """
        Calculate the number of lines that should be skipped.
        """
        skip = 0
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            end = extract_timestamp(json.loads(lines[-1]))
            skip = end - start

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_time=skip + 1)
            await reader.read()
            self.assertEqual(0, queue.length())

    async def test_normal_speed(self):
        """
        Test that when using normal speed, the time it takes is equivalent to the length of the corpus.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            end = extract_timestamp(json.loads(lines[-1]))
            length = end - start

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=1)
            start = time.time()
            await reader.read()
            self.assertEqual(600, queue.length())
            self.assertEqual(length, round(time.time() - start))

    async def test_double_speed(self):
        """
        Test that when using double speed, the time it takes is equivalent to half the length of the corpus.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            end = extract_timestamp(json.loads(lines[-1]))
            length = end - start

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=2)
            start = time.time()
            await reader.read()
            self.assertEqual(600, queue.length())
            self.assertEqual(length / 2., round(time.time() - start, 1))

    async def test_half_speed(self):
        """
        Test that when using half speed, the time it takes is equivalent to ouble the length of the corpus.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            end = extract_timestamp(json.loads(lines[-1]))
            length = end - start

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=0.5)
            start = time.time()
            await reader.read()
            self.assertEqual(600, queue.length())
            self.assertEqual(length * 2., round(time.time() - start, 1))

    async def test_max_lines(self):
        """
        Test that when limiting the number of lines, only a few are returned.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, max_lines=100)
            await reader.read()
            self.assertEqual(100, queue.length())

    async def test_max_lines_zero(self):
        """
        Test that when reading zero lines, no lines are returned.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, max_lines=0)
            await reader.read()
            self.assertEqual(0, queue.length())

    async def test_max_lines_all(self):
        """
        Test that when reading all lines, all lines are returned.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, max_lines=600)
            await reader.read()
            self.assertEqual(600, queue.length())

    async def test_max_lines_excess(self):
        """
        Test that when reading excess lines, all lines are returned.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, max_lines=1200)
            await reader.read()
            self.assertEqual(600, queue.length())

    async def test_max_time(self):
        """
        Test that when limiting the time, only a few are returned.
        """

        """
        Calculate the start and end of the corpus.
        """
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            end = extract_timestamp(json.loads(lines[-1]))

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, max_time=1)
            await reader.read()
            self.assertEqual(100, queue.length())
            self.assertEqual(start, extract_timestamp(queue.head()))
            self.assertEqual(start, extract_timestamp(queue.tail()))

    async def test_max_time_zero(self):
        """
        Test that when the time is zero, nothing is returned.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, max_time=0)
            await reader.read()
            self.assertEqual(0, queue.length())

    async def test_max_time_all(self):
        """
        Test that when all the time is allowed, the entire corpus is returned.
        """

        """
        Calculate the start and end of the corpus.
        """
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            end = extract_timestamp(json.loads(lines[-1]))

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, max_time=end - start + 1)
            await reader.read()
            self.assertEqual(600, queue.length())

    async def test_max_time_excess(self):
        """
        Test that when excess time is allowed, the entire corpus is returned.
        """

        """
        Calculate the start and end of the corpus.
        """
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            start = extract_timestamp(json.loads(lines[0]))
            end = extract_timestamp(json.loads(lines[-1]))

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, max_time=end - start + 2)
            await reader.read()
            self.assertEqual(600, queue.length())

    async def test_skip_retweets(self):
        """
        Test that when skipping retweets, none of the tweets read from the corpus are retweets.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_retweets=True)
            await reader.read()
            self.assertTrue(queue.length())
            self.assertTrue(all( not is_retweet(tweet) for tweet in queue.queue ))

        """
        Test that all the correct tweets are in the queue.
        """
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                if is_retweet(tweet):
                    self.assertFalse(tweet in queue.queue)
                else:
                    self.assertTrue(tweet in queue.queue)

    async def test_no_skip_retweets(self):
        """
        Test that when not skipping retweets, all of the tweets in the corpus are retained.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_retweets=False)
            await reader.read()
            self.assertTrue(queue.length())
            self.assertTrue(any( is_retweet(tweet) for tweet in queue.queue ))

        """
        Test that all the tweets are in the queue.
        """
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                self.assertTrue(tweet in queue.queue)

    async def test_skip_unverified(self):
        """
        Test that when skipping tweets from unverified authors, none of the tweets read from the corpus are from unverified authors.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_unverified=True)
            await reader.read()
            self.assertTrue(queue.length())
            self.assertTrue(all( is_verified(tweet) for tweet in queue.queue ))

        """
        Test that all the correct tweets are in the queue.
        """
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                if is_verified(tweet):
                    self.assertTrue(tweet in queue.queue)
                else:
                    self.assertFalse(tweet in queue.queue)

    async def test_no_skip_unverified(self):
        """
        Test that when not skipping tweets from unverified authors, all of the tweets in the corpus are retained.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_unverified=False)
            await reader.read()
            self.assertTrue(queue.length())
            self.assertTrue(any( is_verified(tweet) for tweet in queue.queue ))

        """
        Test that all the tweets are in the queue.
        """
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                self.assertTrue(tweet in queue.queue)

    async def test_skip_retweets_but_not_unverified(self):
        """
        Test that when skipping retweets, but not tweets from unverified authors, retweets are not retained.
        """

        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            queue = Queue()
            reader = SimulatedFileReader(queue, f, speed=10, skip_retweets=True, skip_unverified=False)
            await reader.read()
            self.assertTrue(queue.length())
            self.assertTrue(not any( is_retweet(tweet) for tweet in queue.queue ))
            self.assertTrue(any( is_verified(tweet) for tweet in queue.queue ))
