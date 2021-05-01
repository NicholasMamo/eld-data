"""
Test the functionality of the buffered consumer.
"""

import asyncio
import json
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from logger import logger
from nlp import Document, Tokenizer
from queues import Queue
from queues.consumers.buffered_consumer import DummySimulatedBufferedConsumer
import twitter

logger.set_logging_level(logger.LogLevel.WARNING)

class TestSimulatedBufferedConsumer(unittest.TestCase):
    """
    Test the implementation of the buffered consumer.
    """

    def async_test(f):
        def wrapper(*args, **kwargs):
            coro = asyncio.coroutine(f)
            future = coro(*args, **kwargs)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(future)
        return wrapper

    def test_init_name(self):
        """
        Test that the buffered consumer passes on the name to the base class.
        """

        name = 'Test Consumer'
        consumer = DummySimulatedBufferedConsumer(Queue(), periodicity=10, name=name)
        self.assertEqual(name, str(consumer))

    @async_test
    async def test_binning(self):
        """
        Test that binning works as it should.
        """

        """
        Create an empty queue.
        Use it to create a buffered consumer and set it running.
        """
        queue = Queue()
        consumer = DummySimulatedBufferedConsumer(queue, periodicity=10)
        running = asyncio.ensure_future(consumer.run(max_inactivity=3))
        await asyncio.sleep(0.5)

        """
        Load all tweets into the queue.
        """
        with open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tests', 'corpora', 'CRYCHE-500.json')) as f:
            start = twitter.extract_timestamp(json.loads(f.readline()))
            f.seek(0)
            read = 0
            for line in f:
                tweet = json.loads(line)
                current = twitter.extract_timestamp(tweet)

                """
                Load all tweets unless it's almost time to start processing.
                When it's time to start processing, check that after adding that tweet, the buffer is processed and emptied.
                """
                if current - start < 10:
                    queue.enqueue(tweet)
                    await asyncio.sleep(0.25)
                else:
                    self.assertEqual(read, consumer.buffer.length())
                    queue.enqueue(tweet)
                    await asyncio.sleep(0.2)
                    self.assertEqual(0, consumer.buffer.length())
                    break

                read += 1
                self.assertEqual(read, consumer.buffer.length())

    def test_get_timestamp_tweet(self):
        """
        Test getting the timestamp from a tweet dictionary.
        """

        with open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tests', 'corpora', 'CRYCHE-500.json')) as f:
            consumer = DummySimulatedBufferedConsumer(Queue(), periodicity=10)
            for line in f:
                tweet = json.loads(line)
                self.assertEqual(twitter.extract_timestamp(tweet), consumer._get_timestamp(tweet))

    def test_get_timestamp_document(self):
        """
        Test getting the timestamp from a document.
        """

        with open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tests', 'corpora', 'CRYCHE-500.json')) as f:
            consumer = DummySimulatedBufferedConsumer(Queue(), periodicity=10)
            tokenizer = Tokenizer()
            for line in f:
                tweet = json.loads(line)
                text = twitter.full_text(tweet)
                tokens = tokenizer.tokenize(text)
                document = Document(text=text, dimensions=tokens, attributes={ 'timestamp': twitter.extract_timestamp(tweet) })
                self.assertEqual(twitter.extract_timestamp(tweet), consumer._get_timestamp(document))

    def test_get_timestamp_object_raises_ValueError(self):
        """
        Test getting the timestamp from an arbitrary object raises a ValueError.
        """

        consumer = DummySimulatedBufferedConsumer(Queue(), periodicity=10)
        tokenizer = Tokenizer()
        self.assertRaises(ValueError, consumer._get_timestamp, tokenizer)
