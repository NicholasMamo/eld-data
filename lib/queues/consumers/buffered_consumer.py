"""
The :class:`~queues.consumers.Consumer` is designed to operate in real-time.
The :class:`~queues.consumers.buffered_consumer.BufferedConsumer` builds on it, but it buffers input before processing.
Essentially, this buffering step transforms the :class:`~queues.consumers.Consumer` into a windowed approach.

The two :class:`~queues.Queue` instances in the :class:`~queues.consumers.buffered_consumer.BufferedConsumer` have these roles:

- Queue: The normal queue receives input from the :class:`~twitter.file.FileReader` or :class:`~twitter.listener.TweetListener`.

- Buffer: The :class:`~queues.consumers.buffered_consumer.BufferedConsumer` constantly empties the normal queue into the buffer.
  After every window, the algorithm processes the tweets collected so far in it.
  While processing, the buffer continuesreceiving new tweets, which will be processed in the next time window.

This package provides two types of buffered consumers:

- The :class:`~queues.consumers.buffered_consumer.BufferedConsumer` bases its periodicity on the machine's time.
  Therefore it is opportune when running in a live environment.

- The :class:`~queues.consumers.buffered_consumer.SimulatedBufferedConsumer` bases its periodicity on the tweets it is receiving.
  It can be used both in a live environment, but especially when loading tweets from the :class:`~twitter.file.FileReader`.
"""

from abc import ABC, abstractmethod

import asyncio
import os
import sys
import time

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from nlp import Document
from objects.attributable import Attributable
from queues import Queue
from queues.consumers import Consumer

import twitter

class BufferedConsumer(Consumer):
    """
    When calling the :func:`~queues.consumers.buffered_consumer.BufferedConsumer.run` function, the :class:`~queues.consumers.buffered_consumer.BufferedConsumer` splits into two processes:

    1. The consumption simply moves tweets from the :class:`~queues.Queue` into the the buffer, another :class:`~queues.Queue`.

    2. The processing wakes up every time window to process the tweets collectd in the buffer so far.
       While processing, the buffer receives new tweets, but these are only processed in the next time window.

    Apart from maintaining the buffer as a :class:`~queues.Queue`, the state also includes the periodicity, or the length of the time window, in seconds.
    This affects how often the buffer is processed.

    :ivar periodicity: The time window, in seconds, of the buffered consumer, or how often the consumer processes the buffer's contents.
    :vartype periodicity: int
    :ivar buffer: The buffer of objects that have to be processed.
    :vartype buffer: :class:`~queues.Queue`
    """

    def __init__(self, queue, periodicity=60, *args, **kwargs):
        """
        Initialize the buffered consumer with its queue and periodicity.

        Additional arguments and keyword arguments are passed on to the base class.

        :param queue: The queue that is consumed.
        :type queue: :class:`~queues.Queue`
        :param periodicity: The time window, in seconds, of the buffered consumer, or how often the consumer processes the buffer's contents.
        :type periodicity: int
        """

        super(BufferedConsumer, self).__init__(queue, *args, **kwargs)
        self.periodicity = periodicity
        self.buffer = Queue()

    async def run(self, wait=0, max_inactivity=-1, *args, **kwargs):
        """
        Update the flags to indicate that the consumer is running and start the buffered consumer's two roles:

        1. The consumption simply moves tweets from the :class:`~queues.Queue` into the the buffer, another :class:`~queues.Queue`.

        2. The processing wakes up every time window to process the tweets collectd in the buffer so far.
           While processing, the buffer receives new tweets, but these are only processed in the next time window.

        Similarly to the :class:`~queues.consumers.Consumer`, the buffered consumer also accepts the ``wait`` and ``max_inactivity`` parameters.

        :param wait: The time in seconds to wait until starting to understand the event.
                     This is used when the file listener spends a lot of time skipping documents.
        :type wait: int
        :param max_inactivity: The maximum time in seconds to wait idly without input before stopping.
                               If it is negative, the consumer keeps waiting for input until the maximum time expires.
        :type max_inactivity: int

        :return: The output of the consumption process.
        :rtype: any
        """

        await asyncio.sleep(wait)
        self._started()
        results = await asyncio.gather(
            self._consume(*args, max_inactivity=max_inactivity, **kwargs),
            self._process(*args, **kwargs),
        )
        self._stopped()
        return results[1]

    async def _consume(self, max_inactivity):
        """
        Consume the queue.

        :param max_inactivity: The maximum time in seconds to wait idly without input before stopping.
                               If it is negative, the consumer keeps waiting for input until the maximum time expires.
        :type max_inactivity: int
        """

        """
        The consumer should keep working until it is stopped.
        """
        while self.active:
            """
            If the queue is idle, wait for input.
            """
            active = await self._wait_for_input(max_inactivity=max_inactivity)
            if not active:
                break

            """
            The consuming phase empties the queue and stores the elements in the buffer.
            The buffer is processed separately in the :func:`~queues.consumers.buffered_consumer.BufferedConsumer.process` function.
            """
            items = self.queue.dequeue_all()
            self.buffer.enqueue(*items)

        self.active = False

    @abstractmethod
    async def _process():
        """
        Process the buffered items.
        """

        pass

    async def _sleep(self):
        """
        Sleep until the window is over.
        The function periodically checks if the consumer has been asked to stop.
        """

        for i in range(self.periodicity):
            await asyncio.sleep(1)
            if not self.active:
                break

        """
        If the periodicity is a float, sleep for the remaining milliseconds.
        """
        if self.active:
            await asyncio.sleep(self.periodicity % 1)

class SimulatedBufferedConsumer(BufferedConsumer):
    """
    The :class:`~queues.consumers.buffered_consumer.SimulatedBufferedConsumer` is exactly like the :class:`~queues.consumers.buffered_consumer.BufferedConsumer`, but its periodicity is not real-time.
    Whereas the :class:`~queues.consumers.buffered_consumer.BufferedConsumer`'s time window is based on the machine's time, the :class:`~queues.consumers.buffered_consumer.SimulatedBufferedConsumer` looks at the tweet's timestamps.
    This makes the :class:`~queues.consumers.buffered_consumer.SimulatedBufferedConsumer` ideal in situations where it is necessary to simulate the live environment, for example when using a :class:`~twitter.file.FileReader`.
    """

    def __init__(self, queue, periodicity, *args, **kwargs):
        """
        Initialize the buffered consumer with its queue and periodicity.

        Additional arguments and keyword arguments are passed on to the base class.

        :param queue: The queue that is consumed.
        :type queue: :class:`~queues.Queue`
        :param periodicity: The time window in seconds of the buffered consumer, or how often it is invoked.
        :type periodicity: int
        """

        super(SimulatedBufferedConsumer, self).__init__(queue, periodicity, *args, **kwargs)

    async def _sleep(self):
        """
        Sleep until the window is over.
        At this point, the queue is emptied into a buffer for processing.
        The function periodically checks if the consumer has been asked to stop.
        """

        """
        Wait until there's something in the queue, to get a reference point for when the sleep should end.
        """
        while self.buffer.head() is None and self.active:
            await asyncio.sleep(0.1)

        if self.active:
            head = self.buffer.head()
            start = self._get_timestamp(head)

        """
        Check if the consumer should stop.
        The consumer should stop if:

            #. It has been shut down; or

            #. The buffer's periodicity has been reached.
        """
        while True:
            tail = self.buffer.tail()
            if not self.active or self._get_timestamp(tail) - start >= self.periodicity:
                break

            await asyncio.sleep(0.1)

    def _get_timestamp(self, item):
        """
        Extract the timestamp from the item.

        :param item: The item from which to extract the timestamp.
                     If the item is a tweet dictionary, the function looks for the ``timestamp`` key.
                     Otherwise, if the item is an :class:`~objects.attributable.Attributable`, the function looks for the ``timestamp`` attribute.
        :type item: dict or :class:`~objects.attributable.Attributable`

        :return: The item's timestamp.
        :rtype: int

        :raises ValueError: When the timestamp cannot be extracted from the item.
        """

        if type(item) is dict:
            return twitter.extract_timestamp(item)
        elif isinstance(item, Attributable):
            return item.attributes['timestamp']
        else:
            raise ValueError(f"Expected tweet dictionary or Attributable; received { type(item) }")

class DummySimulatedBufferedConsumer(SimulatedBufferedConsumer):
    """
    The :class:`~queues.consumers.buffered_consumer.DummySimulatedBufferedConsumer` is a trivial implementation of the :class:`~queues.consumers.buffered_consumer.SimulatedBufferedConsumer`, meant only for unit testing.
    In its processing, it does nothing with the tweets, and discards them immediately.
    """

    async def _process(self):
        """
        Discard all tweets from the buffer.
        """

        while self.active:
            if self.buffer.length() > 0:
                self.buffer.dequeue_all()
            await self._sleep()
