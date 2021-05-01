"""
After adding tweets to a :class:`~queues.Queue`, consumers process that data.
Each consumer dequeues the accumulated data and performs some processing on it.

There is one class that is the base of all consumers: the basic :class:`~queues.consumers.Consumer`
From it stem all other consumers, including the :class:`~queues.consumers.buffered_consumer.BufferedConsumer`.

These consumers are meant to be usable with minimal changes.
Although that makes them less flexible than building your own consumer, it also makes it easier to get started with EvenTDT.

Normally, you would either set up a stream using the :class:`~twitter.listeners.TweetListener` or by reading from a file using the :class:`~twitter.file.FileReader`.
These streams store tweets in a :class:`~queue.Queue`—the same one that is given to any :class:`~queues.consumers.Consumer` class.
You can then run the :func:`~queues.consumers.Consumer.run` function.
More commonly, if you are not interested in the implementation details, you can run consumers using the :mod:`~tools.consume` command-line tool.

The consumers package is the most central one in EvenTDT because it ties together all other packages.
Apart from some simple consumers, the package includes consumers based on algorithms presented in papers.
These complete solutions can be used as baselines to generate a :class:`~summarization.timeline.Timeline` and later for summarization.
These consumers revolve around a :class:`~tdt.algorithms.TDTAlgorithm` and are meant to be as faithful as possible to the original approaches.
"""

from abc import ABC, abstractmethod

import asyncio
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from lib.queues import Queue

class Consumer(ABC):
    """
    The abstract :class:`~queues.consumers.Consumer` class outlines the necessary functions of any consumer.
    All consumers follow a simple workflow.
    After being initialized with a :class:`~queues.Queue` that supplies data, the consumer run using the :func:`~queues.consumers.Consumer.run` function.

    Consumers have two other state variables apart from the :class:`~queues.Queue`: the ``active`` and ``stopped`` variables.
    The ``active`` variable indicates whether the consumer is still accepting objects.
    The ``stopped`` variable indicates whether the consumer has finished consuming all objects.
    Generally, consumers keep accepting objects until the ``active`` variable is disabled, at which point they process the last objects and set the ``stopped`` flag to ``True``.

    :ivar queue: The :class:`~queues.Queue` that is to be consumed.
    :vartype queue: :class:`~queues.Queue`
    :ivar active: A boolean indicating whether the consumer is still accepting data.
    :vartype active: bool
    :ivar stopped: A boolean indicating whether the consumer has finished processing.
    :vartype stopped: bool
    :ivar name: The consumer's name.
                This is optional and has no function except to differentiate between consumers.
    :vartype name: str
    """

    def __init__(self, queue, name=None, *args, **kwargs):
        """
        Initialize the consumer with its :class:`~queues.Queue`.

        :param queue: The queue that will be consumed.
        :type queue: :class:`~queues.Queue`
        :param name: The consumer's name.
                     This is optional and has no function except to differentiate between consumers.
        :type name: None or str
        """

        self.queue = queue
        self.active = False
        self.stopped = True
        self.name = name or ''

    async def run(self, wait=0, max_inactivity=-1, *args, **kwargs):
        """
        Update the flags to indicate that the consumer is running and start consuming the :class:`~queues.Queue`.

        If the :class:`~queues.Queue` is being populated by a :class:`~twitter.file.FileReader`, there might be an initial delay until the :class:`~queues.Queue` receives any data.
        This is because the :class:`~twitter.file.FileReader` supports skipping tweets, which introduces some latency.
        When skipping a lot of time or lines, this latency can get very large.
        You can use the ``wait`` parameter to delay running the consumer by a few seconds to wait for the :class:`~twitter.file.FileReader` to finish skipping part of the corpus.

        In addition, corpora may be sparse with periods of time during which little data is fed to the consumer.
        This can also happen when the :class:`~twitter.listeners.TweetListener` fails to collect tweets because of errors in the Twitter API.
        The ``max_inactivity`` parameter is the allowance, in seconds, for how long the consumer should wait without receiving input before it decides no more data will arrive and stop.

        :param wait: The time, in seconds, to wait until starting to consume the :class:`~queues.Queue`.
                     This is used when the :class:`~twitter.file.FileReader` spends a lot of time skipping tweets.
        :type wait: int
        :param max_inactivity: The maximum time, in seconds, to wait idly without input before stopping the consumer.
                               If it is negative, the consumer keeps waiting for input indefinitely.
        :type max_inactivity: int

        :return: The output of the consumption process.
        :rtype: any
        """

        await asyncio.sleep(wait)
        self._started()
        results = await asyncio.gather(
            self._consume(*args, max_inactivity=max_inactivity, **kwargs),
        )
        self._stopped()
        return results[0]

    def stop(self):
        """
        Set a flag to stop accepting new tweets.

        .. note::
            Contrary to the name of the function, the function sets the ``active`` flag to ``False``, not the ``stopped`` flag to ``True``.
            This function merely asks the consumer to stop accepting new tweets for processing.
            When the consumer actually stops, after it finishes processing whatever tweets it has, it sets the ``stopped`` flag to ``True`` itself.
        """

        self.active = False

    @abstractmethod
    async def _consume(self, max_inactivity, *args, **kwargs):
        """
        Consume the queue.
        This is the function where most processing occurs.

        :param max_inactivity: The maximum time in seconds to wait idly without input before stopping.
                               If it is negative, the consumer keeps waiting for input until the maximum time expires.
        :type max_inactivity: int
        """

        pass

    async def _wait_for_input(self, max_inactivity, sleep=0.25):
        """
        Wait for input from the queue.
        When input is received, the function returns True.
        If no input is found for the given number of seconds, the function returns False.
        If the maximum inactivity is negative, it is disregarded.

        :param max_inactivity: The maximum time in seconds to wait idly without input before stopping.
                               If it is negative, it is ignored.
        :type max_inactivity: int
        :param sleep: The number of seconds to sleep while waiting for input.
        :type sleep: float

        :return: A boolean indicating whether the consumer should continue, or whether it has been idle for far too long.
        :rtype: bool
        """

        """
        If the queue is empty, it could be an indication of downtime.
        Therefore the consumer should yield for a bit.
        """
        inactive = 0
        while (self.active and not self.queue.length() and
              (max_inactivity < 0 or inactive < max_inactivity)):
            await asyncio.sleep(sleep)
            inactive += sleep

        if not self.active:
            return False

        """
        If there are objects in the queue after waiting, return `True`.
        """
        if self.queue.length():
            return True

        """
        If the queue is still empty, return `False` because the queue is idle.
        """
        if inactive >= max_inactivity and max_inactivity >= 0:
            return False

        return True

    def _started(self):
        """
        A function that sets the active and stopped flags to indicate that the consumer has started operating.
        """

        self.active = True
        self.stopped = False

    def _stopped(self):
        """
        A function that sets the active and stopped flags to indicate that the consumer has stopped operating.
        """

        self.active = False
        self.stopped = True

    def __str__(self):
        """
        Get the string representation of the consumer.
        This function returns the consumer's name if it is set, ``None`` otherwise.

        :return: The consumer's name, or an empty string if it is not set.
        :rtype: str
        """

        return self.name
