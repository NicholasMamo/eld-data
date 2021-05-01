"""
File readers emulate the Twitter stream by reading corpora and feeding them into a :class:`~queues.Queue` so that they can be processed by :ref:`consumers <consumers>`.
These readers expect that the corpora were collected using a :ref:`listener <twitter_listeners>`.
The input files have one object on each line.

File readers are useful in evaluation or experimental settings.
By simulating the Twitter stream, the different file readers can be used to replicate the original, real-time setting of a stream.
"""

from abc import ABC, abstractmethod

from .. import *

import asyncio
import json

class FileReader(ABC):
    """
    The :class:`~twitter.file.FileReader` is a class that describes the general state of file readers.

    Generally speaking, any file reader should implement the :func:`~twitter.file.FileReader.read` function, which reads tweets and adds them to the :class:`~queues.Queue`.
    This :class:`~queues.Queue` is one of the variables that make up the :class:`~twitter.file.FileReader`'s state.
    Accompanying it is the ``file``, which is the pointer to an opened file.

    The :class:`~twitter.file.FileReader` also stores the maximum number of lines, ``max_lines``, and the maximum time, in seconds, ``max_time`` it should spend reading tweets from the corpora.
    These variables refer to the number of lines to read from the file, and the maximum time as represented in the corpus.
    For example, if the ``max_time`` is 60 seconds, the reader reads all the tweets from the corpus published within the first minute.
    It does not refer to the processing time it should spend reading the corpus.

    The other two variables in the :class:`~twitter.file.FileReader`'s state are the ``active`` and ``stopped`` flags:

    - The ``active`` flag indicates whether the reader is still reading data.
      It is possible that the reader is in the process of reading data while the ``active`` flag is set to ``False``.

    - The ``stopped`` flag indicates whether the reader has finished processing and is idle.

    To stop reading data, call the :func:`~twitter.file.FileReader.stop` function.
    This function sets the ``active`` flag to ``False``.
    When the :func:`~twitter.file.FileReader.read` function actually finishes reading, it sets the ``stopped`` variable to ``True``.

    :ivar queue: The queue to which to add tweets.
    :vartype queue: :class:`~queues.Queue`
    :ivar ~.file: The opened file pointer from where to read the tweets.
    :vartype ~.file: file
    :ivar max_lines: The maximum number of lines to read.
                     If the number is negative, it is ignored.
    :vartype max_lines: int
    :ivar max_time: The maximum time in seconds to spend reading from the file.
                    The time is taken from tweets' timestamps.
                    If the number is negative, it is ignored.
    :vartype max_time: int
    :ivar active: A boolean indicating whether the reader is still reading data.
    :vartype active: bool
    :ivar stopped: A boolean indicating whether the consumer has finished processing.
    :vartype stopped: bool
    :ivar skip_retweets: A boolean indicating whether to skip retweets.
    :vartype skip_retweets: bool
    :ivar skip_unverified: A boolean indicating whether to skip tweets by unverified authors.
    :vartype skip_unverified: bool
    """

    def __init__(self, queue, f, max_lines=-1, max_time=-1, skip_lines=0, skip_time=0,
                 skip_unverified=False, skip_retweets=False):
        """
        Create the file reader with the :class:`~queues.Queue` where to add tweets and the file from where to read them.
        The ``max_lines`` and ``max_time`` parameters can be used to read only a part of the corpus.

        :param queue: The queue to which to add the tweets.
        :type queue: :class:`~queues.Queue`
        :param f: The opened file from where to read the tweets.
        :type f: file
        :param max_lines: The maximum number of lines to read.
                          If the number is negative, it is ignored.
        :type max_lines: int
        :param max_time: The maximum time in seconds to spend reading from the file.
                         The time is taken from tweets' timestamps.
                         If the number is negative, it is ignored.
        :type max_time: int
        :param skip_lines: The number of lines to skip from the beginning of the file.
        :type skip_lines: int
        :param skip_time: The number of seconds to skip from the beginning of the file.
        :type skip_time: int
        :param skip_unverified: A boolean indicating whether to skip tweets by unverified authors.
        :type skip_unverified: bool
        :param skip_retweets: A boolean indicating whether to skip retweets.
        :type skip_retweets: bool

        :raises ValueError: When the number of lines to skip is not an integer.
        :raises ValueError: When the number of lines to skip is negative.
        :raises ValueError: When the number of seconds to skip is negative.
        """

        self.queue = queue
        self.file = f
        self.max_lines = max_lines
        self.max_time = max_time
        self.active = False
        self.stopped = True
        self.skip_retweets = skip_retweets
        self.skip_unverified = skip_unverified

        """
        Validate the inputs.
        """
        if skip_lines % 1:
            raise ValueError(f"The number of lines to skip must be an integer; received {skip_lines}")

        if skip_lines < 0:
            raise ValueError(f"The number of lines to skip cannot be negative; received {skip_lines}")

        if skip_time < 0:
            raise ValueError(f"The number of seconds to skip cannot be negative; received {skip_time}")

        self.skip(skip_lines, skip_time)

    @staticmethod
    def reading(f):
        """
        This decorator sets the appropriate flags when the file reader starts reading, and when it stops reading.

        :param f: The function to wrap, should be the :class:`~file.FileReader.read` function.
        :type f: function
        """

        async def wrapper(self, *args, **kwargs):
            """
            Set the ``active`` flag to ``True`` and the ``stopped`` flag to ``False`` before starting to read.
            After finishing reading, set the ``active`` flag to ``False`` and the ``stopped`` flag to ``True``.
            """

            self.active = True
            self.stopped = False

            await f(self, *args, **kwargs)

            self.active = False
            self.stopped = True

        return wrapper

    def valid(self, tweet):
        """
        Check whether the given tweet is valid, or whether it should be skipped.
        The rules to decide whether a tweet is valid is based on the parameters of this class:

        - ``skip_retweets``: if ``True``, this retweets are marked as invalid.

        :param tweet: The tweet to consider.
        :type tweet: dict

        :return: A boolean indicating whether the tweet is valid.
        :rtype: bool
        """

        valid = not self.skip_retweets or not is_retweet(tweet)
        valid = valid and (not self.skip_unverified or is_verified(tweet))
        return valid

    def skip(self, lines, time):
        """
        Skip a number of lines from the file.
        This virtually just reads lines without storing them, but moving the file pointer.

        .. note::

            The number of lines and seconds that are skipped depend on the largest number.

        :param lines: The number of lines to skip.
        :type lines: int
        :param time: The number of seconds to skip from the beginning of the file.
                     The time is taken from tweets' timestamps.
        :type time: int
        """

        file = self.file

        """
        Extract the timestamp from the first tweet, then reset the file pointer.
        """
        line = file.readline()
        if not line:
            return
        start = extract_timestamp(json.loads(line))
        file.seek(0)

        """
        Skip a number of lines first.
        """
        if lines >= 0:
            for i in range(int(lines)):
                file.readline()

        """
        Skip a number of seconds from the file.
        Once a line that should not be skipped is skipped, the read is rolled back.
        """
        pos = file.tell()
        line = file.readline()
        if not line:
            return
        next = json.loads(line)
        while extract_timestamp(next) - start < time:
            pos = file.tell()
            line = file.readline()
            if not line:
                break
            next = json.loads(line)

        file.seek(pos)

    @abstractmethod
    async def read(self):
        """
        Read the file and add each line as a dictionary to the queue.
        """

        pass

    def stop(self):
        """
        Set a flag to stop accepting new objects.

        .. note::
            Contrary to the name of the function, the function sets the ``active`` flag to ``False``, not the ``stopped`` flag.
            This function merely asks that the consumer stops accepting new objects for processing.
        """

        self.active = False

from .simulated_reader import SimulatedFileReader
