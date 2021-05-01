"""
The :class:`~twitter.file.simulated_reader.SimulatedFileReader` simulates the real-time stream by adding tweets to the queue as if the tweets were being received in real-time.
It pretends that the event is ongoing, and adds data to the queue according to when they happened.
This means that in high-volume periods, the :class:`~twitter.file.simulated_reader.SimulatedFileReader` adds many tweets to the :class:`~queues.Queue`.
In low-volume periods, it enqueues fewer tweets.

Since the :class:`~twitter.file.simulated_reader.SimulatedFileReader` has high fidelity, it is most appropriate in experimental or evaluation settings.
Since the volume is likely to change, it also tests the :ref:`consumers' <consumers>` performance in volatile or high-volume situations.
"""

import asyncio
import json
import os
import sys
import time

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from twitter import *
from twitter.file import FileReader

class SimulatedFileReader(FileReader):
    """
    The :class:`~twitter.file.simulated_reader.SimulatedFileReader` is based on the :class:`~twitter.file.FileReader`, so it reads tweets from a file and adds it to a queue.
    This works like a simulation, as if the event was happening at the same time.

    In addition to the parameters accepted by the :class:`~twitter.file.FileReader`, it also accepts the ``speed``.
    Since the :class:`~twitter.file.simulated_reader.SimulatedFileReader` simulates the stream as if it is happening in real-time, the ``speed`` is a function of time as well.

    :ivar speed: The reading speed as a function of time.
                 If it is set to 0.5, for example, the event progresses at half the speed.
                 If it is set to 2, the event progresses at double the speed.
    :vartype speed: int
    """

    def __init__(self, queue, f, speed=1, *args, **kwargs):
        """
        Create the :class:`~twitter.file.simulated_reader.SimulatedFileReader` with the file from where to read tweets and the :class:`~queues.Queue` where to store them.
        The ``speed`` is an extra parameter in addition to the :class:`~twitter.file.FileReader`'s parameters.

        :param queue: The queue to which to add the tweets.
        :type queue: :class:`~queues.Queue`
        :param f: The opened file from where to read the tweets.
        :type f: file
        :param speed: The reading speed, considered to be a function of time.
                      If it is set to 0.5, for example, the event progresses at half the speed.
                      If it is set to 2, the event progresses at double the speed.
        :type speed: int

        :raises ValueError: When the speed is zero or negative.
        """

        super(SimulatedFileReader, self).__init__(queue, f, *args, **kwargs)

        """
        Validate the inputs.
        """
        if speed <= 0:
            raise ValueError(f"The speed must be positive; received {speed}")

        self.speed = speed

    @FileReader.reading
    async def read(self):
        """
        Read the file and add each line as a dictionary to the queue.
        """

        file = self.file

        """
        Extract the timestamp from the first tweet, then reset the file pointer.
        """
        pos = file.tell()
        line = file.readline()
        if not line:
            return
        first = extract_timestamp(json.loads(line))
        file.seek(pos)

        """
        Go through each line and add it to the queue.
        """
        start = time.time()
        for i, line in enumerate(file):
            tweet = json.loads(line)
            created_at = extract_timestamp(tweet)

            """
            If the maximum number of lines, or the time, has been exceeded, stop reading.
            """
            if self.max_lines >= 0 and i >= self.max_lines:
                break

            if self.max_time >= 0 and created_at - first >= self.max_time:
                break

            """
            If the tweet is 'in the future', stop reading until the reader catches up.
            It is only after it catches up that the tweet is added to the queue.
            """
            elapsed = time.time() - start
            if (created_at - first) / self.speed > elapsed and self.active:
                await asyncio.sleep((created_at - first) / self.speed - elapsed)

            """
            If the reader has been interrupted, stop reading.
            """
            if not self.active:
                break

            """
            Only add a tweet if it is valid.
            """
            if self.valid(tweet):
                self.queue.enqueue(tweet)
