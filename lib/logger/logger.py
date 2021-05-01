"""
A logging interface.
"""

from datetime import datetime

import os
import sys

paths = [
    os.path.join(os.path.dirname(__file__), '..'),
    os.path.join(os.path.dirname(__file__), '..', '..')
]

for path in paths:
    if path not in sys.path:
        sys.path.insert(1, path)

from objects.ordered_enum import OrderedEnum
from config import conf

class LogLevel(OrderedEnum):
    """
    The logger's logging level.
    It is based on a :class:`~objects.ordered_enum.OrderedEnum`.

    Valid logging levels:

        #. `INFO` - Information and higher-level logging only
        #. `WARNING` - Warnings and higher-level logging only
        #. `ERROR` - Errors and higher-level logging only
    """

    INFO = 1
    WARNING = 2
    ERROR = 3

def log_time():
    """
    Get the time string.

    :return: The time string.
    :rtype: str
    """

    return datetime.now().strftime("%H:%M:%S")

def process_name(process):
    """
    Get the string with the name of the process.

    :param process: The name of the process, if available.
    :type process: str

    :return: A string with the formatted name of the process.
             If no process name is given, this function returns an empty string.
    :rtype: str
    """

    if not process:
        return ''
    else:
        name = f"{ process[:20] }…" if len(process) > 20 else process
        return f"({ name })"

def set_logging_level(level):
    """
    Set the logging level.

    :param level: The logging level.
    :type level: :class:`~logger.logger.LogLevel`
    """

    conf.LOG_LEVEL = level

def info(*args, process=None):
    """
    Log an information message.
    All arguments are passed on to the ``print`` function.

    :param process: The name of the process calling the log.
                    If it is given, the logger prints part of its name while logging.
    :type process: None or str
    """

    if conf.LOG_LEVEL <= LogLevel.INFO:
        prefix = f"{ process_name(process) } { log_time() }: INFO:"
        print(prefix.strip(), *args)

def warning(*args, process=None):
    """
    Log a warning.
    All arguments are passed on to the ``print`` function.

    :param process: The name of the process calling the log.
                    If it is given, the logger prints part of its name while logging.
    :type process: None or str
    """

    if conf.LOG_LEVEL <= LogLevel.WARNING:
        prefix = f"{ process_name(process) } { log_time() }: WARNING:"
        print(prefix.strip(), *args)

def error(*args, process=None):
    """
    Log an error.
    All arguments are passed on to the ``print`` function.

    :param process: The name of the process calling the log.
                    If it is given, the logger prints part of its name while logging.
    :type process: None or str
    """

    if conf.LOG_LEVEL <= LogLevel.ERROR:
        prefix = f"{ process_name(process) } { log_time() }: ERROR:"
        print(prefix.strip(), *args)
