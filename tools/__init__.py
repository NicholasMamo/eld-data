"""
EvenTDT comes with tools to help you collect corpora from Twitter and process them in various ways.
You can use EvenTDT's tools to :mod:`create timelines <tools.consume>`, :mod:`extract participants <tools.participants>`, :mod:`perform evaluations <tools.evaluation>` and more.
Keep reading to learn more about the tools available in EvenTDT.
"""

import copy
import datetime
import json
import os
import re
import sys
import time

file_path = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(file_path, '..')
lib = os.path.join(root, 'eventdt')
sys.path.insert(-1, root)
sys.path.insert(-1, lib)

from objects.exportable import Exportable

def meta(args):
    """
    Get the meta arguments.

    :param args: The command-line arguments.
    :type args: :class:`argparse.Namespace`

    :return: The meta arguments as a dictionary.
    :rtype: dict
    """

    meta = copy.deepcopy(vars(args))
    meta['_date'] = datetime.datetime.now().isoformat()
    meta['_timestamp'] = time.time()
    meta['_cmd'] = ' '.join(sys.argv)
    return meta

def save(file, data):
    """
    Save the data to the given file.
    The function saves the data as a JSON file.

    :param file: The path to the file where to save the data.
    :type file: str
    :param data: The data to save.
                 The function expects a dictionary that can be JSON serializable.
                 The function tries to convert the values that cannot be serialized to arrays.
                 Only classes that inherit the :class:`~objects.exportable.Exportable` can be converted to arrays.
                 This is done through the :func:`~objects.exportable.Exportable.to_array` function.
    :type data: dict
    """

    """
    Create the directory if it doesn't exist.
    """
    dir = os.path.dirname(file)
    if dir and not os.path.exists(dir):
        os.mkdir(dir)

    """
    Encode the data and save it.
    """
    data = Exportable.encode(data)
    with open(file, 'w') as f:
        f.write(json.dumps(data))

def load(file):
    """
    Load the data from the given file.

    :param file: The path to the file from where to load the data.
    :type file: str

    :return: A new dictionary with the loaded data.
    :rtype: dict
    """

    """
    Read the data as a JSON string.
    Then, decode the data and return it.
    """
    with open(file, 'r') as f:
        line = f.readline()
        data = json.loads(line)

    return Exportable.decode(data)

def cache_exists(file, cache_dir='.cache'):
    """
    Check whether cache exists for the given file.
    The cache exists in a cache directory and has the same name as the given file.

    :param file: The path to the file whose cache will be sought.
    :type file: str
    :param cache_dir: The directory where cache is stored.
                      This is relative to the file's directory.
    :type cache: str

    :return: A boolean indicating whether cache exists for the given file.
    :rtype: bool
    """

    dir = os.path.dirname(file)
    filename = os.path.basename(file)
    cache_dir = os.path.join(dir, cache_dir)
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        cache_file = os.path.join(cache_dir, filename)
        return os.path.exists(cache_file) and os.path.isfile(cache_file)

    return False

def is_json(file):
    """
    Check whether the given file is JSON-encoded.
    The function checks only if the filename ends with the ``json`` suffix.

    :param file: The path to the file to check if it is JSON-encoded.
    :type file: str

    :return: A boolean indicating whether the file is JSON-encoded.
    :rtype: bool
    """

    return file.endswith('.json')

def is_file(string):
    """
    Check whether the given string is a file path.

    :param string: The string to check if it is a file path.
    :type string: str

    :return: A boolean indicating whether the given string is a file path.
    :rtype: str
    """

    pattern = re.compile('.*\..*')
    return pattern.match(string)
