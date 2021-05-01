"""
Exportable objects are normal objects that can be exported as a JSON string and loaded back.
"""

from abc import ABC, abstractmethod
import copy
import importlib
import json
import re

class Exportable(ABC):
    """
    An abstract class of an object that can be exported as a JSON string and imported back.
    """

    ALIASES = { 'nlp.term_weighting': 'nlp.weighting',
                'summarization.timeline.timeline': 'summarization.timeline' }

    @abstractmethod
    def to_array(self):
        """
        Export the object as a dictionary.

        :return: The object as a dictionary.
        :rtype: dict
        """

        pass

    @staticmethod
    @abstractmethod
    def from_array(array):
        """
        Create an instance of the object from the given dictionary.

        :param array: The dictionary with the attributes to create the object instance.
        :type array: dict

        :return: A new instance of an object with the same attributes stored in the object.
        :rtype: object
        """

        pass

    @staticmethod
    def encode(data):
        """
        Try to encode the given data.
        This function expects a dictionary, a list or an object and checks if values are JSON serializable.
        If this is not possible, instances of :class:`~objects.exportable.Exportable` are converted to arrays.
        This is done through the :func:`~objects.exportable.Exportable.to_array` function.

        :param data: The data to encode.
        :type data: dict or list

        :return: The encoded data.
        :rtype: dict or list or object
        """

        data = copy.deepcopy(data)

        if type(data) is dict:
            """
            The first case is when a dictionary is given and all keys need to be encoded because some may represent an object.
            """
            for key in data:
                try:
                    data[key] = json.loads(json.dumps(data.get(key)))
                except TypeError:
                    if type(data[key]) in [ dict, list ]:
                        data[key] = Exportable.encode(data.get(key))
                    else:
                        data[key] = data.get(key).to_array()
        elif type(data) is list:
            """
            The second case is when a list is given and all items need to be encoded because some may represent an object.
            """
            for i, item in enumerate(data):
                try:
                    data[i] = json.loads(json.dumps(item))
                except TypeError:
                    if type(item) in [ dict, list ]:
                        data[i] = Exportable.encode(item)
                    else:
                        data[i] = item.to_array()
        else:
            """
            The third case is when an object is given and it can be encoded.
            """
            data = data.to_array()

        return data

    @staticmethod
    def decode(data):
        """
        A function that recursively decodes cached data.
        By decoded, it means that objects are created where necessary or possible.
        Only classes that inherit the :class:`~objects.exportable.Exportable` can be decoded.
        This is done through the :func:`~objects.exportable.Exportable.from_array` function.

        .. note::

            When decoding, the function expects either a dictionary or a list.
            JSON objects cannot be anything else.

        :param data: The data to decode.
        :type data: dict or list

        :return: A dictionary, list or object, but this time decoded.
        :rtype: dict or list or object
        """

        data = copy.deepcopy(data)

        if type(data) is dict and 'class' in data:
            """
            The first case is when the dictionary itself represents an object.
            """
            module = importlib.import_module(Exportable.get_module(data.get('class')))
            cls = getattr(module, Exportable.get_class(data.get('class')))
            data = cls.from_array(data)
        elif type(data) is dict:
            """
            The second case is when a dictionary is given and all keys need to be decoded because some may represent an object.
            """
            for key in data:
                if type(data.get(key)) is dict and 'class' in data.get(key):
                    module = importlib.import_module(Exportable.get_module(data.get(key).get('class')))
                    cls = getattr(module, Exportable.get_class(data.get(key).get('class')))
                    data[key] = cls.from_array(data.get(key))
                else:
                    data[key] = Exportable.decode(data.get(key))
        elif type(data) is list:
            """
            The second case is when a list is given and all items need to be decoded because some may represent an object.
            """
            for i, item in enumerate(data):
                if type(item) is dict and 'class' in item:
                    module = importlib.import_module(Exportable.get_module(item.get('class')))
                    cls = getattr(module, Exportable.get_class(item.get('class')))
                    data[i] = cls.from_array(item)
                else:
                    data[i] = Exportable.decode(item)

        return data

    @staticmethod
    def get_module(cls):
        """
        Get the module name from the given path.

        :param cls: The full class name.
        :type cls: str

        :return: The module name.
        :rtype: str

        :raises ValueError: When the class name is invalid.
        """

        class_pattern = re.compile('<class \'(.+)?\.?\'>')
        if not class_pattern.match(cls):
            raise ValueError(f"Invalid class name {cls}")

        path = class_pattern.findall(cls)[0]

        """
        If the path is an alias, replace it with the proper package name.
        """
        for alias in Exportable.ALIASES:
            if path.startswith(alias):
                path = path.replace(alias, Exportable.ALIASES[alias])

        path = path.split('.')

        return '.'.join(path[:-1])

    @staticmethod
    def get_class(cls):
        """
        Get the class name from the given path.

        :param cls: The full class name.
        :type cls: str

        :return: The class name.
        :rtype: str

        :raises ValueError: When the class name is invalid.
        """

        class_pattern = re.compile('<class \'(.+)?\.?\'>')
        if not class_pattern.match(cls):
            raise ValueError(f"Invalid class name {cls}")

        path = class_pattern.findall(cls)[0].split('.')
        return path[-1]
