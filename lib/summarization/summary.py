"""
The :class:`~summarization.summary.Summary` object is a special class that represents summaries.
Essentially, this class is a group of :class:`~nlp.document.Document`, which can represent sentences, tweets and other text.
Moreover, it includes additional, summary-related functionality and it can accept additional attributes.

The point of the :class:`~summarization.summary.Summary` object is that it retains the original :class:`~nlp.document.Document` instances.
Therefore a :class:`~summarization.summary.Summary` instance is not a simple string.
It can be represented or manipulated (such as by using a :class:`~nlp.cleaners.Cleaner`) as the application necessitates by having access to its original components.
"""

import importlib
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..')
if path not in sys.path:
    sys.path.insert(1, path)

from objects.attributable import Attributable
from objects.exportable import Exportable

class Summary(Attributable, Exportable):
    """
    A summary is essentially a list of :class:`~nlp.document.Document`.
    However, summaries may also have other attributes.

    The point of the summary object is to encapsulate the list of :class:`~nlp.document.Document` instances that make it up, while retaining the original text.
    The list of :class:`~nlp.document.Document` can be accessed using the ``documents`` property, which means that :class:`~nlp.document.Document` instances can be added or removed dynamically.

    In addition, this class provides the :func:`~summarization.summary.Summary.__str__` special function, which concatenates its documents to form a textual summary.
    If the documents contain a ``timestamp`` attribute, it automatically sorts the documents in chronological order.

    :ivar ~.documents: The documents that make up the summary.
    :vartype ~.documents: list of :class:`~nlp.document.Document`
    """

    def __init__(self, documents=None, *args, **kwargs):
        """
        Create the summary.

        :param documents: A list of documents that make up the summary.
        :type documents: list of :class:`~nlp.document.Document`
        """

        super(Summary, self).__init__(*args, **kwargs)
        self.documents = documents

    @property
    def documents(self):
        """
        Get the list of documents in the summary.

        :return: The list of documents.
        :rtype: list of :class:`~nlp.document.Document`
        """

        return self.__documents

    @documents.setter
    def documents(self, documents=None):
        """
        Override the documents.

        :param documents: The new documents.
        :type documents: list of :class:`~nlp.document.Document` or :class:`~nlp.document.Document` or None
        """

        if documents is None:
            self.__documents = [ ]
        elif type(documents) is list:
            self.__documents = list(documents)
        else:
            self.__documents = [ documents ]

    def __str__(self):
        """
        Get the string representation of the summary.
        This is equivalent to concatenating the text of all documents.

        If all documents have a ``timestamp`` attribute, this function automatically sorts them in chronological order.

        :return: The string representation of the summary.
        :rtype: str
        """

        if all( 'timestamp' in document.attributes for document in self.documents ):
            documents = sorted(self.documents, key=lambda document: float(document.attributes.get('timestamp')))
            return ' '.join([ document.text for document in documents ])

        return ' '.join([ document.text for document in self.documents ])

    def to_array(self):
        """
        Export the summary as a dictionary.

        :return: The summary as a dictionary.
        :rtype: dict
        """

        return {
            'class': str(Summary),
            'attributes': self.attributes,
            'documents': [ document.to_array() for document in self.documents ]
        }

    @staticmethod
    def from_array(array):
        """
        Create a summary instance from the given dictionary.

        :param array: The dictionary with the necessary information to re-create the summary.
        :type array: dict

        :return: A new summary.
        :rtype: :class:`~summarization.summary.Summary`
        """

        documents = [ ]
        for vector in array.get('documents'):
            module = importlib.import_module(Exportable.get_module(vector.get('class')))
            cls = getattr(module, Exportable.get_class(vector.get('class')))
            documents.append(cls.from_array(vector))

        return Summary(documents=documents, attributes=array.get('attributes'))
