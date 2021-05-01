"""
The :class:`~summarization.timeline.nodes.document_node.DocumentNode` is a simple :class:`~summarization.timeline.nodes.Node` that only stores :class:`~nlp.document.Document` instances.
"""

import importlib
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from objects.exportable import Exportable
from summarization.timeline.nodes import Node
from vsm import vector_math
from vsm.clustering import Cluster

class DocumentNode(Node):
    """
    The :class:`~summarization.timeline.nodes.document_node.DocumentNode` class only stores :class:`~nlp.document.Document` instances.
    When it needs to compare incoming :class:`nlp.document.Document` instances, it compares them against the centroid of its own information.

    :ivar ~.documents: The list of documents in this node.
    :vartype ~.documents: list of :class:`~nlp.document.Document`
    """

    def __init__(self, created_at, documents=None):
        """
        Create the node with an optional initial list of :class:`~nlp.document.Document` instances.

        :param created_at: The timestamp when the node was created.
        :type created_at: float
        :param documents: The initial list of documents in this node.
        :type documents: None or list of :class:`~nlp.document.Document`
        """

        super(DocumentNode, self).__init__(created_at)
        self.documents = documents or [ ]

    def add(self, documents, *args, **kwargs):
        """
        Add documents to the node.

        :param documents: A document, or a list of documents to add to the node.
        :type documents: :class:`~nlp.document.Document` or list of :class:`~nlp.document.Document`
        """

        if type(documents) is list:
            for document in documents:
                self.add(document, *args, **kwargs)
        else:
            document = documents
            if document not in self.documents:
                self.documents.append(document)

    def get_all_documents(self, *args, **kwargs):
        """
        Get all the documents in this node.

        :return: A list of documents in the node.
        :rtype: list of :class:`~nlp.document.Document`
        """

        return self.documents

    def similarity(self, documents, *args, **kwargs):
        """
        Compute the similarity between this node's documents and the centroid of the given documents.

        :param documents: The documents with which to compute similarity.
        :type documents: list of :class:`~nlp.document.Docunet`

        :return: The similarity between this node's documents and the given documents.
        :rtype: float
        """

        centroid = Cluster(self.documents).centroid
        centroid.normalize()

        document_centroid = Cluster(documents).centroid
        document_centroid.normalize()

        return vector_math.cosine(centroid, document_centroid)

    def to_array(self):
        """
        Export the document node as an associative array.

        :return: The document node as an associative array.
        :rtype: dict
        """

        return {
            'class': str(DocumentNode),
            'created_at': self.created_at,
            'documents': [ document.to_array() for document in self.documents ],
        }

    @staticmethod
    def from_array(array):
        """
        Create a :class:`~summarization.timeline.nodes.document_node.DocumentNode` instance from the given associative array.

        :param array: The associative array with the attributes to create the :class:`~summarization.timeline.nodes.document_node.DocumentNode`.
        :type array: dict

        :return: A new instance of the :class:`~summarization.timeline.nodes.document_node.DocumentNode` with the same attributes stored in the object.
        :rtype: :class:`~summarization.timeline.nodes.document_node.DocumentNode`
        """

        documents = [ ]
        for document in array.get('documents'):
            module = importlib.import_module(Exportable.get_module(document.get('class')))
            cls = getattr(module, Exportable.get_class(document.get('class')))
            documents.append(cls.from_array(document))

        return DocumentNode(created_at=array.get('created_at'), documents=documents)

    @staticmethod
    def merge(created_at, *args):
        """
        Create a new :class:`~summarization.timeline.nodes.document_node.DocumentNode` by combining the documents in all of the given nodes.
        The new :class:`~summarization.timeline.nodes.document_node.DocumentNode` will have the given timestamp, but it will inherit all of the documents in the nodes.

        All the nodes are provided as additional arguments (using ``*args``).
        If none are given, this function creates an empty :class:`~summarization.timeline.nodes.document_node.DocumentNode`.

        :param created_at: The timestamp when the node was created.
        :type created_at: float

        :return: A new node with all of the data stored in the given nodes.
        :rtype: :class:`~summarization.timeline.nodes.document_node.DocumentNode`
        """

        node = DocumentNode(created_at)

        for _node in args:
            node.add(_node.documents)

        return node
