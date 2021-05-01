"""
The Term Frequency-Inverse Document Frequency (TF-IDF) term-weighting scheme is one of the most popular schemes.
The scheme promotes features that appear commonly in a document, but rarely outside of it.
The TFIDF is simply the multiplication of the :class:`~nlp.weighting.local_schemes.tf.TF` and :class:`~nlp.weighting.global_schemes.idf.IDF` term-weighting schemes:
The weight :math:`tfidf_{t,d}` of term :math:`t` in document :math:`d` is computed as follows:

.. math::

    tfidf_{t,d} = tf_{t,d} \\cdot idf_{t}

where :math:`tf_{t,d}` is the score assignedd by the :class:`~nlp.weighting.local_schemes.tf.TF` scheme, and :math:`idf_{t}` is the score assigned by the :class:`~nlp.weighting.global_schemes.idf.IDF` scheme.
"""

import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from objects.exportable import Exportable
from nlp.weighting import TermWeightingScheme

class TFIDF(Exportable, TermWeightingScheme):
    """
    This class is a quick way of instantiating a :class:`~nlp.weighting.TermWeightingScheme` that represents the Term Frequency-Inverse Document Frequency (TF-IDF) scheme.
    It automatically constructs the scheme from the specifications required to create the :class:`~nlp.weighting.global_schemes.idf.IDF` scheme.
    The result is a :class:`~nlp.weighting.TermWeightingScheme` with the following components:

    1. :class:`~nlp.weighting.local_schemes.tf.TF` as a local term-weighting scheme, and
    2. :class:`~nlp.weighting.global_schemes.idf.IDF` as a global term-weighting scheme (from the given specifications).

    Since this term-weighting scheme is so common, this class provides functionality to export it (using the :func:`~nlp.weighting.tfidf.TFIDF.to_array` function) and import it (using the :func:`~nlp.weighting.tfidf.TFIDF.from_array` function) back again.
    """

    def __init__(self, idf, documents):
        """
        Initialize the TF-IDF term-weighting scheme by supplying details about the :class:`~nlp.weighting.global_schemes.idf.IDF` scheme.
        These include the IDF table and the number of documents in it.

        :param idf: The IDF table used in conjunction with term weighting.
                    The keys are the terms, and the corresponding values are the number of documents in which they appear.
        :type idf: dict
        :param documents: The number of documents in the IDF table.
        :type documents: int
        """

        # NOTE: The imports are located here because of circular dependencies
        from nlp.weighting.local_schemes.tf import TF
        from nlp.weighting.global_schemes.idf import IDF

        tf = TF()
        idf = IDF(idf, documents)
        super(TFIDF, self).__init__(tf, idf)

    def to_array(self):
        """
        Export the TF-IDF as an associative array.

        :return: The TF-IDF as an associative array.
        :rtype: dict
        """

        return {
            'class': str(TFIDF),
            'idf': self.global_scheme.to_array(),
        }

    @staticmethod
    def from_array(array):
        """
        Create an instance of the TF-IDF from the given associative array.

        :param array: The associative array with the attributes to create the TF-IDF.
        :type array: dict

        :return: A new instance of the TF-IDF with the same attributes stored in the object.
        :rtype: :class:`~nlp.weighting.tfidf.TFIDF`
        """

        return TFIDF(idf=array.get('idf').get('idf'), documents=array.get('idf').get('documents'))
