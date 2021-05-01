"""
Text mining tasks generally depend a lot on the document representation.
EvenTDT provides functionality to make it easier to work with documents:

1. The :class:`~nlp.document.Document` class, based on the :class:`~vsm.vector.Vector` class, but with text-specific functionality;
2. The :class:`~nlp.tokenizer.Tokenizer` class to split a piece of text into tokens; and
3. The :class:`~nlp.weighting.TermWeightingScheme` abstract class, as well as different term-weighting schemes, to assign weight to terms.
"""

from .document import Document
from .tokenizer import Tokenizer
