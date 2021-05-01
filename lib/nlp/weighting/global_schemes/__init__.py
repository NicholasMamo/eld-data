"""
Global term-weighting schemes assign weights to tokens based on the contents of a corpus, and not just the document they appear appear in.
This kind of term-weighting schemes is usually accompanied by :ref:`a local term-weighting scheme <nlp_local>`.
"""

from .idf import IDF
