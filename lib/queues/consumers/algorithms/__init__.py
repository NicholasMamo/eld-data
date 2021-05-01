"""
In addition to the base and simple consumers, EvenTDT includes consumers that replicate methods proposed in literature.
These approaches are as faithful to the original techniques as possible, although they depend a lot on the details available in the respective papers.

You can use these consumers as baselines or to have a working base from where to start implementing your own consumers.
To run these consumers, check out the :mod:`~tools.consume` tool.

.. warning::

    In many cases, these consumers replicate most faithfully the TDT approach, not the summarization approach.
"""

from .eld_consumer import ELDConsumer
from .zhao_consumer import ZhaoConsumer
