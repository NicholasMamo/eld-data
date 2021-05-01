"""
External extrapolators look for new participants outside of the event's corpus.
This allows extrapolation to bypass limitations in the corpus.
For example, a corpus may be too small, or too biased, to capture all of the event's participants.
"""

from .wikipedia_extrapolator import WikipediaExtrapolator
