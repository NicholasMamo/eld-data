"""
External post-processors use information from outside the event's corpus to modify participants.
These post-processors are useful when resolution or extrapolation maps participants to semantic concepts, like Wikipedia articles.
In these cases, the extra information can be used to post-process participants.
"""

from .wikipedia_postprocessor import WikipediaPostprocessor
