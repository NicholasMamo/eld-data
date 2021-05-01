"""
External resolvers map candidate participants in the event's corpus to an alternate representation.
These resolvers are useful because they can transform participants into semantic concepts that are not available in the corpus.
"""

from .wikipedia_name_resolver import WikipediaNameResolver
from .wikipedia_search_resolver import WikipediaSearchResolver
