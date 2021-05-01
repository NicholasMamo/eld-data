"""
Attributable objects are normal objects that can have attributes.
"""

class Attributable(object):
    """
    Attributable objects have a dictionary that stores attributes.

    :ivar attributes: The object's attributes.
                      The keys are the attribute names and the values are the respective values.
    :vartype attributes: dict
    """

    def __init__(self, attributes=None):
        """
        Initialize the object with the attributes.

        :param attributes: The starting attributes.
                           If ``None`` is given, an empty dictionary is created instead.
        :type attributes: dict or None
        """

        self.attributes = dict(attributes or { })
