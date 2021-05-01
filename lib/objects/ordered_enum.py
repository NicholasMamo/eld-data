"""
An enum that supports ordering using comparison.
Used when order is important.
"""

from enum import Enum

class OrderedEnum(Enum):
    """
    An enum that supports order comparisons.
    Outlined in https://docs.python.org/3/library/enum.html#orderedenum
    """

    def __eq__(self, other):
        """
        Equality comparison.

        :param other: The other value to compare with.
        :type other: :class:`~objects.ordered_enum.OrderedEnum`
        """

        if self.__class__ == other.__class__:
            return self.value == other.value
        else:
            return self.value == other

    def __ge__(self, other):
        """
        Greater than or equal to comparison.

        :param other: The other value to compare with.
        :type other: :class:`~objects.ordered_enum.OrderedEnum`
        """

        if self.__class__ == other.__class__:
            return self.value >= other.value
        else:
            return self.value >= other

    def __le__(self, other):
        """
        Less than or equal to comparison.

        :param other: The other value to compare with.
        :type other: :class:`~objects.ordered_enum.OrderedEnum`
        """

        if self.__class__ == other.__class__:
            return self.value <= other.value
        else:
            return self.value <= other

    def __gt__(self, other):
        """
        Greater than comparison.

        :param other: The other value to compare with.
        :type other: :class:`~objects.ordered_enum.OrderedEnum`
        """

        if self.__class__ == other.__class__:
            return self.value > other.value
        else:
            return self.value > other

    def __lt__(self, other):
        """
        Less than comparison.

        :param other: The other value to compare with.
        :type other: :class:`~objects.ordered_enum.OrderedEnum`
        """

        if self.__class__ == other.__class__:
            return self.value < other.value
        else:
            return self.value < other
