"""
Min/Max technical indicators.

This module implements indicators that track rolling minimum and maximum values.

Classes:
    RollingMinMax: Efficient rolling min/max calculator.
"""

from collections import deque
from typing import Deque, Tuple

class RollingMinMax:
    """
    Efficient O(1) amortized rolling min/max calculator.

    This implementation uses a deque to keep track of potential min/max values.
    This is more efficient than recalculating min/max on the entire window every time.

    Attributes:
        period (int): The lookback period for the calculation.
        is_ready (bool): Whether the indicator has sufficient data.
        min (float): The current minimum value in the window.
        max (float): The current maximum value in the window.
    """

    def __init__(self, period: int):
        """
        Initialize the RollingMinMax calculator.

        Args:
            period (int): The lookback period. Must be > 0.
        """
        if not isinstance(period, int) or period <= 0:
            raise ValueError("Period must be a positive integer.")

        self.period = period
        self._values: Deque[float] = deque(maxlen=period)
        self._min_deque: Deque[Tuple[float, int]] = deque()
        self._max_deque: Deque[Tuple[float, int]] = deque()
        self._count = 0

    def update(self, value: float) -> None:
        """
        Update the rolling min/max with a new value.

        Args:
            value (float): The new data point.
        """
        # Remove indices from deques that are out of the window
        if self._min_deque and self._min_deque[0][1] <= self._count - self.period:
            self._min_deque.popleft()
        if self._max_deque and self._max_deque[0][1] <= self._count - self.period:
            self._max_deque.popleft()

        # Remove values from deques that are "dominated" by the new value
        while self._min_deque and self._min_deque[-1][0] >= value:
            self._min_deque.pop()
        self._min_deque.append((value, self._count))

        while self._max_deque and self._max_deque[-1][0] <= value:
            self._max_deque.pop()
        self._max_deque.append((value, self._count))

        self._values.append(value)
        self._count += 1

    @property
    def is_ready(self) -> bool:
        """
        Check if the calculator has enough data.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return len(self._values) == self.period

    @property
    def min(self) -> float:
        """
        Get the current minimum value.

        Returns:
            float: The minimum value, or float('inf') if not ready.
        """
        return self._min_deque[0][0] if self._min_deque else float('inf')

    @property
    def max(self) -> float:
        """
        Get the current maximum value.

        Returns:
            float: The maximum value, or float('-inf') if not ready.
        """
        return self._max_deque[0][0] if self._max_deque else float('-inf')

    def reset(self) -> None:
        """Reset the calculator to its initial state."""
        self._values.clear()
        self._min_deque.clear()
        self._max_deque.clear()
        self._count = 0
