"""
Volatility technical indicators.

This module implements indicators that measure market volatility.

Classes:
    RollingStdDev: Efficient rolling standard deviation.
"""

import math
from collections import deque
from typing import Deque

class RollingStdDev:
    """
    Efficient O(1) rolling standard deviation calculator.

    This implementation uses a running sum and sum of squares to calculate
    standard deviation in O(1) time for each update.

    Mathematical Formula:
        variance = (N * sum(x^2) - (sum(x))^2) / (N * (N-1))
        std_dev = sqrt(variance)

    Attributes:
        period (int): The lookback period for the calculation.
        is_ready (bool): Whether the indicator has sufficient data.
        value (float): The current standard deviation value.
    """

    def __init__(self, period: int):
        """
        Initialize the RollingStdDev calculator.

        Args:
            period (int): The lookback period. Must be > 1.
        """
        if not isinstance(period, int) or period <= 1:
            raise ValueError("Period must be an integer greater than 1.")

        self.period = period
        self._values: Deque[float] = deque(maxlen=period)
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, value: float) -> None:
        """
        Update the rolling standard deviation with a new value.

        Args:
            value (float): The new data point.
        """
        if len(self._values) == self.period:
            old_value = self._values[0]
            self._sum -= old_value
            self._sum_sq -= old_value ** 2

        self._values.append(value)
        self._sum += value
        self._sum_sq += value ** 2

    @property
    def is_ready(self) -> bool:
        """
        Check if the calculator has enough data.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return len(self._values) == self.period

    @property
    def value(self) -> float:
        """
        Get the current standard deviation.

        Returns:
            float: The standard deviation, or 0.0 if not ready.
        """
        if not self.is_ready:
            return 0.0

        # Using population standard deviation formula: variance = (E[X^2] - (E[X])^2)
        # which is equivalent to (n*sum(x^2) - (sum(x))^2) / (n^2)
        n = self.period
        sum_x = self._sum
        sum_x_sq = self._sum_sq
        
        numerator = n * sum_x_sq - sum_x ** 2
        denominator = n * n
        
        if denominator <= 0:
            return 0.0
            
        variance = numerator / denominator
        return math.sqrt(variance) if variance > 0 else 0.0

    def reset(self) -> None:
        """Reset the calculator to its initial state."""
        self._values.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
