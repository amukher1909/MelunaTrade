"""
Smoothing strategy classes for technical indicators.

This module implements the Strategy pattern for different smoothing algorithms
used in technical indicators, particularly for RSI calculation variants.

Classes:
    SmoothingStrategy: Abstract base class for smoothing algorithms
    WildersSmoothing: Wilder's exponential smoothing (α = 1/N)
    EmaSmoothing: Standard exponential moving average smoothing (α = 2/(N+1))
"""

from abc import ABC, abstractmethod
from typing import Optional


class SmoothingStrategy(ABC):
    """
    Abstract base class for smoothing algorithms used in technical indicators.
    
    This enables the Strategy pattern for different smoothing methods,
    particularly important for RSI calculations where Wilder's smoothing
    differs from standard EMA smoothing.
    """
    
    def __init__(self, period: int):
        """
        Initialize the smoothing strategy.
        
        Args:
            period (int): The smoothing period for the algorithm.
        """
        self.period = period
        self._current_value: Optional[float] = None
    
    @abstractmethod
    def get_alpha(self) -> float:
        """
        Get the smoothing factor (alpha) for this strategy.
        
        Returns:
            float: The alpha value used for exponential smoothing.
        """
        pass
    
    def update(self, new_value: float) -> float:
        """
        Update the smoothed value with a new data point.
        
        Args:
            new_value (float): New value to incorporate into smoothed result.
            
        Returns:
            float: The updated smoothed value.
        """
        if self._current_value is None:
            # First value - no smoothing needed
            self._current_value = new_value
        else:
            # Apply exponential smoothing: new_smooth = α * new_value + (1-α) * old_smooth
            alpha = self.get_alpha()
            self._current_value = alpha * new_value + (1 - alpha) * self._current_value
        
        return self._current_value
    
    @property
    def value(self) -> Optional[float]:
        """
        Get the current smoothed value.
        
        Returns:
            Optional[float]: Current smoothed value or None if no updates yet.
        """
        return self._current_value
    
    def reset(self) -> None:
        """Reset the smoothing strategy to initial state."""
        self._current_value = None


class WildersSmoothing(SmoothingStrategy):
    """
    Wilder's exponential smoothing implementation.
    
    Uses α = 1/N as the smoothing factor, which is different from standard EMA.
    This is the original smoothing method used by J. Welles Wilder Jr. in RSI
    and other indicators like ATR, ADX.
    
    Mathematical Formula:
        α = 1 / period
        smoothed_value = α * new_value + (1-α) * previous_smoothed_value
    """
    
    def get_alpha(self) -> float:
        """
        Get Wilder's smoothing factor.
        
        Returns:
            float: Alpha value of 1/period for Wilder's smoothing.
        """
        return 1.0 / self.period


class EmaSmoothing(SmoothingStrategy):
    """
    Standard Exponential Moving Average smoothing implementation.
    
    Uses α = 2/(N+1) as the smoothing factor, which is the standard EMA formula.
    This provides more weight to recent values compared to Wilder's smoothing.
    
    Mathematical Formula:
        α = 2 / (period + 1)
        smoothed_value = α * new_value + (1-α) * previous_smoothed_value
    """
    
    def get_alpha(self) -> float:
        """
        Get standard EMA smoothing factor.
        
        Returns:
            float: Alpha value of 2/(period+1) for standard EMA smoothing.
        """
        return 2.0 / (self.period + 1)