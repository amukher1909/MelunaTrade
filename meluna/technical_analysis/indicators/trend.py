"""
Trend-following technical indicators.

This module implements moving average indicators that follow price trends.
All indicators use O(1) streaming updates for maximum performance.

Classes:
    SMA: Simple Moving Average with O(1) rolling sum technique
    EMA: Exponential Moving Average with SMA seeding during warm-up
"""

import math
from typing import Dict, Any, Optional
from collections import deque

from ..base import BaseIndicator
from ..exceptions import InvalidParameterError


class SMA(BaseIndicator):
    """
    Simple Moving Average (SMA) indicator.
    
    Calculates the arithmetic mean of prices over a specified period using
    an efficient O(1) rolling sum technique that avoids recalculation.
    
    Mathematical Formula:
        SMA = (P1 + P2 + ... + Pn) / n
        
    For streaming updates:
        new_SMA = old_SMA + (new_price - oldest_price) / n
    
    Features:
        - O(1) incremental updates using running sum
        - Proper warm-up period handling (returns NaN until ready)
        - Memory efficient using deque with maxlen
        - Thread-safe operations
        - Configurable input field (close, open, high, low)
    
    Example:
        >>> sma = SMA(period=20, input_field='close')
        >>> for bar in market_data:
        ...     sma.update(bar)
        ...     if sma.is_ready:
        ...         print(f"SMA(20): {sma.value:.2f}")
    """
    
    required_inputs = ('close',)  # Default, overridden by input_field parameter
    
    def __init__(self, period: int, input_field: str = 'close'):
        """
        Initialize Simple Moving Average indicator.
        
        Args:
            period (int): Number of periods for the moving average calculation.
                Must be positive integer >= 1.
            input_field (str): OHLCV field to use for calculation.
                Defaults to 'close'. Common values: 'open', 'high', 'low', 'close'.
                
        Raises:
            InvalidParameterError: If period is not a positive integer.
        """
        super().__init__(period, input_field)
        
        # Override required_inputs based on input_field parameter
        self.required_inputs = (input_field,)
        
        # Running sum for O(1) calculation efficiency
        self._sum = 0.0
    
    def update(self, data_point: Dict[str, Any]) -> None:
        """
        Process a new data point and update the SMA value.
        
        Uses efficient O(1) rolling sum technique:
        1. If buffer is full, remove oldest value from sum
        2. Add new value to buffer and sum
        3. Calculate average from current sum
        
        Args:
            data_point (Dict[str, Any]): Market data containing required input field.
                Must contain the field specified in input_field parameter.
                
        Raises:
            MissingInputError: If required input field is missing.
            InvalidDataError: If input data contains invalid values (NaN, None, inf).
        """
        # Validate input data contains required fields and values
        self._validate_input_data(data_point)
        
        # Extract the value for our calculation
        value = data_point[self.input_field]
        
        # Remove oldest value from sum if buffer is at capacity
        if len(self._buffer) == self.period:
            oldest_value = self._buffer[0]  # Get value that will be removed
            self._sum -= oldest_value
        
        # Add new value to buffer and sum
        self._buffer.append(value)
        self._sum += value
        
        # Update metadata and store output
        self._update_metadata(data_point)
        self._store_output(self.value)
    
    @property
    def value(self) -> float:
        """
        Get the current Simple Moving Average value.
        
        Returns:
            float: Current SMA value, or NaN if not enough data points yet.
                Valid values are returned only when is_ready is True.
        """
        if not self.is_ready:
            return math.nan
        
        return self._sum / self.period
    
    def reset(self) -> None:
        """
        Reset the indicator to its initial state.
        
        This includes resetting the running sum.
        """
        super().reset()
        self._sum = 0.0


class EMA(BaseIndicator):
    """
    Exponential Moving Average (EMA) indicator.
    
    Calculates exponentially-weighted moving average that gives more weight
    to recent prices. Uses SMA for initial seeding to ensure mathematical accuracy.
    
    Mathematical Formula:
        EMA_today = α * Price_today + (1-α) * EMA_yesterday
        where α = 2 / (period + 1) by default, or custom alpha if provided
    
    Initialization Strategy:
        - During first N data points: Use SMA for seeding
        - After N points: Switch to exponential calculation
        - This ensures smooth transition and mathematical correctness
    
    Features:
        - Configurable smoothing factor (alpha parameter)
        - SMA seeding during warm-up period for accuracy
        - O(1) updates after initialization
        - Proper handling of initialization edge cases
        - Thread-safe operations
    
    Example:
        >>> # Standard EMA with default alpha = 2/(period+1)
        >>> ema = EMA(period=12, input_field='close')
        >>> 
        >>> # Custom alpha for different smoothing
        >>> ema_custom = EMA(period=12, alpha=0.1, input_field='close')
        >>> 
        >>> for bar in market_data:
        ...     ema.update(bar)
        ...     if ema.is_ready:
        ...         print(f"EMA(12): {ema.value:.2f}")
    """
    
    required_inputs = ('close',)  # Default, overridden by input_field parameter
    
    def __init__(self, period: int, input_field: str = 'close', alpha: Optional[float] = None):
        """
        Initialize Exponential Moving Average indicator.
        
        Args:
            period (int): Number of periods for EMA calculation and SMA seeding.
                Must be positive integer >= 1.
            input_field (str): OHLCV field to use for calculation.
                Defaults to 'close'. Common values: 'open', 'high', 'low', 'close'.
            alpha (Optional[float]): Custom smoothing factor between 0 and 1.
                If None, uses standard EMA formula: α = 2/(period+1).
                Higher values give more weight to recent data.
                
        Raises:
            InvalidParameterError: If period is not positive or alpha is out of range.
        """
        super().__init__(period, input_field)
        
        # Override required_inputs based on input_field parameter
        self.required_inputs = (input_field,)
        
        # Validate and set alpha parameter
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or not 0 < alpha <= 1:
                raise InvalidParameterError("alpha", alpha, "value between 0 and 1", "EMA")
            self._alpha = alpha
        else:
            # Standard EMA formula: α = 2/(N+1)
            self._alpha = 2.0 / (period + 1)
        
        # SMA for initial seeding during warm-up period
        self._sma_seed: Optional[SMA] = None
        if period > 1:
            self._sma_seed = SMA(period=period, input_field=input_field)
        
        # Current EMA value (None until seeded)
        self._ema_value: Optional[float] = None
    
    def update(self, data_point: Dict[str, Any]) -> None:
        """
        Process a new data point and update the EMA value.
        
        Implementation strategy:
        1. During seeding phase: Update internal SMA until it's ready
        2. When SMA becomes ready: Initialize EMA with SMA value
        3. After initialization: Use standard EMA calculation
        
        Args:
            data_point (Dict[str, Any]): Market data containing required input field.
                
        Raises:
            MissingInputError: If required input field is missing.
            InvalidDataError: If input data contains invalid values.
        """
        # Validate input data
        self._validate_input_data(data_point)
        
        # Extract the value for calculation
        value = data_point[self.input_field]
        
        # Handle seeding phase for period > 1
        if self._sma_seed is not None and self._ema_value is None:
            # Update SMA seeding indicator
            self._sma_seed.update(data_point)
            
            # Check if SMA is ready to seed our EMA
            if self._sma_seed.is_ready:
                self._ema_value = self._sma_seed.value
                # We're now initialized, no longer need SMA
                self._sma_seed = None
        
        # Handle special case for period = 1 (EMA = current value)
        elif self.period == 1:
            self._ema_value = value
        
        # Standard EMA calculation for initialized indicator
        elif self._ema_value is not None:
            # EMA_new = α * value_new + (1-α) * EMA_old
            self._ema_value = self._alpha * value + (1 - self._alpha) * self._ema_value
        
        # Update our buffer for is_ready calculation
        self._buffer.append(value)
        
        # Update metadata and store output
        self._update_metadata(data_point)
        self._store_output(self.value)
    
    @property
    def value(self) -> float:
        """
        Get the current Exponential Moving Average value.
        
        Returns:
            float: Current EMA value, or NaN during warm-up/seeding period.
                Valid values returned only when is_ready is True and EMA is initialized.
        """
        if not self.is_ready or self._ema_value is None:
            return math.nan
        
        return self._ema_value
    
    @property
    def alpha(self) -> float:
        """
        Get the smoothing factor (alpha) used by this EMA.
        
        Returns:
            float: The alpha value used in EMA calculation.
        """
        return self._alpha
    
    def reset(self) -> None:
        """
        Reset the indicator to its initial state.
        
        This includes resetting the SMA seeding indicator if present.
        """
        super().reset()
        self._ema_value = None
        
        # Recreate SMA seeding indicator if needed
        if self.period > 1:
            self._sma_seed = SMA(period=self.period, input_field=self.input_field)
        else:
            self._sma_seed = None