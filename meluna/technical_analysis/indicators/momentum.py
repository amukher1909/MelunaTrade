"""
Momentum technical indicators.

This module implements indicators that measure the speed and strength of price movements.
All indicators use O(1) streaming updates for maximum performance.

Classes:
    RSI: Relative Strength Index with configurable smoothing strategies
"""

import math
from typing import Dict, Any, Optional, Literal
from collections import deque

from ..base import BaseIndicator
from ..exceptions import InvalidParameterError
from .smoothing import SmoothingStrategy, WildersSmoothing, EmaSmoothing


class RSI(BaseIndicator):
    """
    Relative Strength Index (RSI) momentum indicator.
    
    Measures the speed and change of price movements to identify
    overbought/oversold conditions. Uses configurable smoothing strategies
    for different calculation methods.
    
    Mathematical Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Gains = max(0, current_price - previous_price)  
        Losses = max(0, previous_price - current_price)
    
    Smoothing Methods:
        - 'wilders': Original Wilder's smoothing (α = 1/N)
        - 'ema': Standard EMA smoothing (α = 2/(N+1))
    
    Features:
        - Configurable smoothing strategy via Strategy pattern
        - Proper handling of zero average loss edge case
        - Values bounded between 0-100 automatically
        - O(1) streaming updates after warm-up period
        - Thread-safe operations
    
    Example:
        >>> # Standard RSI with Wilder's smoothing (most common)
        >>> rsi = RSI(period=14, smoothing_strategy='wilders')
        >>> 
        >>> # RSI with EMA smoothing for faster response
        >>> rsi_ema = RSI(period=14, smoothing_strategy='ema')
        >>> 
        >>> for bar in market_data:
        ...     rsi.update(bar)
        ...     if rsi.is_ready:
        ...         value = rsi.value
        ...         if value > 70:
        ...             print(f"Overbought: RSI = {value:.1f}")
        ...         elif value < 30:
        ...             print(f"Oversold: RSI = {value:.1f}")
    """
    
    required_inputs = ('close',)  # Default, overridden by input_field parameter
    
    def __init__(
        self, 
        period: int, 
        input_field: str = 'close',
        smoothing_strategy: Literal['wilders', 'ema'] = 'wilders'
    ):
        """
        Initialize Relative Strength Index indicator.
        
        Args:
            period (int): Number of periods for RSI calculation.
                Standard period is 14. Must be positive integer >= 2.
            input_field (str): OHLCV field to use for calculation.
                Defaults to 'close'. Common values: 'open', 'high', 'low', 'close'.
            smoothing_strategy (str): Smoothing method to use for gain/loss averages.
                'wilders': Original Wilder's smoothing (α = 1/period)
                'ema': Standard EMA smoothing (α = 2/(period+1))
                
        Raises:
            InvalidParameterError: If period < 2 or smoothing_strategy is invalid.
        """
        # RSI requires at least 2 periods for gain/loss calculation
        if not isinstance(period, int) or period < 2:
            raise InvalidParameterError(
                "period", period, "integer >= 2 (need at least 2 periods for gain/loss)", "RSI"
            )
        
        super().__init__(period, input_field)
        
        # Override required_inputs based on input_field parameter
        self.required_inputs = (input_field,)
        
        # Set ready threshold to period + 1 (need one extra for first gain/loss calculation)
        self._ready_threshold = period + 1
        
        # Create smoothing strategies for gains and losses
        if smoothing_strategy == 'wilders':
            self._gain_smoother = WildersSmoothing(period)
            self._loss_smoother = WildersSmoothing(period)
        elif smoothing_strategy == 'ema':
            self._gain_smoother = EmaSmoothing(period)
            self._loss_smoother = EmaSmoothing(period)
        else:
            raise InvalidParameterError(
                "smoothing_strategy", 
                smoothing_strategy, 
                "either 'wilders' or 'ema'", 
                "RSI"
            )
        
        self.smoothing_strategy = smoothing_strategy
        
        # Track previous price for gain/loss calculation
        self._previous_price: Optional[float] = None
        
        # Current RSI value
        self._rsi_value: Optional[float] = None
    
    def update(self, data_point: Dict[str, Any]) -> None:
        """
        Process a new data point and update the RSI value.
        
        Algorithm:
        1. Calculate gain/loss from price change
        2. Update smoothed averages of gains and losses
        3. Calculate Relative Strength (RS = avg_gain / avg_loss)
        4. Convert to RSI = 100 - (100 / (1 + RS))
        
        Args:
            data_point (Dict[str, Any]): Market data containing required input field.
                
        Raises:
            MissingInputError: If required input field is missing.
            InvalidDataError: If input data contains invalid values.
        """
        # Validate input data
        self._validate_input_data(data_point)
        
        # Extract current price
        current_price = data_point[self.input_field]
        
        # Add to buffer for is_ready calculation
        self._buffer.append(current_price)
        
        # Skip first data point (need previous price for gain/loss)
        if self._previous_price is None:
            self._previous_price = current_price
            self._update_metadata(data_point)
            self._store_output(self.value)
            return
        
        # Calculate price change
        price_change = current_price - self._previous_price
        
        # Calculate gain and loss
        gain = max(0.0, price_change)
        loss = max(0.0, -price_change)
        
        # Update smoothed averages
        avg_gain = self._gain_smoother.update(gain)
        avg_loss = self._loss_smoother.update(loss)
        
        # Calculate RSI
        if avg_loss == 0:
            # Special case: no losses means RSI = 100
            self._rsi_value = 100.0
        else:
            # Standard RSI calculation
            rs = avg_gain / avg_loss
            self._rsi_value = 100.0 - (100.0 / (1.0 + rs))
        
        # Update for next iteration
        self._previous_price = current_price
        
        # Update metadata and store output
        self._update_metadata(data_point)
        self._store_output(self.value)
    
    @property
    def value(self) -> float:
        """
        Get the current RSI value.
        
        Returns:
            float: Current RSI value between 0-100, or NaN during warm-up period.
                Values > 70 typically indicate overbought conditions.
                Values < 30 typically indicate oversold conditions.
        """
        if not self.is_ready or self._rsi_value is None:
            return math.nan
        
        return self._rsi_value
    
    @property
    def average_gain(self) -> float:
        """
        Get the current smoothed average gain value.
        
        Returns:
            float: Current smoothed average of gains, or NaN if not ready.
        """
        if not self.is_ready or self._gain_smoother.value is None:
            return math.nan
        return self._gain_smoother.value
    
    @property
    def average_loss(self) -> float:
        """
        Get the current smoothed average loss value.
        
        Returns:
            float: Current smoothed average of losses, or NaN if not ready.
        """
        if not self.is_ready or self._loss_smoother.value is None:
            return math.nan
        return self._loss_smoother.value
    
    @property
    def relative_strength(self) -> float:
        """
        Get the current Relative Strength (RS) ratio.
        
        Returns:
            float: RS = average_gain / average_loss, or NaN if not ready.
                Used in RSI calculation: RSI = 100 - (100 / (1 + RS))
        """
        if not self.is_ready:
            return math.nan
        
        avg_gain = self.average_gain
        avg_loss = self.average_loss
        
        if math.isnan(avg_gain) or math.isnan(avg_loss):
            return math.nan
        
        if avg_loss == 0:
            return math.inf  # Infinite RS when no losses
        
        return avg_gain / avg_loss
    
    def reset(self) -> None:
        """
        Reset the indicator to its initial state.
        
        This includes resetting the smoothing strategies.
        """
        super().reset()
        self._previous_price = None
        self._rsi_value = None
        self._gain_smoother.reset()
        self._loss_smoother.reset()