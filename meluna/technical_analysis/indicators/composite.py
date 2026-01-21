"""
Composite technical indicators.

This module implements composite indicators that are built from other indicators.
"""

import math
from typing import Dict, Any, List

from ..base import BaseIndicator
from ..exceptions import InvalidParameterError
from .trend import SMA, EMA
from .volatility import RollingStdDev
from .minmax import RollingMinMax

class BollingerBands(BaseIndicator):
    """
    Bollinger Bands (BBands) indicator.

    Comprises a middle band (SMA) and upper/lower bands based on standard deviation.

    Mathematical Formula:
        Middle Band = SMA(period)
        Upper Band = Middle Band + (K * StdDev(period))
        Lower Band = Middle Band - (K * StdDev(period))
        Bandwidth = (Upper Band - Lower Band) / Middle Band

    Attributes:
        middle_band (SMA): The middle band (SMA) indicator.
        std_dev (RollingStdDev): The rolling standard deviation calculator.
    """

    required_inputs = ('close',)

    def __init__(self, period: int = 20, k: float = 2.0, input_field: str = 'close'):
        """
        Initialize Bollinger Bands indicator.

        Args:
            period (int): The lookback period for SMA and StdDev.
            k (float): The number of standard deviations for the bands.
            input_field (str): The input field to use.
        """
        super().__init__(period, input_field)
        self.required_inputs = (input_field,)
        
        # Validate k parameter
        if k <= 0:
            raise InvalidParameterError("k", k, "positive number (> 0)")
        
        self._k = k
        
        self.middle_band = SMA(period, input_field)
        
        # Handle period=1 case where standard deviation is 0
        if period == 1:
            self.std_dev = None
        else:
            self.std_dev = RollingStdDev(period)
        
        self._children: List[BaseIndicator] = [self.middle_band]

    def update(self, data_point: Dict[str, Any]) -> None:
        """
        Update the Bollinger Bands with a new data point.

        Args:
            data_point (Dict[str, Any]): The new data point.
        """
        self._validate_input_data(data_point)
        value = data_point[self.input_field]

        self.middle_band.update(data_point)
        if self.std_dev is not None:
            self.std_dev.update(value)
        
        self._update_metadata(data_point)
        self._store_output(self.value)

    @property
    def value(self) -> Dict[str, float]:
        """
        Get the current Bollinger Bands values.

        Returns:
            Dict[str, float]: A dictionary with 'upper', 'middle', 'lower', 'bandwidth'.
        """
        if not self.is_ready:
            return {'upper': math.nan, 'middle': math.nan, 'lower': math.nan, 'bandwidth': math.nan}

        middle = self.middle_band.value
        std_dev_val = self.std_dev.value if self.std_dev is not None else 0.0
        
        upper = middle + self._k * std_dev_val
        lower = middle - self._k * std_dev_val
        
        bandwidth = (upper - lower) / middle if middle != 0 else 0.0

        return {'upper': upper, 'middle': middle, 'lower': lower, 'bandwidth': bandwidth}

    @property
    def is_ready(self) -> bool:
        """
        Check if the indicator is ready.

        Returns:
            bool: True if the middle band is ready, False otherwise.
        """
        return self.middle_band.is_ready

class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.

    A trend-following momentum indicator that shows the relationship between two
    exponential moving averages (EMAs) of a security’s price.

    Mathematical Formula:
        MACD Line = EMA(fast_period) - EMA(slow_period)
        Signal Line = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal Line

    Attributes:
        fast_ema (EMA): The fast EMA indicator.
        slow_ema (EMA): The slow EMA indicator.
        signal_ema (EMA): The signal line EMA indicator.
    """

    required_inputs = ('close',)

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, input_field: str = 'close'):
        """
        Initialize MACD indicator.

        Args:
            fast_period (int): The period for the fast EMA.
            slow_period (int): The period for the slow EMA.
            signal_period (int): The period for the signal line EMA.
            input_field (str): The input field to use.
        """
        # Validate parameters
        if fast_period <= 0:
            raise InvalidParameterError("fast_period", fast_period, "positive integer (> 0)")
        if slow_period <= 0:
            raise InvalidParameterError("slow_period", slow_period, "positive integer (> 0)")
        if signal_period <= 0:
            raise InvalidParameterError("signal_period", signal_period, "positive integer (> 0)")
        if fast_period >= slow_period:
            raise InvalidParameterError("fast_period", fast_period, f"must be less than slow_period ({slow_period})")
        
        # The warm-up period is determined by the slow EMA plus the signal EMA.
        super().__init__(period=slow_period + signal_period - 1, input_field=input_field)
        self.required_inputs = (input_field,)

        self.fast_ema = EMA(fast_period, input_field)
        self.slow_ema = EMA(slow_period, input_field)
        # The signal EMA is calculated on the MACD line, not on the input field.
        # We will feed it manually.
        self.signal_ema = EMA(signal_period, input_field='macd') # input_field is a placeholder

        self._children: List[BaseIndicator] = [self.fast_ema, self.slow_ema, self.signal_ema]
        self._macd_value = math.nan

    def update(self, data_point: Dict[str, Any]) -> None:
        """
        Update the MACD with a new data point.

        Args:
            data_point (Dict[str, Any]): The new data point.
        """
        self._validate_input_data(data_point)

        self.fast_ema.update(data_point)
        self.slow_ema.update(data_point)

        if self.fast_ema.is_ready and self.slow_ema.is_ready:
            self._macd_value = self.fast_ema.value - self.slow_ema.value
            self.signal_ema.update({'macd': self._macd_value})

        self._update_metadata(data_point)
        self._store_output(self.value)

    @property
    def value(self) -> Dict[str, float]:
        """
        Get the current MACD values.

        Returns:
            Dict[str, float]: A dictionary with 'macd', 'signal', 'histogram'.
        """
        if not self.is_ready:
            return {'macd': math.nan, 'signal': math.nan, 'histogram': math.nan}

        signal = self.signal_ema.value
        histogram = self._macd_value - signal if not math.isnan(self._macd_value) and not math.isnan(signal) else math.nan

        return {'macd': self._macd_value, 'signal': signal, 'histogram': histogram}

    @property
    def is_ready(self) -> bool:
        """
        Check if the indicator is ready.

        Returns:
            bool: True if the signal EMA is ready, False otherwise.
        """
        return self.signal_ema.is_ready

class Stochastic(BaseIndicator):
    """
    Stochastic Oscillator (Fast and Slow variants).

    A momentum indicator comparing a particular closing price of a security
    to a range of its prices over a certain period of time.

    Mathematical Formula:
        Raw %K = 100 * (Current Close - Lowest Low) / (Highest High - Lowest Low)
        Slow %K = SMA(Raw %K, smooth_k) [if smooth_k > 1]
        Fast %K = Raw %K [if smooth_k = 1]
        %D = SMA(%K, d_period)

    By default, calculates Slow Stochastic (smooth_k=3) to match pandas-ta behavior.
    Set smooth_k=1 for Fast Stochastic.

    Attributes:
        low_min_max (RollingMinMax): The rolling min/max for low prices.
        high_min_max (RollingMinMax): The rolling min/max for high prices.
        k_sma (SMA): The SMA for smoothing %K (if smooth_k > 1).
        d_sma (SMA): The SMA for the %D line.
    """

    required_inputs = ('close', 'low', 'high')

    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3, input_field: str = 'close'):
        """
        Initialize Stochastic Oscillator.

        Args:
            k_period (int): The lookback period for %K.
            d_period (int): The smoothing period for %D.
            smooth_k (int): The smoothing period for %K (default 3 for Slow Stochastic).
            input_field (str): The input field to use for the calculation.
        """
        # Validate parameters
        if k_period <= 0:
            raise InvalidParameterError("k_period", k_period, "positive integer (> 0)")
        if d_period < 0:
            raise InvalidParameterError("d_period", d_period, "non-negative integer (>= 0)")
        if smooth_k < 1:
            raise InvalidParameterError("smooth_k", smooth_k, "positive integer (>= 1)")
        
        super().__init__(period=k_period, input_field=input_field)
        self.required_inputs = (input_field, 'low', 'high')

        self._k_period = k_period
        self._d_period = d_period
        self._smooth_k = smooth_k

        self.low_min_max = RollingMinMax(k_period)
        self.high_min_max = RollingMinMax(k_period)
        
        # Add smoothing SMA for %K (for Slow Stochastic)
        self.k_sma = None
        if smooth_k > 1:
            self.k_sma = SMA(smooth_k, input_field='raw_k')  # placeholder
        
        self.d_sma = None
        if d_period > 0:
            self.d_sma = SMA(d_period, input_field='k') # placeholder

        self._children: List[BaseIndicator] = []
        if self.k_sma:
            self._children.append(self.k_sma)
        if self.d_sma:
            self._children.append(self.d_sma)

        self._raw_k_value = math.nan
        self._k_value = math.nan

    def update(self, data_point: Dict[str, Any]) -> None:
        """
        Update the Stochastic Oscillator with a new data point.

        Args:
            data_point (Dict[str, Any]): The new data point.
        """
        self._validate_input_data(data_point)
        close = data_point[self.input_field]
        low = data_point['low']
        high = data_point['high']

        self.low_min_max.update(low)
        self.high_min_max.update(high)

        if self.low_min_max.is_ready and self.high_min_max.is_ready:
            lowest_low = self.low_min_max.min
            highest_high = self.high_min_max.max

            # Calculate raw %K
            if highest_high == lowest_low:
                self._raw_k_value = 50.0
            else:
                self._raw_k_value = 100 * (close - lowest_low) / (highest_high - lowest_low)
            
            # Apply smoothing to %K if smooth_k > 1 (Slow Stochastic)
            if self.k_sma:
                self.k_sma.update({'raw_k': self._raw_k_value})
                if self.k_sma.is_ready:
                    self._k_value = self.k_sma.value
                else:
                    self._k_value = math.nan
            else:
                # Fast Stochastic - use raw %K
                self._k_value = self._raw_k_value
            
            # Update %D with smoothed/raw %K
            if self.d_sma and not math.isnan(self._k_value):
                self.d_sma.update({'k': self._k_value})

        self._update_metadata(data_point)
        self._store_output(self.value)

    @property
    def value(self) -> Dict[str, float]:
        """
        Get the current Stochastic Oscillator values.

        Returns:
            Dict[str, float]: A dictionary with 'k' and 'd'.
        """
        # %K is ready when RollingMinMax indicators are ready
        # For Slow Stochastic, also need k_sma to be ready
        k_ready = self.low_min_max.is_ready and self.high_min_max.is_ready
        if self.k_sma:
            k_ready = k_ready and self.k_sma.is_ready
        
        if not k_ready:
            return {'k': math.nan, 'd': math.nan}

        # %D is ready when SMA is ready (if d_period > 0)
        d_value = math.nan
        if self.d_sma and self.d_sma.is_ready:
            d_value = self.d_sma.value
        
        return {'k': self._k_value, 'd': d_value}

    @property
    def is_ready(self) -> bool:
        """
        Check if the indicator is ready.

        Returns:
            bool: True if the %K is ready (including smoothing if enabled).
        """
        k_ready = self.low_min_max.is_ready and self.high_min_max.is_ready
        if self.k_sma:
            k_ready = k_ready and self.k_sma.is_ready
        return k_ready
