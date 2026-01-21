"""Base class for technical indicators."""

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Any, List, Optional, Union, Tuple
import math
import logging
from datetime import datetime

from .exceptions import (
    InvalidParameterError, 
    MissingInputError, 
    InsufficientDataError, 
    InvalidDataError
)

logger = logging.getLogger(__name__)


class BaseIndicator(ABC):
    """Abstract base for streaming technical indicators."""
    
    # Class attribute to be overridden by subclasses
    required_inputs: Tuple[str, ...] = ()
    
    def __init__(self, period: int, input_field: str = 'close', **kwargs):
        """Initialize indicator with period and input field."""
        self._validate_period(period)
        
        self.period = period
        self.input_field = input_field
        
        # Core data structures for O(1) operations
        self._buffer: deque = deque(maxlen=period)
        self._output_history: deque = deque(maxlen=1000)  # Store recent outputs
        
        # State management
        self._ready_threshold = period
        self._data_count = 0
        
        # Composite pattern support
        self._children: List['BaseIndicator'] = []
        
        # Metadata
        self._name = self.__class__.__name__
        self._last_update_time: Optional[datetime] = None
        
        logger.debug(f"Initialized {self._name} with period={period}, input_field={input_field}")
    
    @abstractmethod
    def update(self, data_point: Dict[str, Any]) -> None:
        """
        Process a new data point and update the indicator state.
        
        This method must be implemented by all concrete indicator classes.
        It should:
        1. Validate the input data point
        2. Update internal state incrementally (O(1) complexity)
        3. Store the data point in the rolling window buffer
        4. Calculate and store the new indicator value
        
        Args:
            data_point (Dict[str, Any]): Market data containing OHLCV fields.
                Must include all fields specified in required_inputs.
                Common fields: 'open', 'high', 'low', 'close', 'volume', 'timestamp'
                
        Raises:
            MissingInputError: If required input fields are missing.
            InvalidDataError: If input data contains invalid values.
        """
        pass
    
    @property
    @abstractmethod
    def value(self) -> Union[float, Dict[str, float]]:
        """
        Get the current calculated value of the indicator.
        
        This property must be implemented by all concrete indicator classes.
        
        Returns:
            Union[float, Dict[str, float]]: The current indicator value.
                - Simple indicators return float (may be NaN if not ready)
                - Composite indicators return dict with named values
                - Returns NaN/empty dict during warm-up period
        
        Note:
            This property should return math.nan for simple indicators
            or an appropriate empty/NaN structure for composite indicators
            when is_ready is False.
        """
        pass
    
    @property
    def is_ready(self) -> bool:
        """
        Check if the indicator has sufficient data for valid calculations.
        
        Returns:
            bool: True if the indicator can produce valid output, False during warm-up.
        """
        return self._data_count >= self._ready_threshold
    
    @property
    def children(self) -> List['BaseIndicator']:
        """
        Get the list of child indicators for composite patterns.
        
        Returns:
            List[BaseIndicator]: List of child indicator instances.
                Empty list for simple (leaf) indicators.
        """
        return self._children.copy()  # Return copy to prevent external modification
    
    def get_history(self, n: int = 10) -> List[Union[float, Dict[str, float]]]:
        """
        Retrieve the last n calculated values from the indicator's history.
        
        Args:
            n (int): Number of recent values to return. Defaults to 10.
            
        Returns:
            List[Union[float, Dict[str, float]]]: List of recent indicator values,
                ordered from oldest to newest. May contain NaN values for
                periods when the indicator was not ready.
        """
        if n <= 0:
            return []
        
        history_length = len(self._output_history)
        start_idx = max(0, history_length - n)
        
        return list(self._output_history)[start_idx:]
    
    def reset(self) -> None:
        """
        Reset the indicator to its initial state.
        
        This method clears all internal buffers and state, returning the
        indicator to its post-construction state. Useful for backtesting
        multiple scenarios or reinitializing indicators.
        """
        self._buffer.clear()
        self._output_history.clear()
        self._data_count = 0
        self._last_update_time = None
        
        # Reset child indicators
        for child in self._children:
            child.reset()
        
        logger.debug(f"Reset {self._name} indicator state")
    
    def _validate_input_data(self, data_point: Dict[str, Any]) -> None:
        """
        Validate that the input data point contains required fields and valid values.
        
        Args:
            data_point (Dict[str, Any]): Market data to validate.
            
        Raises:
            MissingInputError: If required input fields are missing.
            InvalidDataError: If input data contains invalid values.
        """
        # Check for missing required fields
        missing_fields = [field for field in self.required_inputs if field not in data_point]
        if missing_fields:
            raise MissingInputError(missing_fields, list(self.required_inputs), self._name)
        
        # Validate numeric fields are not None/NaN and are finite
        for field in self.required_inputs:
            value = data_point[field]
            if value is None:
                raise InvalidDataError(field, value, "value is None", self._name)
            
            # Check if numeric field is finite
            if isinstance(value, (int, float)):
                if math.isnan(value):
                    raise InvalidDataError(field, value, "value is NaN", self._name)
                if math.isinf(value):
                    raise InvalidDataError(field, value, "value is infinite", self._name)
    
    def _update_metadata(self, data_point: Dict[str, Any]) -> None:
        """
        Update indicator metadata after processing a data point.
        
        Args:
            data_point (Dict[str, Any]): The processed data point.
        """
        self._data_count += 1
        
        # Update timestamp if available
        if 'timestamp' in data_point:
            self._last_update_time = data_point['timestamp']
    
    def _store_output(self, output_value: Union[float, Dict[str, float]]) -> None:
        """
        Store the calculated output value in the history buffer.
        
        Args:
            output_value: The calculated indicator value to store.
        """
        self._output_history.append(output_value)
    
    @staticmethod
    def _validate_period(period: int) -> None:
        """
        Validate that the period parameter is a positive integer.
        
        Args:
            period (int): Period value to validate.
            
        Raises:
            InvalidParameterError: If period is not a positive integer.
        """
        if not isinstance(period, int):
            raise InvalidParameterError("period", period, "positive integer")
        
        if period <= 0:
            raise InvalidParameterError("period", period, "positive integer (> 0)")
    
    def __repr__(self) -> str:
        """String representation of the indicator."""
        ready_status = "ready" if self.is_ready else f"warming up ({self._data_count}/{self._ready_threshold})"
        return f"{self._name}(period={self.period}, {ready_status})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()