"""Exception classes for technical analysis library."""

from typing import Any, List, Optional


class IndicatorError(Exception):
    """Base exception for indicator errors."""
    
    def __init__(self, message: str, indicator_name: Optional[str] = None):
        self.indicator_name = indicator_name
        if indicator_name:
            message = f"[{indicator_name}] {message}"
        super().__init__(message)


class InvalidParameterError(IndicatorError):
    """Invalid indicator parameters."""
    
    def __init__(self, parameter_name: str, value: Any, expected: str, indicator_name: Optional[str] = None):
        message = f"Invalid parameter '{parameter_name}': got {value}, expected {expected}"
        super().__init__(message, indicator_name)
        self.parameter_name = parameter_name
        self.value = value
        self.expected = expected


class MissingInputError(IndicatorError):
    """Missing required input fields."""
    
    def __init__(self, missing_fields: List[str], required_fields: List[str], indicator_name: Optional[str] = None):
        missing_str = ", ".join(missing_fields)
        required_str = ", ".join(required_fields)
        message = f"Missing required input fields: {missing_str}. Required: {required_str}"
        super().__init__(message, indicator_name)
        self.missing_fields = missing_fields
        self.required_fields = required_fields


class InsufficientDataError(IndicatorError):
    """Insufficient data for calculation."""
    
    def __init__(self, current_count: int, required_count: int, indicator_name: Optional[str] = None):
        message = f"Insufficient data for calculation: have {current_count} data points, need {required_count}"
        super().__init__(message, indicator_name)
        self.current_count = current_count
        self.required_count = required_count


class InvalidDataError(IndicatorError):
    """Malformed or invalid input data."""
    
    def __init__(self, field_name: str, value: Any, reason: str, indicator_name: Optional[str] = None):
        message = f"Invalid data in field '{field_name}': {value} ({reason})"
        super().__init__(message, indicator_name)
        self.field_name = field_name
        self.value = value
        self.reason = reason


class IndicatorNotFoundError(IndicatorError):
    """Unknown indicator requested."""
    
    def __init__(self, indicator_name: str, available_indicators: Optional[List[str]] = None):
        if available_indicators:
            available_str = ", ".join(sorted(available_indicators))
            message = f"Unknown indicator '{indicator_name}'. Available indicators: {available_str}"
        else:
            message = f"Unknown indicator '{indicator_name}'"
        super().__init__(message)
        self.indicator_name = indicator_name
        self.available_indicators = available_indicators or []