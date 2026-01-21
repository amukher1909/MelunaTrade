"""
Meluna Technical Analysis Library

A high-performance, streaming technical indicator library designed for 
event-driven backtesting and real-time trading applications.

This library provides:
- Factory pattern for intuitive indicator creation
- O(1) streaming indicator updates using efficient data structures
- Abstract base class ensuring consistent indicator interfaces  
- Composite pattern support for complex hierarchical indicators
- Thread-safe implementations for multi-symbol processing
- Integration with Meluna's event-driven architecture

Example Usage:
    import meluna.technical_analysis as ta
    
    # Factory pattern - primary interface
    sma = ta.create('sma', period=20)
    macd = ta.create('macd', fast_period=12, slow_period=26, signal_period=9)
    
    # Direct class access
    sma = ta.SMA(period=20)
    bbands = ta.BollingerBands(period=20, k=2.0)
    
    # Utility functions
    indicators = ta.list_indicators()
    info = ta.describe('rsi')
"""

__version__ = "1.0.0"
__author__ = "Meluna Development Team"

# Public API exports
from .base import BaseIndicator
from .exceptions import (
    IndicatorError,
    InvalidParameterError, 
    MissingInputError,
    InsufficientDataError,
    InvalidDataError,
    IndicatorNotFoundError
)
from .indicators import (
    SMA, EMA, RSI,
    SmoothingStrategy, WildersSmoothing, EmaSmoothing
)
from .indicators.composite import MACD, BollingerBands, Stochastic
from .factory import (
    create, 
    list_indicators, 
    describe,
    validate_period,
    validate_alpha,
    validate_k_factor,
    validate_input_field
)

__all__ = [
    # Core classes
    "BaseIndicator",
    
    # Factory functions
    "create",
    "list_indicators", 
    "describe",
    
    # Basic indicators
    "SMA",
    "EMA",
    "RSI",
    
    # Composite indicators
    "MACD",
    "BollingerBands", 
    "Stochastic",
    
    # Smoothing strategies
    "SmoothingStrategy",
    "WildersSmoothing",
    "EmaSmoothing",
    
    # Validation utilities
    "validate_period",
    "validate_alpha",
    "validate_k_factor",
    "validate_input_field",
    
    # Exceptions
    "IndicatorError",
    "InvalidParameterError",
    "MissingInputError", 
    "InsufficientDataError",
    "InvalidDataError",
    "IndicatorNotFoundError",
    
    # Metadata
    "__version__",
    "__author__",
]