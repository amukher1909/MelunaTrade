# meluna/analysis/metrics/__init__.py

"""
Metrics and calculation engines for the Meluna trading analytics platform.

This module contains core calculation logic, metrics engines, and 
analytical tools for processing backtest results.
"""

from .BacktestMetrics import BacktestMetrics
from .MetricsEngine import (
    MetricsEngine, 
    DataValidator, 
    CacheManager, 
    DataLoader,
    DataValidationError,
    FileNotFoundError,
    DataSchema
)
from .PortfolioMetrics import PortfolioMetrics
from .TradeAnalyzer import TradeAnalyzer, TradeMetrics

# Import example modules (these contain usage examples and demos)
from . import metrics_engine_examples
from . import portfolio_metrics_examples

__all__ = [
    'BacktestMetrics',
    'MetricsEngine',
    'DataValidator',
    'CacheManager', 
    'DataLoader',
    'DataValidationError',
    'FileNotFoundError',
    'DataSchema',
    'PortfolioMetrics',
    'TradeAnalyzer',
    'TradeMetrics',
    'metrics_engine_examples',
    'portfolio_metrics_examples'
]