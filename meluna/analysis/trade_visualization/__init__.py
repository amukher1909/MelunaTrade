# meluna/analysis/trade_visualization/__init__.py

"""
Individual Trade Visualization Module

This module provides comprehensive analysis and visualization capabilities for individual trades
in the Meluna backtesting framework. It includes calculation modules for trade metrics,
price action analysis, and base interfaces for visualization components.

Main Components:
- calculations/TradeMetrics.py: Trade-specific metrics (P&L, MFE/MAE, time analysis)
- calculations/PriceActionMetrics.py: Price action analysis (volatility, volume, swings)
- visualizations/: Base interfaces and components for trade visualization

Usage:
    from meluna.analysis.trade_visualization import calculate_trade_metrics, calculate_price_action_metrics
    from meluna.analysis.trade_visualization.calculations import TradeMetricsCalculator, PriceActionCalculator
"""

from .calculations.TradeMetrics import (
    TradeVisualizationMetrics,
    TradeMetricsCalculator,
    calculate_trade_metrics
)

from .calculations.PriceActionMetrics import (
    VolumeMetrics,
    VolatilityMetrics,
    SwingPoints,
    TrendMetrics,
    PriceActionAnalysis,
    PriceActionCalculator,
    calculate_price_action_metrics
)

__all__ = [
    # Trade Metrics
    'TradeVisualizationMetrics',
    'TradeMetricsCalculator',
    'calculate_trade_metrics',
    
    # Price Action Metrics
    'VolumeMetrics',
    'VolatilityMetrics',
    'SwingPoints',
    'TrendMetrics',
    'PriceActionAnalysis',
    'PriceActionCalculator',
    'calculate_price_action_metrics',
]

__version__ = "1.0.0"