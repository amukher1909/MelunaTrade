# meluna/analysis/trade_visualization/calculations/__init__.py

"""
Trade Visualization Calculation Modules

This package contains calculation modules for individual trade analysis:

- TradeMetrics: Trade-specific calculations (P&L curve, MFE/MAE, time analysis, R/R ratios)
- PriceActionMetrics: Price action analysis (volatility, volume, swing detection, trend analysis)

These modules provide the foundational calculations required for comprehensive
individual trade visualization and analysis.
"""

from .TradeMetrics import (
    TradeVisualizationMetrics,
    TradeMetricsCalculator,
    calculate_trade_metrics
)

from .PriceActionMetrics import (
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