# meluna/analysis/__init__.py

"""
Analysis module for the Meluna backtesting framework.

This module provides comprehensive analytics and metrics calculation tools
for backtest analysis, including dashboard components and metrics engines.

Structure:
- metrics/: Core calculation logic and analytics engines
- dashboard/: UI components and dashboard functionality  
- trade_visualization/: Trade visualization tools and charts
"""

# Import metrics components
from .metrics import (
    BacktestMetrics,
    PortfolioMetrics,
    TradeAnalyzer,
    TradeMetrics,
    MetricsEngine,
    DataValidator,
    CacheManager,
    DataLoader,
    DataValidationError,
    FileNotFoundError,
    DataSchema
)

# Import dashboard components
from .dashboard import (
    DashboardDataService,
    LoadingState,
    DataValidationResult,
    StrategyVersionInfo,
    CachedAnalyticsData,
    EnhancedTradingDashboard
)

# Import trade visualization components
from . import trade_visualization

__all__ = [
    # Metrics
    'BacktestMetrics',
    'PortfolioMetrics',
    'TradeAnalyzer',
    'TradeMetrics',
    'MetricsEngine',
    'DataValidator',
    'CacheManager', 
    'DataLoader',
    'DataValidationError',
    'FileNotFoundError',
    'DataSchema',
    # Dashboard
    'DashboardDataService',
    'LoadingState',
    'DataValidationResult',
    'StrategyVersionInfo',
    'CachedAnalyticsData',
    'EnhancedTradingDashboard',
    # Trade visualization
    'trade_visualization'
]