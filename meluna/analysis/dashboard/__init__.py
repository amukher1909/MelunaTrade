# meluna/analysis/dashboard/__init__.py

"""
Dashboard components for the Meluna trading analytics platform.

This module contains all UI components, tabs, and dashboard-related
functionality for visualizing backtest results and analytics.
"""

from .DashboardComponents import (
    create_header_bar,
    create_navigation_tabs,
    create_main_layout,
    create_tab_specific_content,
    get_available_strategies,
    get_available_versions,
    create_real_kpi_cards_from_data,
    create_loading_kpi_cards,
    COLORS,
    DASHBOARD_CSS
)
from .DashboardDataService import (
    DashboardDataService,
    LoadingState,
    DataValidationResult,
    StrategyVersionInfo,
    CachedAnalyticsData
)
from .EnhancedDashboards import EnhancedTradingDashboard
from .EnhancedDashboardWithKPI import EnhancedKPIDashboard
from .IndividualTradeTab import create_individual_trade_content
from .InteractiveComponents import create_css_enhancements
from .InteractiveDashboardManager import InteractiveDashboardManager, create_interactive_dashboard_manager
from .InteractiveEquityCurve import create_interactive_equity_curve_component
from .KPICards import KPICardGenerator, KPICardRenderer
from .PortfolioOverviewTab import create_portfolio_overview_content
from .RiskAnalysisTab import create_risk_analysis_content
from .TradeAnalyticsTab import create_trade_analytics_content
from .UnifiedPortfolioStatisticsTable import create_unified_portfolio_statistics_table

__all__ = [
    'create_header_bar',
    'create_navigation_tabs',
    'create_main_layout',
    'create_tab_specific_content',
    'get_available_strategies',
    'get_available_versions',
    'create_real_kpi_cards_from_data',
    'create_loading_kpi_cards',
    'COLORS',
    'DASHBOARD_CSS',
    'DashboardDataService',
    'LoadingState',
    'DataValidationResult',
    'StrategyVersionInfo',
    'CachedAnalyticsData',
    'EnhancedTradingDashboard',
    'EnhancedKPIDashboard',
    'create_individual_trade_content',
    'create_css_enhancements',
    'InteractiveDashboardManager',
    'create_interactive_dashboard_manager',
    'create_interactive_equity_curve_component',
    'KPICardGenerator',
    'KPICardRenderer',
    'create_portfolio_overview_content',
    'create_risk_analysis_content',
    'create_trade_analytics_content',
    'create_unified_portfolio_statistics_table'
]