# meluna/analysis/KPICards.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go

from ..metrics.PortfolioMetrics import PortfolioMetrics
from ..metrics.TradeAnalyzer import TradeAnalyzer, TradeMetrics
from .DashboardDataService import CachedAnalyticsData

logger = logging.getLogger(__name__)

# Professional Financial Color Palette
COLORS = {
    'primary': '#0D1B2A',      # Deep Navy Blue - Headers, primary text
    'secondary': '#1B2951',    # Slate Blue - Secondary elements
    'tertiary': '#415A77',     # Steel Gray - Tertiary elements
    'profit': '#00C851',       # Profit Green - Positive performance
    'loss': '#FF4444',         # Loss Red - Negative performance
    'caution': '#FF8800',      # Caution Orange - Warnings, neutral
    'info': '#007BFF',         # Insight Blue - Information, benchmarks
    'background': '#FFFFFF',   # Pure White - Card backgrounds
    'page_bg': '#F8F9FA',      # Light Gray - Page background
    'border': '#E9ECEF'        # Warm Gray - Dividers, borders
}

# Performance thresholds for color coding
PERFORMANCE_THRESHOLDS = {
    'sharpe_ratio': {'excellent': 2.0, 'good': 1.0, 'poor': 0.5},
    'sortino_ratio': {'excellent': 2.5, 'good': 1.5, 'poor': 0.8},
    'cagr': {'excellent': 0.15, 'good': 0.08, 'poor': 0.03},
    'max_drawdown': {'excellent': -0.05, 'good': -0.10, 'poor': -0.20},
    'volatility': {'excellent': 0.10, 'good': 0.15, 'poor': 0.25},
    'profit_factor': {'excellent': 2.0, 'good': 1.5, 'poor': 1.1},
    'win_rate': {'excellent': 0.65, 'good': 0.55, 'poor': 0.45},
    'mfe_efficiency': {'excellent': 0.8, 'good': 0.6, 'poor': 0.4}
}

@dataclass
class KPICardData:
    """Container for KPI card display data."""
    title: str
    value: str
    subtitle: str
    color: str
    icon: str
    trend: Optional[str]
    tooltip: str
    detailed_info: Dict[str, Any]
    benchmark_comparison: Optional[str] = None
    percentile_rank: Optional[float] = None
    methodology: Optional[str] = None


class NumberFormatter:
    """Utility class for professional number formatting."""
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """Format a decimal as percentage."""
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        return f"{value * 100:.{decimal_places}f}%"
    
    @staticmethod
    def format_currency(value: float, currency: str = "₹", decimal_places: int = 2) -> str:
        """Format a number as currency."""
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        
        abs_value = abs(value)
        if abs_value >= 1e9:
            return f"{currency}{value/1e9:.1f}B"
        elif abs_value >= 1e6:
            return f"{currency}{value/1e6:.1f}M"
        elif abs_value >= 1e3:
            return f"{currency}{value/1e3:.1f}K"
        else:
            return f"{currency}{value:.{decimal_places}f}"
    
    @staticmethod
    def format_ratio(value: float, decimal_places: int = 2) -> str:
        """Format a ratio with appropriate decimal places."""
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        return f"{value:.{decimal_places}f}"
    
    @staticmethod
    def format_days(value: float) -> str:
        """Format number of days."""
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        
        days = int(value)
        if days >= 365:
            years = days / 365.25
            return f"{years:.1f}y"
        elif days >= 30:
            months = days / 30.44
            return f"{months:.1f}m"
        else:
            return f"{days}d"
    
    @staticmethod
    def format_large_number(value: Union[int, float], decimal_places: int = 0) -> str:
        """Format large numbers with appropriate suffixes."""
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        
        abs_value = abs(value)
        if abs_value >= 1e9:
            return f"{value/1e9:.{decimal_places}f}B"
        elif abs_value >= 1e6:
            return f"{value/1e6:.{decimal_places}f}M"
        elif abs_value >= 1e3:
            return f"{value/1e3:.{decimal_places}f}K"
        else:
            return f"{int(value) if decimal_places == 0 else round(value, decimal_places)}"


class PerformanceColorizer:
    """Utility class for determining performance-based colors."""
    
    @staticmethod
    def get_color_for_metric(metric_name: str, value: float) -> str:
        """Get color based on metric value and performance thresholds."""
        if pd.isna(value) or np.isinf(value):
            return COLORS['tertiary']
        
        thresholds = PERFORMANCE_THRESHOLDS.get(metric_name, {})
        if not thresholds:
            return COLORS['primary']
        
        excellent = thresholds.get('excellent')
        good = thresholds.get('good')
        poor = thresholds.get('poor')
        
        # For metrics where higher is better
        if metric_name in ['sharpe_ratio', 'sortino_ratio', 'cagr', 'profit_factor', 'win_rate', 'mfe_efficiency']:
            if excellent and value >= excellent:
                return COLORS['profit']
            elif good and value >= good:
                return COLORS['info']
            elif poor and value >= poor:
                return COLORS['caution']
            else:
                return COLORS['loss']
        
        # For metrics where lower is better (drawdown, volatility)
        elif metric_name in ['max_drawdown', 'volatility']:
            if excellent and value >= excellent:  # Less negative for drawdown
                return COLORS['profit']
            elif good and value >= good:
                return COLORS['info']
            elif poor and value >= poor:
                return COLORS['caution']
            else:
                return COLORS['loss']
        
        return COLORS['primary']
    
    @staticmethod
    def get_trend_direction(current: float, benchmark: Optional[float] = None) -> Optional[str]:
        """Determine trend direction based on performance."""
        if benchmark is None or pd.isna(current) or pd.isna(benchmark):
            return None
        
        diff = current - benchmark
        if abs(diff) < 0.001:  # Essentially equal
            return 'neutral'
        elif diff > 0:
            return 'up'
        else:
            return 'down'


class BenchmarkProvider:
    """Provides benchmark data and comparisons for metrics."""
    
    # Industry benchmark data (can be updated with real market data)
    MARKET_BENCHMARKS = {
        'sharpe_ratio': {'sp500': 0.7, 'market_avg': 0.5},
        'sortino_ratio': {'sp500': 1.0, 'market_avg': 0.8},
        'cagr': {'sp500': 0.10, 'market_avg': 0.08},
        'max_drawdown': {'sp500': -0.20, 'market_avg': -0.15},
        'volatility': {'sp500': 0.16, 'market_avg': 0.18},
        'profit_factor': {'market_avg': 1.3},
        'win_rate': {'market_avg': 0.5},
    }
    
    @staticmethod
    def get_benchmark_comparison(metric_name: str, value: float, 
                               benchmark_type: str = 'market_avg') -> Optional[str]:
        """Get benchmark comparison text."""
        benchmarks = BenchmarkProvider.MARKET_BENCHMARKS.get(metric_name, {})
        benchmark_value = benchmarks.get(benchmark_type)
        
        if benchmark_value is None or pd.isna(value):
            return None
        
        diff = value - benchmark_value
        
        if metric_name in ['sharpe_ratio', 'sortino_ratio', 'cagr', 'profit_factor', 'win_rate']:
            # Higher is better
            if diff > 0:
                return f"+{NumberFormatter.format_percentage(abs(diff))} vs {benchmark_type.replace('_', ' ').title()}"
            else:
                return f"-{NumberFormatter.format_percentage(abs(diff))} vs {benchmark_type.replace('_', ' ').title()}"
        
        elif metric_name in ['max_drawdown', 'volatility']:
            # Lower is better (for drawdown, less negative is better)
            if metric_name == 'max_drawdown':
                if diff > 0:  # Less negative drawdown is better
                    return f"+{NumberFormatter.format_percentage(abs(diff))} better than {benchmark_type.replace('_', ' ').title()}"
                else:
                    return f"{NumberFormatter.format_percentage(abs(diff))} worse than {benchmark_type.replace('_', ' ').title()}"
            else:  # Volatility
                if diff < 0:  # Lower volatility is better
                    return f"-{NumberFormatter.format_percentage(abs(diff))} vs {benchmark_type.replace('_', ' ').title()}"
                else:
                    return f"+{NumberFormatter.format_percentage(abs(diff))} vs {benchmark_type.replace('_', ' ').title()}"
        
        return None


class KPICardGenerator:
    """Main class for generating KPI cards with real metrics data."""
    
    def __init__(self, analytics_data: CachedAnalyticsData):
        """
        Initialize KPI card generator with analytics data.
        
        Args:
            analytics_data: Cached analytics data from DashboardDataService
        """
        self.analytics_data = analytics_data
        self.portfolio_metrics = analytics_data.portfolio_metrics or {}
        self.trade_metrics = analytics_data.trade_metrics
        
        # Initialize utility classes
        self.formatter = NumberFormatter()
        self.colorizer = PerformanceColorizer()
        self.benchmark_provider = BenchmarkProvider()
        
        logger.info(f"KPICardGenerator initialized for {analytics_data.strategy}/{analytics_data.version}")
    
    def generate_performance_cards(self) -> List[KPICardData]:
        """Generate performance KPI cards (CAGR, Total Return, Sharpe Ratio, etc.)."""
        cards = []
        
        # CAGR Card
        cagr = self.portfolio_metrics.get('CAGR', 0)
        cagr_card = KPICardData(
            title="CAGR",
            value=self.formatter.format_percentage(cagr),
            subtitle=self.benchmark_provider.get_benchmark_comparison('cagr', cagr) or "Compound Annual Growth Rate",
            color=self.colorizer.get_color_for_metric('cagr', cagr),
            icon="fas fa-chart-line",
            trend=self.colorizer.get_trend_direction(cagr, 0.08),  # 8% market average
            tooltip="Compound Annual Growth Rate: The annualized rate of return calculated over the entire backtest period",
            detailed_info={
                'calculation': 'CAGR = (End Value / Start Value)^(1/Years) - 1',
                'period': f"{self.analytics_data.strategy}/{self.analytics_data.version}",
                'interpretation': 'Higher is better. Shows consistent growth over time.'
            },
            methodology="Calculated using geometric mean of returns over the entire backtest period"
        )
        cards.append(cagr_card)
        
        # Total Return Card
        total_return = self.portfolio_metrics.get('Total Return', 0)
        total_return_card = KPICardData(
            title="Total Return",
            value=self.formatter.format_percentage(total_return),
            subtitle="Cumulative Performance",
            color=self.colorizer.get_color_for_metric('cagr', total_return),  # Use CAGR thresholds
            icon="fas fa-trending-up",
            trend=self.colorizer.get_trend_direction(total_return, 0),
            tooltip="Total Return: Cumulative percentage return over the entire backtest period",
            detailed_info={
                'calculation': 'Total Return = (End Value - Start Value) / Start Value',
                'raw_return': total_return,
                'annualized': cagr
            },
            methodology="Simple cumulative return calculation from start to end of backtest"
        )
        cards.append(total_return_card)
        
        # Sharpe Ratio Card
        sharpe_ratio = self.portfolio_metrics.get('Sharpe Ratio', 0)
        sharpe_card = KPICardData(
            title="Sharpe Ratio",
            value=self.formatter.format_ratio(sharpe_ratio),
            subtitle=self.benchmark_provider.get_benchmark_comparison('sharpe_ratio', sharpe_ratio) or "Risk-Adjusted Returns",
            color=self.colorizer.get_color_for_metric('sharpe_ratio', sharpe_ratio),
            icon="fas fa-balance-scale",
            trend=self.colorizer.get_trend_direction(sharpe_ratio, 0.7),  # S&P 500 average
            tooltip="Sharpe Ratio: Measures risk-adjusted returns by comparing excess returns to volatility",
            detailed_info={
                'calculation': 'Sharpe = (Return - Risk Free Rate) / Volatility',
                'risk_free_rate': 0.0,  # Assuming 0% risk-free rate
                'volatility': self.portfolio_metrics.get('Annualized Volatility', 0),
                'interpretation': '>2.0 Excellent, >1.0 Good, >0.5 Fair, <0.5 Poor'
            },
            methodology="Calculated using annualized excess returns divided by annualized volatility"
        )
        cards.append(sharpe_card)
        
        # Sortino Ratio Card
        sortino_ratio = self.portfolio_metrics.get('Sortino Ratio', 0)
        sortino_card = KPICardData(
            title="Sortino Ratio",
            value=self.formatter.format_ratio(sortino_ratio),
            subtitle="Downside Risk-Adjusted Returns",
            color=self.colorizer.get_color_for_metric('sortino_ratio', sortino_ratio),
            icon="fas fa-shield-alt",
            trend=self.colorizer.get_trend_direction(sortino_ratio, 1.0),
            tooltip="Sortino Ratio: Similar to Sharpe but only penalizes downside volatility",
            detailed_info={
                'calculation': 'Sortino = (Return - Risk Free Rate) / Downside Deviation',
                'downside_deviation': self.portfolio_metrics.get('Downside Deviation', 0),
                'interpretation': 'Generally higher than Sharpe ratio as it only considers bad volatility'
            },
            methodology="Uses downside deviation instead of total volatility for risk adjustment"
        )
        cards.append(sortino_card)
        
        return cards
    
    def generate_risk_cards(self) -> List[KPICardData]:
        """Generate risk KPI cards (Max Drawdown, Volatility, VaR, etc.)."""
        cards = []
        
        # Maximum Drawdown Card
        max_drawdown = self.portfolio_metrics.get('Max Drawdown', 0)
        dd_duration = self.portfolio_metrics.get('Longest Drawdown Duration (days)', 0)
        max_dd_card = KPICardData(
            title="Max Drawdown",
            value=self.formatter.format_percentage(max_drawdown),
            subtitle=f"Recovery: {self.formatter.format_days(dd_duration)}",
            color=self.colorizer.get_color_for_metric('max_drawdown', max_drawdown),
            icon="fas fa-arrow-down",
            trend=self.colorizer.get_trend_direction(max_drawdown, -0.15),  # -15% market average
            tooltip="Maximum Drawdown: Largest peak-to-trough decline during the backtest period",
            detailed_info={
                'calculation': 'Max DD = Min((Portfolio Value - Running Max) / Running Max)',
                'duration_days': dd_duration,
                'interpretation': 'Lower absolute values are better. Shows worst-case scenario.',
                'num_drawdown_periods': self.portfolio_metrics.get('Number of Drawdown Periods', 0)
            },
            methodology="Calculated as maximum percentage decline from any peak to subsequent trough"
        )
        cards.append(max_dd_card)
        
        # Volatility Card
        volatility = self.portfolio_metrics.get('Annualized Volatility', 0)
        vol_card = KPICardData(
            title="Volatility",
            value=self.formatter.format_percentage(volatility),
            subtitle=self.benchmark_provider.get_benchmark_comparison('volatility', volatility) or "Annualized",
            color=self.colorizer.get_color_for_metric('volatility', volatility),
            icon="fas fa-wave-square",
            trend=self.colorizer.get_trend_direction(volatility, 0.16),  # S&P 500 average
            tooltip="Volatility: Standard deviation of returns, measuring price fluctuation intensity",
            detailed_info={
                'calculation': 'Vol = StdDev(Daily Returns) * sqrt(252)',
                'daily_volatility': volatility / np.sqrt(252) if volatility > 0 else 0,
                'interpretation': 'Lower is generally better, but some volatility indicates active strategy'
            },
            methodology="Annualized standard deviation of daily returns"
        )
        cards.append(vol_card)
        
        # VaR 95% Card
        var_95 = self.portfolio_metrics.get('VaR 95% (%)', 0) / 100  # Convert to decimal
        var_card = KPICardData(
            title="VaR (95%)",
            value=self.formatter.format_percentage(var_95),
            subtitle="Daily Value at Risk",
            color=COLORS['loss'] if var_95 < -0.02 else COLORS['caution'] if var_95 < -0.01 else COLORS['info'],
            icon="fas fa-exclamation-triangle",
            trend=None,
            tooltip="Value at Risk (95%): Expected maximum daily loss in 95% of trading days",
            detailed_info={
                'calculation': 'VaR = 5th percentile of daily returns',
                'var_99': self.portfolio_metrics.get('VaR 99% (%)', 0) / 100,
                'interpretation': '5% chance of losing more than this amount in a single day'
            },
            methodology="Historical simulation method using 5th percentile of return distribution"
        )
        cards.append(var_card)
        
        # CVaR 95% Card
        cvar_95 = self.portfolio_metrics.get('CVaR 95% (%)', 0) / 100  # Convert to decimal
        cvar_card = KPICardData(
            title="CVaR (95%)",
            value=self.formatter.format_percentage(cvar_95),
            subtitle="Conditional Value at Risk",
            color=COLORS['loss'] if cvar_95 < -0.03 else COLORS['caution'] if cvar_95 < -0.015 else COLORS['info'],
            icon="fas fa-shield-alt",
            trend=None,
            tooltip="Conditional VaR: Average loss when losses exceed VaR threshold",
            detailed_info={
                'calculation': 'CVaR = Average of losses beyond VaR threshold',
                'tail_expectation': 'Expected loss in worst 5% of outcomes',
                'interpretation': 'More conservative risk measure than VaR'
            },
            methodology="Expected Shortfall calculation - average of tail losses"
        )
        cards.append(cvar_card)
        
        return cards
    
    def generate_trade_efficiency_cards(self) -> List[KPICardData]:
        """Generate trade efficiency KPI cards (Profit Factor, Win Rate, MFE Efficiency, etc.)."""
        cards = []
        
        if not self.trade_metrics:
            # Return placeholder cards if no trade data available
            return self._generate_placeholder_trade_cards()
        
        # Profit Factor Card
        profit_factor = self.trade_metrics.profit_factor
        pf_card = KPICardData(
            title="Profit Factor",
            value=self.formatter.format_ratio(profit_factor),
            subtitle=self.benchmark_provider.get_benchmark_comparison('profit_factor', profit_factor) or "Gross Profit / Gross Loss",
            color=self.colorizer.get_color_for_metric('profit_factor', profit_factor),
            icon="fas fa-calculator",
            trend=self.colorizer.get_trend_direction(profit_factor, 1.3),
            tooltip="Profit Factor: Ratio of gross profits to gross losses",
            detailed_info={
                'calculation': 'Profit Factor = Sum(Winning Trades) / Sum(|Losing Trades|)',
                'gross_profits': getattr(self.trade_metrics, 'gross_profits', 0),
                'gross_losses': getattr(self.trade_metrics, 'gross_losses', 0),
                'interpretation': '>2.0 Excellent, >1.5 Good, >1.1 Marginal, <1.0 Losing'
            },
            methodology="Simple ratio calculation from trade-by-trade P&L data"
        )
        cards.append(pf_card)
        
        # Win Rate Card
        # Calculate win rate from trade analyzer if available
        if hasattr(self.analytics_data, 'trade_log') and self.analytics_data.trade_log is not None:
            winning_trades = (self.analytics_data.trade_log['pnl'] > 0).sum()
            total_trades = len(self.analytics_data.trade_log)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0.5  # Default fallback
        
        win_rate_card = KPICardData(
            title="Win Rate",
            value=self.formatter.format_percentage(win_rate),
            subtitle=f"{int(winning_trades) if 'winning_trades' in locals() else 'N/A'} of {total_trades if 'total_trades' in locals() else 'N/A'} trades",
            color=self.colorizer.get_color_for_metric('win_rate', win_rate),
            icon="fas fa-trophy",
            trend=self.colorizer.get_trend_direction(win_rate, 0.5),
            tooltip="Win Rate: Percentage of profitable trades",
            detailed_info={
                'calculation': 'Win Rate = Winning Trades / Total Trades',
                'winning_trades': winning_trades if 'winning_trades' in locals() else 0,
                'total_trades': total_trades if 'total_trades' in locals() else 0,
                'interpretation': '>65% Excellent, >55% Good, >45% Fair, <45% Poor'
            },
            methodology="Simple percentage of profitable trades"
        )
        cards.append(win_rate_card)
        
        # Average Win Card
        avg_win = self.trade_metrics.avg_win if hasattr(self.trade_metrics, 'avg_win') else 0
        avg_win_card = KPICardData(
            title="Avg Win",
            value=self.formatter.format_currency(avg_win),
            subtitle="Per winning trade",
            color=COLORS['profit'] if avg_win > 0 else COLORS['tertiary'],
            icon="fas fa-arrow-up",
            trend='up' if avg_win > 0 else None,
            tooltip="Average Win: Mean profit per winning trade",
            detailed_info={
                'calculation': 'Avg Win = Sum(Winning Trades) / Count(Winning Trades)',
                'interpretation': 'Higher values indicate larger individual wins'
            },
            methodology="Average P&L of all profitable trades"
        )
        cards.append(avg_win_card)
        
        # Average Loss Card
        avg_loss = getattr(self.trade_metrics, 'avg_loss', 0)
        avg_loss_card = KPICardData(
            title="Avg Loss",
            value=self.formatter.format_currency(avg_loss),
            subtitle="Per losing trade",
            color=COLORS['loss'] if avg_loss < 0 else COLORS['tertiary'],
            icon="fas fa-arrow-down",
            trend='down' if avg_loss < 0 else None,
            tooltip="Average Loss: Mean loss per losing trade",
            detailed_info={
                'calculation': 'Avg Loss = Sum(Losing Trades) / Count(Losing Trades)',
                'interpretation': 'Smaller absolute values indicate better risk control'
            },
            methodology="Average P&L of all losing trades"
        )
        cards.append(avg_loss_card)
        
        # MFE Efficiency Card
        mfe_efficiency = self.trade_metrics.mfe_efficiency
        mfe_card = KPICardData(
            title="MFE Efficiency",
            value=self.formatter.format_percentage(mfe_efficiency),
            subtitle="Favorable Excursion Capture",
            color=self.colorizer.get_color_for_metric('mfe_efficiency', mfe_efficiency),
            icon="fas fa-bullseye",
            trend=self.colorizer.get_trend_direction(mfe_efficiency, 0.6),
            tooltip="MFE Efficiency: How much of the maximum favorable excursion was captured",
            detailed_info={
                'calculation': 'MFE Efficiency = Average(Final P&L / MFE) for winning trades',
                'avg_mfe': self.trade_metrics.avg_mfe,
                'interpretation': '>80% Excellent, >60% Good, >40% Fair, <40% Poor'
            },
            methodology="Measures how well the strategy captured favorable price movements"
        )
        cards.append(mfe_card)
        
        # Expectancy Card
        expectancy = self.trade_metrics.expectancy
        expectancy_card = KPICardData(
            title="Expectancy",
            value=self.formatter.format_currency(expectancy),
            subtitle="Expected P&L per trade",
            color=COLORS['profit'] if expectancy > 0 else COLORS['loss'],
            icon="fas fa-chart-bar",
            trend='up' if expectancy > 0 else 'down',
            tooltip="Expectancy: Average expected profit/loss per trade",
            detailed_info={
                'calculation': 'Expectancy = Sum(All Trades P&L) / Total Trades',
                'interpretation': 'Positive values indicate profitable strategy over time'
            },
            methodology="Mathematical expectation of trade outcomes"
        )
        cards.append(expectancy_card)
        
        return cards
    
    def generate_statistical_cards(self) -> List[KPICardData]:
        """Generate statistical KPI cards (Skewness, Kurtosis, Statistical Significance, etc.)."""
        cards = []
        
        # Skewness Card
        skewness = self.portfolio_metrics.get('Skewness', 0)
        skew_interpretation = "Negative" if skewness < -0.5 else "Positive" if skewness > 0.5 else "Symmetric"
        skew_card = KPICardData(
            title="Skewness",
            value=self.formatter.format_ratio(skewness, 3),
            subtitle=f"{skew_interpretation} Distribution",
            color=COLORS['profit'] if skewness > 0 else COLORS['loss'] if skewness < -0.5 else COLORS['info'],
            icon="fas fa-chart-area",
            trend='up' if skewness > 0 else 'down' if skewness < -0.5 else 'neutral',
            tooltip="Skewness: Measures asymmetry of return distribution",
            detailed_info={
                'calculation': 'Skewness = E[(X-μ)³] / σ³',
                'interpretation': 'Positive: more upside potential, Negative: more downside risk',
                'distribution_shape': skew_interpretation
            },
            methodology="Third moment of return distribution normalized by standard deviation cubed"
        )
        cards.append(skew_card)
        
        # Kurtosis Card
        kurtosis = self.portfolio_metrics.get('Kurtosis', 0)
        kurt_interpretation = "High Tail Risk" if kurtosis > 3 else "Low Tail Risk" if kurtosis < 0 else "Normal"
        kurt_card = KPICardData(
            title="Kurtosis",
            value=self.formatter.format_ratio(kurtosis, 2),
            subtitle=kurt_interpretation,
            color=COLORS['loss'] if kurtosis > 3 else COLORS['profit'] if kurtosis < 0 else COLORS['info'],
            icon="fas fa-exclamation-circle",
            trend='down' if kurtosis > 3 else 'up' if kurtosis < 0 else 'neutral',
            tooltip="Kurtosis: Measures tail risk and extreme events in return distribution",
            detailed_info={
                'calculation': 'Excess Kurtosis = E[(X-μ)⁴] / σ⁴ - 3',
                'interpretation': '>3: Fat tails, <0: Thin tails, ~0: Normal distribution',
                'tail_risk': kurt_interpretation
            },
            methodology="Fourth moment excess kurtosis measuring tail thickness"
        )
        cards.append(kurt_card)
        
        # Statistical Significance Card
        if self.trade_metrics and hasattr(self.trade_metrics, 'expectancy_pvalue'):
            p_value = self.trade_metrics.expectancy_pvalue
            is_significant = p_value < 0.05 if not pd.isna(p_value) else False
            
            sig_card = KPICardData(
                title="Statistical Significance",
                value="Yes" if is_significant else "No",
                subtitle=f"p-value: {p_value:.3f}" if not pd.isna(p_value) else "Insufficient data",
                color=COLORS['profit'] if is_significant else COLORS['caution'],
                icon="fas fa-check-circle" if is_significant else "fas fa-question-circle",
                trend='up' if is_significant else None,
                tooltip="Statistical Significance: Whether results are statistically meaningful",
                detailed_info={
                    'p_value': p_value,
                    'confidence_level': '95%',
                    'interpretation': 'p < 0.05 indicates statistically significant results',
                    'sample_size': len(self.analytics_data.trade_log) if self.analytics_data.trade_log is not None else 0
                },
                methodology="Two-tailed t-test against zero expectancy"
            )
        else:
            sig_card = KPICardData(
                title="Statistical Significance",
                value="N/A",
                subtitle="No trade data available",
                color=COLORS['tertiary'],
                icon="fas fa-question-circle",
                trend=None,
                tooltip="Statistical significance cannot be calculated without trade data",
                detailed_info={},
                methodology="Requires individual trade data for calculation"
            )
        
        cards.append(sig_card)
        
        # Gain-to-Pain Ratio Card
        gain_to_pain = self.portfolio_metrics.get('Gain-to-Pain Ratio', 0)
        gtp_card = KPICardData(
            title="Gain-to-Pain",
            value=self.formatter.format_ratio(gain_to_pain),
            subtitle="Reward vs Risk Ratio",
            color=COLORS['profit'] if gain_to_pain > 2 else COLORS['info'] if gain_to_pain > 1 else COLORS['caution'],
            icon="fas fa-balance-scale-left",
            trend='up' if gain_to_pain > 1 else 'down',
            tooltip="Gain-to-Pain: Ratio of positive returns to negative returns",
            detailed_info={
                'calculation': 'Gain-to-Pain = Sum(Positive Returns) / Sum(|Negative Returns|)',
                'interpretation': '>2: Excellent, >1: Good, <1: Poor',
                'positive_sum': self.portfolio_metrics.get('positive_returns_sum', 0),
                'negative_sum': self.portfolio_metrics.get('negative_returns_sum', 0)
            },
            methodology="Compares cumulative gains to cumulative losses"
        )
        cards.append(gtp_card)
        
        return cards
    
    def _generate_placeholder_trade_cards(self) -> List[KPICardData]:
        """Generate placeholder trade cards when no trade data is available."""
        placeholder_cards = [
            KPICardData(
                title="Profit Factor",
                value="N/A",
                subtitle="No trade data available",
                color=COLORS['tertiary'],
                icon="fas fa-calculator",
                trend=None,
                tooltip="Trade data required to calculate profit factor",
                detailed_info={},
                methodology="Requires individual trade records for calculation"
            ),
            KPICardData(
                title="Win Rate",
                value="N/A",
                subtitle="No trade data available",
                color=COLORS['tertiary'],
                icon="fas fa-trophy",
                trend=None,
                tooltip="Trade data required to calculate win rate",
                detailed_info={},
                methodology="Requires individual trade records for calculation"
            )
        ]
        return placeholder_cards
    
    def generate_all_kpi_cards(self) -> Dict[str, List[KPICardData]]:
        """Generate all KPI cards organized by category."""
        return {
            'performance': self.generate_performance_cards(),
            'risk': self.generate_risk_cards(),
            'trade_efficiency': self.generate_trade_efficiency_cards(),
            'statistical': self.generate_statistical_cards()
        }


class KPICardRenderer:
    """Renders KPI cards as Dash components."""
    
    @staticmethod
    def render_kpi_card(card_data: KPICardData, size: str = 'small') -> html.Div:
        """
        Render a single KPI card as a Dash component.
        
        Args:
            card_data: KPI card data
            size: Card size ('small', 'medium', 'large')
            
        Returns:
            Dash HTML Div containing the rendered card
        """
        # Card size configurations
        size_configs = {
            'small': {'gridColumn': 'span 3', 'gridRow': 'span 1', 'minHeight': '140px'},
            'medium': {'gridColumn': 'span 6', 'gridRow': 'span 1', 'minHeight': '140px'},
            'large': {'gridColumn': 'span 6', 'gridRow': 'span 2', 'minHeight': '300px'}
        }
        
        size_config = size_configs.get(size, size_configs['small'])
        
        # Trend icon
        trend_icon = None
        if card_data.trend == 'up':
            trend_icon = html.I(className="fas fa-arrow-up", style={
                'color': COLORS['profit'], 'fontSize': '14px', 'marginLeft': '8px'
            })
        elif card_data.trend == 'down':
            trend_icon = html.I(className="fas fa-arrow-down", style={
                'color': COLORS['loss'], 'fontSize': '14px', 'marginLeft': '8px'
            })
        elif card_data.trend == 'neutral':
            trend_icon = html.I(className="fas fa-minus", style={
                'color': COLORS['caution'], 'fontSize': '14px', 'marginLeft': '8px'
            })
        
        # Create tooltip content
        tooltip_content = card_data.tooltip
        if card_data.methodology:
            tooltip_content += f"\n\nMethodology: {card_data.methodology}"
        
        card_content = html.Div([
            # Header with icon and title
            html.Div([
                html.I(className=card_data.icon, style={
                    'fontSize': '16px',
                    'color': COLORS['tertiary'],
                    'marginRight': '8px'
                }) if card_data.icon else None,
                html.H4(card_data.title, style={
                    'margin': '0',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'color': COLORS['tertiary'],
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.5px',
                    'lineHeight': '1.2'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '12px'
            }),
            
            # Main value with trend
            html.Div([
                html.H2(card_data.value, style={
                    'margin': '0',
                    'fontSize': '28px' if size == 'small' else '32px',
                    'fontWeight': '700',
                    'color': card_data.color,
                    'fontFamily': 'JetBrains Mono, monospace',
                    'lineHeight': '1'
                }),
                trend_icon
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'marginBottom': '8px'
            }),
            
            # Subtitle
            html.P(card_data.subtitle, style={
                'margin': '0',
                'fontSize': '12px',
                'color': COLORS['tertiary'],
                'textAlign': 'center',
                'lineHeight': '1.3'
            }) if card_data.subtitle else None
            
        ], style={
            'textAlign': 'center',
            'height': '100%',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center'
        }, title=tooltip_content)  # Add tooltip
        
        return html.Div(
            card_content,
            className='dashboard-card kpi-card',
            style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px',
                'padding': '20px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                'transition': 'all 0.3s ease',
                **size_config
            }
        )
    
    @staticmethod
    def render_kpi_card_group(title: str, cards: List[KPICardData], 
                             card_size: str = 'small') -> html.Div:
        """
        Render a group of KPI cards with a title.
        
        Args:
            title: Group title
            cards: List of KPI card data
            card_size: Size for all cards in the group
            
        Returns:
            Dash HTML Div containing the card group
        """
        rendered_cards = [
            KPICardRenderer.render_kpi_card(card, card_size) 
            for card in cards
        ]
        
        return html.Div([
            # Group header
            html.Div([
                html.H3(title, style={
                    'margin': '0',
                    'fontSize': '20px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'flex': '1'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '20px',
                'paddingBottom': '10px',
                'borderBottom': f'2px solid {COLORS["border"]}'
            }),
            
            # Card grid
            html.Div(
                rendered_cards,
                style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(12, 1fr)',
                    'gap': '20px',
                    'gridAutoRows': 'minmax(140px, auto)'
                },
                className='card-grid-container'
            )
        ], style={
            'marginBottom': '40px'
        }, className='card-group')