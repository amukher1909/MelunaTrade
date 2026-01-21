"""
Risk Analysis Tab - Comprehensive Risk Assessment and Exposure Analysis

This module implements the Risk Analysis tab featuring institutional-grade risk metrics,
tail risk analysis, and correlation studies for sophisticated risk management.

Features:
- VaR and CVaR analysis with multiple confidence levels (95%, 99%, 99.9%)
- Rolling volatility and correlation charts with benchmark comparisons
- Tail risk analysis with extreme return distribution visualization
- Underwater drawdown plots with recovery period analysis
- Monte Carlo simulation results and scenario analysis
- Volatility regime detection and classification analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from sklearn.mixture import GaussianMixture
import warnings

from dash import dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate

from .DashboardComponents import (
    create_kpi_card, create_chart_container, create_card_grid, 
    create_card_group, create_base_card, COLORS, CARD_SIZES
)
from .DashboardDataService import DashboardDataService, CachedAnalyticsData
from ..metrics.PortfolioMetrics import PortfolioMetrics

logger = logging.getLogger(__name__)


class RiskAnalysisTab:
    """
    Risk Analysis tab component providing comprehensive risk assessment capabilities.
    
    Integrates with PortfolioMetrics module and extends it with additional risk metrics
    including correlation analysis, Monte Carlo simulations, and regime detection.
    """
    
    def __init__(self, data_service: DashboardDataService):
        """
        Initialize Risk Analysis tab.
        
        Args:
            data_service: Dashboard data service for loading analytics data
        """
        self.data_service = data_service
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def create_tab_content(self, strategy: str = None, version: str = None, 
                          start_date: datetime = None, end_date: datetime = None) -> html.Div:
        """
        Create Risk Analysis tab content with real data integration.
        
        Args:
            strategy: Selected strategy name
            version: Selected version identifier
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            Dash HTML Div containing the complete Risk Analysis tab
        """
        
        # Load analytics data
        analytics_data = None
        if strategy and version:
            try:
                date_range = (start_date, end_date) if start_date and end_date else None
                analytics_data = self.data_service.load_analytics_data(
                    strategy, version, date_range
                )
            except Exception as e:
                logger.error(f"Error loading analytics data: {e}")
        
        # Create content based on data availability
        if analytics_data and analytics_data.portfolio_metrics:
            return self._create_content_with_data(analytics_data)
        else:
            return self._create_placeholder_content()
    
    def _create_content_with_data(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create tab content with real portfolio data."""
        
        # Calculate extended risk metrics
        extended_risk_metrics = self._calculate_extended_risk_metrics(analytics_data)
        
        # Create KPI cards with risk metrics
        kpi_cards = self._create_risk_kpi_cards(analytics_data, extended_risk_metrics)
        
        # Create risk analysis charts
        charts = self._create_risk_charts(analytics_data, extended_risk_metrics)
        
        return html.Div([
            # Tab Header
            html.Div([
                html.H2("Risk Analysis", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P("Comprehensive risk assessment and exposure analysis with institutional-grade metrics", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '20px'}),
            
            # Risk Metrics KPIs
            create_card_group("Value at Risk & Tail Risk Metrics", kpi_cards[:4]),
            
            # VaR/CVaR Analysis (Primary Feature)
            html.Div([
                charts['var_cvar_analysis']
            ], style={
                'marginBottom': '40px',  # Increased margin for better spacing
                'marginTop': '20px'      # Add top margin for breathing room
            }),
            
            # Rolling Volatility Analysis (Full-width for optimal space utilization)
            html.Div([
                charts['rolling_volatility']
            ], style={'marginBottom': '30px'}),
            
            # Correlation Analysis
            html.Div([
                charts['correlation_analysis']
            ], style={'marginBottom': '30px'}),
            
            # Tail Risk and Distribution Analysis
            html.Div([
                create_card_grid([
                    charts['tail_risk_analysis'],
                    charts['regime_detection']
                ])
            ], style={'marginBottom': '30px'}),
            
            # Monte Carlo and Scenario Analysis with improved spacing
            html.Div([
                create_card_grid([
                    charts['monte_carlo'],
                    charts['underwater_plot']
                ])
            ], style={
                'marginBottom': '40px',  # Increased from 30px to 40px for better separation
                'marginTop': '25px'      # Added top margin for breathing room
            }),
            
            # Additional Risk Metrics
            create_card_group("Advanced Risk Metrics", kpi_cards[4:])
            
        ], style={
            'padding': '20px',
            'backgroundColor': COLORS['page_bg'],
            'minHeight': 'calc(100vh - 90px)'
        })
    
    def _calculate_extended_risk_metrics(self, analytics_data: CachedAnalyticsData) -> Dict[str, Any]:
        """Calculate extended risk metrics beyond basic portfolio metrics."""
        
        equity_curve = analytics_data.equity_curve
        if equity_curve is None or equity_curve.empty:
            return {}
        
        returns = equity_curve.pct_change().dropna()
        extended_metrics = {}
        
        try:
            # Multiple VaR confidence levels
            confidence_levels = [0.90, 0.95, 0.99, 0.999]
            for conf in confidence_levels:
                alpha = 1 - conf
                var_value = np.percentile(returns, alpha * 100) * 100
                extended_metrics[f'VaR_{int(conf*100)}%'] = var_value
                
                # CVaR (Expected Shortfall)
                var_threshold = np.percentile(returns, alpha * 100)
                cvar_losses = returns[returns <= var_threshold]
                cvar_value = cvar_losses.mean() * 100 if len(cvar_losses) > 0 else var_value
                extended_metrics[f'CVaR_{int(conf*100)}%'] = cvar_value
            
            # Tail ratio (95th percentile / 5th percentile)
            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)
            extended_metrics['tail_ratio'] = abs(p95 / p5) if p5 != 0 else 0
            
            # Modified VaR (Cornish-Fisher expansion)
            if len(returns) > 30:
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns, fisher=True)
                z_score = stats.norm.ppf(0.05)  # 95% VaR
                
                # Cornish-Fisher adjustment
                cf_adjustment = (z_score + 
                               (z_score**2 - 1) * skewness / 6 +
                               (z_score**3 - 3*z_score) * kurtosis / 24 -
                               (2*z_score**3 - 5*z_score) * skewness**2 / 36)
                
                modified_var = returns.mean() + returns.std() * cf_adjustment
                extended_metrics['modified_var_95%'] = modified_var * 100
            
            # Volatility regime detection using Gaussian Mixture Model
            extended_metrics.update(self._detect_volatility_regimes(returns))
            
            # Rolling correlation with benchmark (simulated for demo)
            extended_metrics.update(self._calculate_rolling_correlations(returns))
            
            # Monte Carlo simulation results
            extended_metrics.update(self._run_monte_carlo_simulation(returns))
            
        except Exception as e:
            logger.error(f"Error calculating extended risk metrics: {e}")
        
        return extended_metrics
    
    def _detect_volatility_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect volatility regimes using Gaussian Mixture Model."""
        
        if len(returns) < 100:
            return {'regime_detection': 'insufficient_data'}
        
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) < 50:
                return {'regime_detection': 'insufficient_data'}
            
            # Fit Gaussian Mixture Model with 2 components (low/high volatility)
            gmm = GaussianMixture(n_components=2, random_state=42)
            vol_data = rolling_vol.values.reshape(-1, 1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm.fit(vol_data)
                regime_labels = gmm.predict(vol_data)
            
            # Identify low and high volatility regimes
            regime_means = [rolling_vol[regime_labels == i].mean() for i in range(2)]
            low_vol_regime = 0 if regime_means[0] < regime_means[1] else 1
            high_vol_regime = 1 - low_vol_regime
            
            # Calculate regime statistics
            low_vol_periods = np.sum(regime_labels == low_vol_regime)
            high_vol_periods = np.sum(regime_labels == high_vol_regime)
            total_periods = len(regime_labels)
            
            current_regime = regime_labels[-1]
            current_regime_name = 'Low Volatility' if current_regime == low_vol_regime else 'High Volatility'
            
            return {
                'current_regime': current_regime_name,
                'low_vol_regime_pct': (low_vol_periods / total_periods) * 100,
                'high_vol_regime_pct': (high_vol_periods / total_periods) * 100,
                'low_vol_avg': regime_means[low_vol_regime] * 100,
                'high_vol_avg': regime_means[high_vol_regime] * 100,
                'regime_labels': regime_labels,
                'rolling_vol': rolling_vol
            }
            
        except Exception as e:
            logger.error(f"Error in volatility regime detection: {e}")
            return {'regime_detection': 'error'}
    
    def _calculate_rolling_correlations(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate rolling correlations with benchmark."""
        
        # Generate simulated benchmark returns (S&P 500-like)
        np.random.seed(42)  # For reproducibility
        benchmark_returns = np.random.normal(0.0004, 0.012, len(returns))  # ~10% annual, 12% vol
        
        # Calculate rolling correlation
        window = min(60, len(returns) // 4)  # 60-day window or 1/4 of data
        if window < 20:
            return {'correlation_analysis': 'insufficient_data'}
        
        try:
            rolling_corr = returns.rolling(window=window).corr(pd.Series(benchmark_returns, index=returns.index))
            rolling_corr = rolling_corr.dropna()
            
            return {
                'current_correlation': rolling_corr.iloc[-1] if not rolling_corr.empty else 0,
                'avg_correlation': rolling_corr.mean(),
                'correlation_volatility': rolling_corr.std(),
                'max_correlation': rolling_corr.max(),
                'min_correlation': rolling_corr.min(),
                'rolling_correlation': rolling_corr,
                'benchmark_returns': pd.Series(benchmark_returns, index=returns.index)
            }
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlations: {e}")
            return {'correlation_analysis': 'error'}
    
    def _run_monte_carlo_simulation(self, returns: pd.Series, n_simulations: int = 1000, 
                                   time_horizon: int = 252) -> Dict[str, Any]:
        """Run Monte Carlo simulation for scenario analysis."""
        
        if len(returns) < 30:
            return {'monte_carlo': 'insufficient_data'}
        
        try:
            # Calculate return statistics
            mean_return = returns.mean()
            vol_return = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, vol_return, 
                                               (n_simulations, time_horizon))
            
            # Calculate cumulative returns for each simulation
            cumulative_returns = np.cumprod(1 + simulated_returns, axis=1) - 1
            final_returns = cumulative_returns[:, -1]
            
            # Calculate statistics
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            return_percentiles = np.percentile(final_returns * 100, percentiles)
            
            # Probability of loss
            prob_loss = np.sum(final_returns < 0) / n_simulations * 100
            
            # Expected shortfall scenarios
            worst_5pct = final_returns[final_returns <= np.percentile(final_returns, 5)]
            expected_shortfall = worst_5pct.mean() * 100
            
            return {
                'simulated_scenarios': cumulative_returns,
                'final_returns': final_returns,
                'return_percentiles': dict(zip(percentiles, return_percentiles)),
                'probability_of_loss': prob_loss,
                'expected_shortfall_5pct': expected_shortfall,
                'best_case_5pct': np.percentile(final_returns * 100, 95),
                'worst_case_5pct': np.percentile(final_returns * 100, 5)
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {'monte_carlo': 'error'}
    
    def _create_risk_kpi_cards(self, analytics_data: CachedAnalyticsData, 
                               extended_metrics: Dict[str, Any]) -> List[html.Div]:
        """Create KPI cards for risk metrics."""
        
        portfolio_metrics = analytics_data.portfolio_metrics
        
        # Primary risk KPI cards
        cards = [
            create_kpi_card(
                title="VaR (95%)",
                value=f"{extended_metrics.get('VaR_95%', portfolio_metrics.get('VaR 95% (%)', 0)):.2f}%",
                subtitle="Daily Value at Risk",
                color=COLORS['loss'],
                icon="fas fa-exclamation-triangle",
                trend="neutral",
                size="small"
            ),
            create_kpi_card(
                title="CVaR (95%)",
                value=f"{extended_metrics.get('CVaR_95%', portfolio_metrics.get('CVaR 95% (%)', 0)):.2f}%",
                subtitle="Conditional Value at Risk",
                color=COLORS['loss'],
                icon="fas fa-shield-alt",
                trend="neutral",
                size="small"
            ),
            create_kpi_card(
                title="Tail Ratio",
                value=f"{extended_metrics.get('tail_ratio', 0):.2f}",
                subtitle="95th/5th percentile",
                color=COLORS['caution'] if extended_metrics.get('tail_ratio', 0) > 1.5 else COLORS['info'],
                icon="fas fa-chart-area",
                trend="up" if extended_metrics.get('tail_ratio', 0) > 1.5 else "neutral",
                size="small"
            ),
            create_kpi_card(
                title="Correlation",
                value=f"{extended_metrics.get('current_correlation', 0):.2f}",
                subtitle="vs Benchmark",
                color=COLORS['info'],
                icon="fas fa-link",
                trend="neutral",
                size="small"
            ),
            
            # Advanced risk metrics
            create_kpi_card(
                title="VaR (99%)",
                value=f"{extended_metrics.get('VaR_99%', portfolio_metrics.get('VaR 99% (%)', 0)):.2f}%",
                subtitle="Extreme Value at Risk",
                color=COLORS['loss'],
                icon="fas fa-exclamation-circle",
                size="small"
            ),
            create_kpi_card(
                title="Modified VaR",
                value=f"{extended_metrics.get('modified_var_95%', 0):.2f}%",
                subtitle="Cornish-Fisher adjusted",
                color=COLORS['loss'],
                icon="fas fa-calculator",
                size="small"
            ),
            create_kpi_card(
                title="Volatility Regime",
                value=extended_metrics.get('current_regime', 'Unknown'),
                subtitle="Current market state",
                color=COLORS['caution'] if extended_metrics.get('current_regime', '') == 'High Volatility' else COLORS['info'],
                icon="fas fa-thermometer-half",
                size="small"
            ),
            create_kpi_card(
                title="Monte Carlo Loss Prob",
                value=f"{extended_metrics.get('probability_of_loss', 0):.1f}%",
                subtitle="1-year horizon",
                color=COLORS['caution'],
                icon="fas fa-dice",
                size="small"
            )
        ]
        
        return cards
    
    def _create_risk_charts(self, analytics_data: CachedAnalyticsData, 
                           extended_metrics: Dict[str, Any]) -> Dict[str, html.Div]:
        """Create comprehensive risk analysis charts."""
        
        charts = {}
        
        # VaR/CVaR Multi-Level Analysis
        charts['var_cvar_analysis'] = self._create_var_cvar_chart(analytics_data, extended_metrics)
        
        # Rolling Volatility Analysis
        charts['rolling_volatility'] = self._create_rolling_volatility_chart(analytics_data, extended_metrics)
        
        # Correlation Analysis
        charts['correlation_analysis'] = self._create_correlation_chart(analytics_data, extended_metrics)
        
        # Tail Risk Distribution Analysis
        charts['tail_risk_analysis'] = self._create_tail_risk_chart(analytics_data, extended_metrics)
        
        # Volatility Regime Detection
        charts['regime_detection'] = self._create_regime_detection_chart(analytics_data, extended_metrics)
        
        # Monte Carlo Scenario Analysis
        charts['monte_carlo'] = self._create_monte_carlo_chart(analytics_data, extended_metrics)
        
        # Enhanced Underwater Plot
        charts['underwater_plot'] = self._create_enhanced_underwater_plot(analytics_data)
        
        return charts
    
    def _create_var_cvar_chart(self, analytics_data: CachedAnalyticsData, 
                               extended_metrics: Dict[str, Any]) -> html.Div:
        """Create VaR/CVaR analysis chart with multiple confidence levels."""
        
        # Create figure with subplots with proper spacing
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'VaR at Multiple Confidence Levels',
                'CVaR at Multiple Confidence Levels', 
                'VaR vs CVaR Comparison',
                'Risk Metrics Distribution'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15,  # Add vertical spacing between plots
            horizontal_spacing=0.12  # Add horizontal spacing between plots
        )
        
        # Confidence levels and values
        confidence_levels = ['90%', '95%', '99%', '99.9%']
        var_values = [extended_metrics.get(f'VaR_{level[:-1]}%', 0) for level in confidence_levels]
        cvar_values = [extended_metrics.get(f'CVaR_{level[:-1]}%', 0) for level in confidence_levels]
        
        # VaR bar chart
        fig.add_trace(
            go.Bar(
                x=confidence_levels,
                y=var_values,
                name='VaR',
                marker_color=COLORS['loss'],
                text=[f'{val:.2f}%' for val in var_values],
                textposition='inside',  # Changed to inside to prevent overlap
                textfont=dict(color='white', size=11)  # White text for visibility
            ),
            row=1, col=1
        )
        
        # CVaR bar chart
        fig.add_trace(
            go.Bar(
                x=confidence_levels,
                y=cvar_values,
                name='CVaR',
                marker_color=COLORS['caution'],
                text=[f'{val:.2f}%' for val in cvar_values],
                textposition='inside',  # Changed to inside to prevent overlap
                textfont=dict(color='white', size=11)  # White text for visibility
            ),
            row=1, col=2
        )
        
        # VaR vs CVaR comparison
        fig.add_trace(
            go.Scatter(
                x=confidence_levels,
                y=var_values,
                mode='lines+markers',
                name='VaR',
                line=dict(color=COLORS['loss'], width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=confidence_levels,
                y=cvar_values,
                mode='lines+markers',
                name='CVaR',
                line=dict(color=COLORS['caution'], width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # Risk metrics summary (radar chart style)
        categories = ['VaR 95%', 'CVaR 95%', 'Tail Ratio', 'Skewness', 'Kurtosis']
        values = [
            abs(extended_metrics.get('VaR_95%', 0)),
            abs(extended_metrics.get('CVaR_95%', 0)),
            extended_metrics.get('tail_ratio', 0),
            abs(analytics_data.portfolio_metrics.get('Skewness', 0)),
            abs(analytics_data.portfolio_metrics.get('Kurtosis', 0))
        ]
        
        # Normalize values for comparison (0-100 scale)
        max_val = max(values) if values else 1
        normalized_values = [(val / max_val) * 100 for val in values]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=normalized_values,
                name='Risk Metrics',
                marker_color=COLORS['info'],
                text=[f'{val:.2f}' for val in values],
                textposition='inside',  # Changed to inside to prevent overlap
                textfont=dict(color='white', size=10)  # White text for visibility
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="VaR/CVaR Multi-Level Risk Analysis",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            height=700,  # Increased height for better spacing
            margin=dict(l=80, r=80, t=100, b=80),  # Add margins to prevent label overlap
            # Add slim plus sign dividers to segment the 4 charts (positioned to avoid touching axes)
            shapes=[
                # Vertical divider line (center) - adjusted to avoid x-axis
                dict(
                    type="line",
                    x0=0.5, x1=0.5,
                    y0=0.15, y1=0.85,  # Reduced span to avoid touching axes
                    xref="paper", yref="paper",
                    line=dict(color=COLORS['border'], width=1.5, dash="solid"),
                    layer="above"
                ),
                # Horizontal divider line (center) - adjusted to avoid y-axis
                dict(
                    type="line",
                    x0=0.15, x1=0.85,  # Reduced span to avoid touching axes
                    y0=0.47, y1=0.47,  # Moved down by couple of pixels
                    xref="paper", yref="paper",
                    line=dict(color=COLORS['border'], width=1.5, dash="solid"),
                    layer="above"
                ),
                # Plus sign center circle (small decorative element)
                dict(
                    type="circle",
                    x0=0.485, x1=0.515,
                    y0=0.455, y1=0.485,  # Moved down by couple of pixels
                    xref="paper", yref="paper",
                    fillcolor=COLORS['background'],
                    line=dict(color=COLORS['border'], width=1.5),
                    layer="above"
                )
            ]
        )
        
        # Update axis labels with improved spacing and formatting
        fig.update_xaxes(
            title_text="Confidence Level", 
            row=1, col=1,
            title_standoff=15,  # Add spacing between axis and title
            tickangle=0         # Keep labels horizontal
        )
        fig.update_xaxes(
            title_text="Confidence Level", 
            row=1, col=2,
            title_standoff=15,
            tickangle=0
        )
        fig.update_xaxes(
            title_text="Confidence Level", 
            row=2, col=1,
            title_standoff=15,
            tickangle=0
        )
        fig.update_xaxes(
            title_text="Risk Metrics", 
            row=2, col=2,
            title_standoff=15,
            tickangle=-45,      # Angle labels to prevent overlap
            tickfont=dict(size=10)  # Smaller font for better fit
        )
        
        fig.update_yaxes(
            title_text="VaR (%)", 
            row=1, col=1,
            title_standoff=15,
            tickformat=".2f"    # Format with 2 decimal places
        )
        fig.update_yaxes(
            title_text="CVaR (%)", 
            row=1, col=2,
            title_standoff=15,
            tickformat=".2f"
        )
        fig.update_yaxes(
            title_text="Risk Value (%)", 
            row=2, col=1,
            title_standoff=15,
            tickformat=".2f"
        )
        fig.update_yaxes(
            title_text="Normalized Score", 
            row=2, col=2,
            title_standoff=15,
            tickformat=".1f"
        )
        
        return create_base_card(
            html.Div([
                html.H3("VaR/CVaR Risk Analysis", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '700px'}  # Updated to match figure height
                )
            ]),
            'chart',
            'chart'
        )
    
    def _create_rolling_volatility_chart(self, analytics_data: CachedAnalyticsData, 
                                         extended_metrics: Dict[str, Any]) -> html.Div:
        """Create rolling volatility analysis chart."""
        
        equity_curve = analytics_data.equity_curve
        if equity_curve is None or equity_curve.empty:
            return create_base_card(
                html.P("No data available for volatility analysis", style={'textAlign': 'center'}),
                size='large'
            )
        
        returns = equity_curve.pct_change().dropna()
        
        # Calculate multiple rolling windows
        windows = [20, 60, 252]  # 1 month, 3 months, 1 year
        colors = [COLORS['info'], COLORS['caution'], COLORS['loss']]
        
        fig = go.Figure()
        
        for i, window in enumerate(windows):
            if len(returns) > window:
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
                rolling_vol = rolling_vol.dropna()
                
                fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name=f'{window}-day Rolling Volatility',
                    line=dict(color=colors[i], width=2),
                    hovertemplate=f'<b>%{{y:.1f}}%</b><br>%{{x}}<extra>{window}-day</extra>'
                ))
        
        # Add regime detection overlay if available
        if 'rolling_vol' in extended_metrics and 'regime_labels' in extended_metrics:
            regime_vol = extended_metrics['rolling_vol'] * 100
            regime_labels = extended_metrics['regime_labels']
            
            # Color background based on regime
            for i in range(len(regime_labels)):
                if i < len(regime_vol):
                    color = COLORS['profit'] if regime_labels[i] == 0 else COLORS['loss']
                    fig.add_vrect(
                        x0=regime_vol.index[i],
                        x1=regime_vol.index[min(i+1, len(regime_vol)-1)],
                        fillcolor=color,
                        opacity=0.1,
                        layer="below",
                        line_width=0
                    )
        
        fig.update_layout(
            title="Rolling Volatility Analysis with Regime Detection",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=40, t=60, b=80)  # Optimized margins for full-width layout
        )
        
        fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
        fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
        
        return create_base_card(
            html.Div([
                html.H3("Rolling Volatility Analysis", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '500px'}  # Increased height for better space utilization
                )
            ]),
            'full-width',  # Changed from 'large' to 'full-width' for optimal space utilization
            'chart'
        )
    
    def _create_correlation_chart(self, analytics_data: CachedAnalyticsData, 
                                  extended_metrics: Dict[str, Any]) -> html.Div:
        """Create correlation analysis chart."""
        
        if 'rolling_correlation' not in extended_metrics:
            return create_base_card(
                html.P("Correlation data not available", style={'textAlign': 'center'}),
                size='large'
            )
        
        rolling_corr = extended_metrics['rolling_correlation']
        
        # Create subplot with correlation and benchmark comparison
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Rolling Correlation with Benchmark', 'Correlation Distribution'],
            vertical_spacing=0.18  # Increased spacing for better visual separation with divider line
        )
        
        # Rolling correlation time series
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode='lines',
                name='Rolling Correlation',
                line=dict(color=COLORS['info'], width=2),
                fill='tonexty',
                fillcolor='rgba(0, 123, 255, 0.1)',
                hovertemplate='<b>%{y:.3f}</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add correlation benchmarks
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS['caution'], 
                     annotation_text="Moderate Correlation", row=1, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color=COLORS['loss'], 
                     annotation_text="High Correlation", row=1, col=1)
        
        # Correlation distribution histogram
        fig.add_trace(
            go.Histogram(
                x=rolling_corr.values,
                nbinsx=20,
                name='Correlation Distribution',
                marker_color=COLORS['info'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Correlation Analysis with Market Benchmark",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            height=800,  # Increased height for better spacing and visual separation
            margin=dict(l=80, r=80, t=80, b=100),  # Full-width optimized margins
            # Add visual separator between subplots
            shapes=[
                dict(
                    type="line",
                    x0=0, x1=1,
                    y0=0.47, y1=0.47,  # Positioned between the two subplots
                    xref="paper", yref="paper",
                    line=dict(color=COLORS['border'], width=2, dash="solid"),
                    layer="above"
                )
            ]
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Correlation", row=2, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        # Add correlation statistics
        corr_stats = html.Div([
            html.Div([
                html.Span("Current: ", style={'fontWeight': '500'}),
                html.Span(f"{extended_metrics.get('current_correlation', 0):.3f}", style={
                    'color': COLORS['info'], 'fontWeight': '600'
                })
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span("Average: ", style={'fontWeight': '500'}),
                html.Span(f"{extended_metrics.get('avg_correlation', 0):.3f}", style={
                    'color': COLORS['info'], 'fontWeight': '600'
                })
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span("Volatility: ", style={'fontWeight': '500'}),
                html.Span(f"{extended_metrics.get('correlation_volatility', 0):.3f}", style={
                    'color': COLORS['caution'], 'fontWeight': '600'
                })
            ])
        ], style={
            'backgroundColor': COLORS['page_bg'],
            'padding': '10px',
            'borderRadius': '4px',
            'marginTop': '10px',
            'fontSize': '12px'
        })
        
        return create_base_card(
            html.Div([
                html.H3("Correlation Analysis", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '800px'}  # Increased height with visual separator for clear subplot distinction
                ),
                corr_stats
            ]),
            'full-width',  # Using 'full-width' for adequate size - chart was too small in screenshot
            'chart'
        )
    
    def _create_tail_risk_chart(self, analytics_data: CachedAnalyticsData, 
                                extended_metrics: Dict[str, Any]) -> html.Div:
        """Create tail risk analysis chart with improved text positioning and spacing."""
        
        equity_curve = analytics_data.equity_curve
        if equity_curve is None or equity_curve.empty:
            return create_base_card(
                html.P("No data available for tail risk analysis", style={'textAlign': 'center'}),
                size='large'
            )
        
        returns = equity_curve.pct_change().dropna() * 100  # Convert to percentage
        
        # Create subplot with improved spacing and margins
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Return Distribution with Tail Risk', 'Q-Q Plot vs Normal Distribution'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
            horizontal_spacing=0.15,  # Increased spacing between subplots
            vertical_spacing=0.12     # Added vertical spacing for better layout
        )
        
        # Return distribution with tail highlights
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                nbinsx=50,
                name='Return Distribution',
                marker_color=COLORS['info'],
                opacity=0.7,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        # Add VaR lines with improved annotation positioning
        var_annotations = []
        y_positions = [0.85, 0.75]  # Staggered vertical positions to prevent overlap
        
        for i, (conf, color) in enumerate([(95, COLORS['caution']), (99, COLORS['loss'])]):
            var_value = extended_metrics.get(f'VaR_{conf}%', 0)
            fig.add_vline(
                x=var_value,
                line_dash="dash",
                line_color=color,
                row=1, col=1
            )
            
            # Add separate annotations with proper positioning to avoid overlap
            var_annotations.append(
                dict(
                    x=var_value,
                    y=y_positions[i],  # Use staggered y-positions
                    xref="x", yref="paper",
                    text=f"VaR {conf}%: {var_value:.2f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor=color,
                    ax=40 if i == 0 else -40,  # Alternate arrow directions
                    ay=-20,
                    font=dict(size=11, color=color),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        # Normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x_range, returns.mean(), returns.std())
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Distribution',
                line=dict(color=COLORS['tertiary'], width=2, dash='dash'),
                hovertemplate='<b>Normal Dist</b><br>Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Q-Q plot with improved marker size and hover info
        sorted_returns = np.sort(returns.values)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_returns,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color=COLORS['info'], size=5, opacity=0.7),  # Slightly larger and more opaque
                hovertemplate='<b>Q-Q Plot</b><br>Theoretical: %{x:.2f}<br>Sample: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add perfect normal line with better styling
        perfect_normal_line = theoretical_quantiles * returns.std() + returns.mean()
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=perfect_normal_line,
                mode='lines',
                name='Perfect Normal',
                line=dict(color=COLORS['loss'], width=2.5, dash='dot'),
                hovertemplate='<b>Perfect Normal</b><br>Theoretical: %{x:.2f}<br>Expected: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Calculate statistics for text display
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        fig.update_layout(
            title={
                'text': "Tail Risk and Distribution Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': "Inter, sans-serif"}
            },
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            height=500,  # Increased height for better text spacing
            margin=dict(l=80, r=80, t=100, b=80),  # Increased margins to prevent text cutoff
            annotations=var_annotations  # Add our custom VaR annotations
        )
        
        # Update axes with improved spacing and formatting
        fig.update_xaxes(
            title_text="Daily Returns (%)", 
            row=1, col=1,
            title_standoff=20,  # Add spacing between axis and title
            tickformat=".1f"    # Format tick labels
        )
        fig.update_xaxes(
            title_text="Theoretical Quantiles", 
            row=1, col=2,
            title_standoff=20,
            tickformat=".1f"
        )
        fig.update_yaxes(
            title_text="Probability Density", 
            row=1, col=1,
            title_standoff=20,
            tickformat=".3f"
        )
        fig.update_yaxes(
            title_text="Sample Quantiles (%)", 
            row=1, col=2,
            title_standoff=20,
            tickformat=".1f"
        )
        
        # Add distribution statistics box with proper spacing
        stats_text = html.Div([
            html.Div([
                html.Span("Distribution Statistics:", style={
                    'fontWeight': '600', 
                    'color': COLORS['primary'],
                    'fontSize': '14px',
                    'marginBottom': '8px',
                    'display': 'block'
                }),
                html.Div([
                    html.Span("Skewness: ", style={'fontWeight': '500', 'marginRight': '5px'}),
                    html.Span(f"{skewness:.3f}", style={
                        'color': COLORS['loss'] if abs(skewness) > 0.5 else COLORS['info'], 
                        'fontWeight': '600'
                    })
                ], style={'marginBottom': '4px'}),
                html.Div([
                    html.Span("Excess Kurtosis: ", style={'fontWeight': '500', 'marginRight': '5px'}),
                    html.Span(f"{kurtosis:.3f}", style={
                        'color': COLORS['loss'] if abs(kurtosis) > 1.0 else COLORS['info'], 
                        'fontWeight': '600'
                    })
                ], style={'marginBottom': '4px'}),
                html.Div([
                    html.Span("Tail Ratio: ", style={'fontWeight': '500', 'marginRight': '5px'}),
                    html.Span(f"{extended_metrics.get('tail_ratio', 0):.2f}", style={
                        'color': COLORS['caution'] if extended_metrics.get('tail_ratio', 0) > 1.5 else COLORS['info'], 
                        'fontWeight': '600'
                    })
                ])
            ])
        ], style={
            'backgroundColor': COLORS['page_bg'],
            'padding': '15px',
            'borderRadius': '6px',
            'marginTop': '15px',
            'fontSize': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'maxWidth': '400px'  # Limit width to prevent overflow
        })
        
        return create_base_card(
            html.Div([
                html.H3("Tail Risk Analysis", style={
                    'margin': '0',
                    'fontSize': '20px',  # Slightly larger for better hierarchy
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '25px'  # Increased bottom margin
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '500px'}  # Match the increased figure height
                ),
                stats_text
            ]),
            'large',
            'chart'
        )
    
    def _create_regime_detection_chart(self, analytics_data: CachedAnalyticsData, 
                                       extended_metrics: Dict[str, Any]) -> html.Div:
        """Create volatility regime detection chart."""
        
        if 'regime_labels' not in extended_metrics:
            return create_base_card(
                html.P("Regime detection data not available", style={'textAlign': 'center'}),
                size='large'
            )
        
        regime_labels = extended_metrics['regime_labels']
        rolling_vol = extended_metrics['rolling_vol'] * 100  # Convert to percentage
        
        # Create regime visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Volatility Regimes Over Time', 'Regime Statistics'],
            vertical_spacing=0.15
        )
        
        # Color-code volatility by regime
        regime_colors = [COLORS['profit'] if label == 0 else COLORS['loss'] for label in regime_labels]
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index[:len(regime_labels)],
                y=rolling_vol.values[:len(regime_labels)],
                mode='markers',
                name='Volatility by Regime',
                marker=dict(
                    color=regime_colors,
                    size=6,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate='<b>%{y:.1f}%</b><br>%{x}<br>Regime: %{text}<extra></extra>',
                text=['Low Vol' if label == 0 else 'High Vol' for label in regime_labels]
            ),
            row=1, col=1
        )
        
        # Add regime averages
        low_vol_avg = extended_metrics.get('low_vol_avg', 0)
        high_vol_avg = extended_metrics.get('high_vol_avg', 0)
        
        # Add regime averages with improved annotation positioning
        fig.add_hline(y=low_vol_avg, line_dash="dash", line_color=COLORS['profit'], row=1, col=1)
        fig.add_hline(y=high_vol_avg, line_dash="dash", line_color=COLORS['loss'], row=1, col=1)
        
        # Add custom annotations with better positioning to avoid overlap
        regime_annotations = [
            dict(
                x=0.85, y=low_vol_avg,
                xref="paper", yref="y",
                text=f"Low Vol Avg: {low_vol_avg:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=COLORS['profit'],
                ax=0, ay=-25,
                font=dict(size=11, color=COLORS['profit']),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLORS['profit'],
                borderwidth=1,
                borderpad=4
            ),
            dict(
                x=0.85, y=high_vol_avg,
                xref="paper", yref="y",
                text=f"High Vol Avg: {high_vol_avg:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=COLORS['loss'],
                ax=0, ay=25,
                font=dict(size=11, color=COLORS['loss']),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLORS['loss'],
                borderwidth=1,
                borderpad=4
            )
        ]
        
        # Regime distribution
        low_vol_pct = extended_metrics.get('low_vol_regime_pct', 0)
        high_vol_pct = extended_metrics.get('high_vol_regime_pct', 0)
        
        fig.add_trace(
            go.Bar(
                x=['Low Volatility', 'High Volatility'],
                y=[low_vol_pct, high_vol_pct],
                name='Regime Distribution',
                marker_color=[COLORS['profit'], COLORS['loss']],
                text=[f'{low_vol_pct:.1f}%', f'{high_vol_pct:.1f}%'],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title={
                'text': "Volatility Regime Detection Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': "Inter, sans-serif"}
            },
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            height=550,  # Increased height for better spacing
            margin=dict(l=80, r=80, t=100, b=80),  # Add margins to prevent text cutoff
            annotations=regime_annotations  # Add our custom regime annotations
        )
        
        # Update axes with improved spacing and formatting
        fig.update_xaxes(
            title_text="Date", 
            row=1, col=1,
            title_standoff=20
        )
        fig.update_xaxes(
            title_text="Regime Type", 
            row=2, col=1,
            title_standoff=20
        )
        fig.update_yaxes(
            title_text="Volatility (%)", 
            row=1, col=1,
            title_standoff=20,
            tickformat=".1f"
        )
        fig.update_yaxes(
            title_text="Time in Regime (%)", 
            row=2, col=1,
            title_standoff=20,
            tickformat=".1f"
        )
        
        # Current regime indicator
        current_regime = extended_metrics.get('current_regime', 'Unknown')
        regime_indicator = html.Div([
            html.Div([
                html.Span("Current Regime: ", style={'fontWeight': '500'}),
                html.Span(current_regime, style={
                    'backgroundColor': COLORS['caution'] if current_regime == 'High Volatility' else COLORS['info'],
                    'color': 'white',
                    'padding': '4px 12px',
                    'borderRadius': '12px',
                    'fontSize': '12px',
                    'fontWeight': '500'
                })
            ])
        ], style={'marginTop': '10px', 'textAlign': 'center'})
        
        return create_base_card(
            html.Div([
                html.H3("Volatility Regime Detection", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '550px'}  # Match the increased figure height
                ),
                regime_indicator
            ]),
            'large',
            'chart'
        )
    
    def _create_monte_carlo_chart(self, analytics_data: CachedAnalyticsData, 
                                  extended_metrics: Dict[str, Any]) -> html.Div:
        """Create Monte Carlo simulation results chart."""
        
        if 'simulated_scenarios' not in extended_metrics:
            return create_base_card(
                html.P("Monte Carlo data not available", style={'textAlign': 'center'}),
                size='large'
            )
        
        scenarios = extended_metrics['simulated_scenarios']
        return_percentiles = extended_metrics['return_percentiles']
        
        # Create Monte Carlo visualization with improved spacing
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Monte Carlo Scenario Paths (1000 simulations)', 'Final Return Distribution'],
            vertical_spacing=0.25  # Increased from 0.15 to 0.25 for better separation
        )
        
        # Plot sample of scenario paths
        time_axis = np.arange(scenarios.shape[1])
        sample_size = min(100, scenarios.shape[0])  # Show 100 paths max
        sample_indices = np.random.choice(scenarios.shape[0], sample_size, replace=False)
        
        for i in sample_indices[:20]:  # Show first 20 for clarity
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=scenarios[i] * 100,
                    mode='lines',
                    line=dict(color=COLORS['tertiary'], width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hovertemplate='<b>%{y:.1f}%</b><br>Day %{x}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add percentile bands
        p5 = np.percentile(scenarios, 5, axis=0) * 100
        p25 = np.percentile(scenarios, 25, axis=0) * 100
        p50 = np.percentile(scenarios, 50, axis=0) * 100
        p75 = np.percentile(scenarios, 75, axis=0) * 100
        p95 = np.percentile(scenarios, 95, axis=0) * 100
        
        # Add percentile bands
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=p95,
                mode='lines',
                name='95th Percentile',
                line=dict(color=COLORS['profit'], width=2),
                fill=None
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=p5,
                mode='lines',
                name='5th Percentile',
                line=dict(color=COLORS['loss'], width=2),
                fill='tonexty',
                fillcolor='rgba(255, 68, 68, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=p50,
                mode='lines',
                name='Median',
                line=dict(color=COLORS['primary'], width=3)
            ),
            row=1, col=1
        )
        
        # Final return distribution
        final_returns = extended_metrics['final_returns'] * 100
        fig.add_trace(
            go.Histogram(
                x=final_returns,
                nbinsx=50,
                name='Final Returns',
                marker_color=COLORS['info'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add percentile lines with improved annotation positioning to prevent overlap
        percentile_positions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]  # Staggered y-positions
        for i, (pct, value) in enumerate(return_percentiles.items()):
            color = COLORS['loss'] if pct <= 10 else COLORS['caution'] if pct <= 25 else COLORS['profit']
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color=color,
                row=2, col=1
            )
            # Add separate annotation with proper positioning to avoid overlap
            if i < len(percentile_positions):
                fig.add_annotation(
                    x=value,
                    y=percentile_positions[i],
                    xref="x2", yref="paper",  # Use x2 for second subplot
                    text=f"P{pct}: {value:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor=color,
                    ax=30 if i % 2 == 0 else -30,  # Alternate arrow directions
                    ay=-15,
                    font=dict(size=10, color=color),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=3
                )
        
        fig.update_layout(
            title="Monte Carlo Scenario Analysis (1-Year Horizon)",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            height=700,  # Increased from 600 to 700 for better spacing
            margin=dict(l=80, r=80, t=100, b=90),  # Added margins to prevent overlap
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Add visual separator between subplots for clear distinction
            shapes=[
                dict(
                    type="line",
                    x0=0, x1=1,
                    y0=0.47, y1=0.47,  # Positioned between the two subplots
                    xref="paper", yref="paper",
                    line=dict(color=COLORS['border'], width=2, dash="solid"),
                    layer="above"
                )
            ]
        )
        
        # Update axes with improved spacing and formatting
        fig.update_xaxes(
            title_text="Trading Days", 
            row=1, col=1,
            title_standoff=20  # Add spacing between axis and title
        )
        fig.update_xaxes(
            title_text="Final Return (%)", 
            row=2, col=1,
            title_standoff=20
        )
        fig.update_yaxes(
            title_text="Cumulative Return (%)", 
            row=1, col=1,
            title_standoff=20
        )
        fig.update_yaxes(
            title_text="Frequency", 
            row=2, col=1,
            title_standoff=20
        )
        
        # Monte Carlo statistics
        mc_stats = html.Div([
            html.Div([
                html.Span("Probability of Loss: ", style={'fontWeight': '500'}),
                html.Span(f"{extended_metrics.get('probability_of_loss', 0):.1f}%", style={
                    'color': COLORS['loss'], 'fontWeight': '600'
                })
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span("Expected Shortfall (5%): ", style={'fontWeight': '500'}),
                html.Span(f"{extended_metrics.get('expected_shortfall_5pct', 0):.1f}%", style={
                    'color': COLORS['loss'], 'fontWeight': '600'
                })
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span("Best Case (95%): ", style={'fontWeight': '500'}),
                html.Span(f"{extended_metrics.get('best_case_5pct', 0):.1f}%", style={
                    'color': COLORS['profit'], 'fontWeight': '600'
                })
            ])
        ], style={
            'backgroundColor': COLORS['page_bg'],
            'padding': '10px',
            'borderRadius': '4px',
            'marginTop': '10px',
            'fontSize': '12px'
        })
        
        return create_base_card(
            html.Div([
                html.H3("Monte Carlo Scenario Analysis", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '700px'}  # Updated to match increased figure height
                ),
                mc_stats
            ]),
            'large',
            'chart'
        )
    
    def _create_enhanced_underwater_plot(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create enhanced underwater plot with recovery analysis."""
        
        equity_curve = analytics_data.equity_curve
        if equity_curve is None or equity_curve.empty:
            return create_base_card(
                html.P("No equity data available for drawdown analysis", style={'textAlign': 'center'}),
                size='large'
            )
        
        # Calculate drawdown and recovery periods
        running_max = equity_curve.cummax()
        drawdown = ((equity_curve - running_max) / running_max) * 100
        
        # Identify drawdown periods and recovery times
        in_drawdown = drawdown < -0.01  # Consider >0.01% as drawdown
        
        fig = go.Figure()
        
        # Main underwater plot
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            line=dict(color=COLORS['loss'], width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.3)',
            hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
        ))
        
        # Highlight recovery periods
        if in_drawdown.any():
            drawdown_starts = []
            recovery_periods = []
            
            i = 0
            while i < len(in_drawdown):
                if in_drawdown.iloc[i]:
                    start_idx = i
                    # Find end of drawdown period
                    while i < len(in_drawdown) and in_drawdown.iloc[i]:
                        i += 1
                    end_idx = i - 1
                    
                    if end_idx < len(equity_curve) - 1:
                        drawdown_starts.append((start_idx, end_idx))
                        
                        # Calculate recovery time
                        peak_value = running_max.iloc[start_idx]
                        recovery_start = end_idx + 1
                        
                        # Find when equity recovers to previous peak
                        for j in range(recovery_start, len(equity_curve)):
                            if equity_curve.iloc[j] >= peak_value:
                                recovery_days = (equity_curve.index[j] - equity_curve.index[end_idx]).days
                                recovery_periods.append(recovery_days)
                                
                                # Add recovery annotation
                                fig.add_annotation(
                                    x=equity_curve.index[j],
                                    y=0,
                                    text=f"Recovery: {recovery_days}d",
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor=COLORS['profit'],
                                    font=dict(size=10, color=COLORS['profit'])
                                )
                                break
                        else:
                            recovery_periods.append(None)  # Still in recovery
                i += 1
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="solid", line_color=COLORS['border'], 
                     annotation_text="Peak Level")
        fig.add_hline(y=-5, line_dash="dash", line_color=COLORS['caution'], 
                     annotation_text="5% Drawdown")
        fig.add_hline(y=-10, line_dash="dash", line_color=COLORS['loss'], 
                     annotation_text="10% Drawdown")
        
        # Mark maximum drawdown point
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        fig.add_trace(go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value],
            mode='markers',
            name='Maximum Drawdown',
            marker=dict(color=COLORS['loss'], size=12, symbol='x'),
            hovertemplate=f'<b>Max DD: {max_dd_value:.2f}%</b><br>{max_dd_idx}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Enhanced Underwater Plot with Recovery Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
        fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
        
        # Recovery statistics
        if recovery_periods and any(r is not None for r in recovery_periods):
            valid_recoveries = [r for r in recovery_periods if r is not None]
            avg_recovery = np.mean(valid_recoveries) if valid_recoveries else 0
            max_recovery = max(valid_recoveries) if valid_recoveries else 0
            
            recovery_stats = html.Div([
                html.Div([
                    html.Span("Average Recovery Time: ", style={'fontWeight': '500'}),
                    html.Span(f"{avg_recovery:.0f} days", style={
                        'color': COLORS['caution'], 'fontWeight': '600'
                    })
                ], style={'marginBottom': '5px'}),
                html.Div([
                    html.Span("Longest Recovery: ", style={'fontWeight': '500'}),
                    html.Span(f"{max_recovery:.0f} days", style={
                        'color': COLORS['loss'], 'fontWeight': '600'
                    })
                ], style={'marginBottom': '5px'}),
                html.Div([
                    html.Span("Drawdown Periods: ", style={'fontWeight': '500'}),
                    html.Span(f"{len(recovery_periods)}", style={
                        'color': COLORS['info'], 'fontWeight': '600'
                    })
                ])
            ], style={
                'backgroundColor': COLORS['page_bg'],
                'padding': '10px',
                'borderRadius': '4px',
                'marginTop': '10px',
                'fontSize': '12px'
            })
        else:
            recovery_stats = html.P("No significant drawdown periods detected", 
                                  style={'color': COLORS['profit'], 'fontStyle': 'italic'})
        
        return create_base_card(
            html.Div([
                html.H3("Underwater Plot with Recovery Analysis", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '400px'}
                ),
                recovery_stats
            ]),
            'large',
            'chart'
        )
    
    def _create_placeholder_content(self) -> html.Div:
        """Create placeholder content when no data is available."""
        
        return html.Div([
            html.Div([
                html.H2("Risk Analysis", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P("Comprehensive risk assessment and exposure analysis", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '40px'}),
            
            create_base_card(
                html.Div([
                    html.I(className="fas fa-shield-alt", style={
                        'fontSize': '64px',
                        'color': COLORS['border'],
                        'marginBottom': '20px'
                    }),
                    html.H3("No Risk Data Available", style={
                        'color': COLORS['primary'],
                        'marginBottom': '15px',
                        'fontSize': '24px',
                        'fontWeight': '600'
                    }),
                    html.P("Please select a strategy and version to view comprehensive risk analytics including:", style={
                        'color': COLORS['tertiary'],
                        'marginBottom': '20px',
                        'fontSize': '16px'
                    }),
                    html.Ul([
                        html.Li("VaR and CVaR analysis at multiple confidence levels"),
                        html.Li("Rolling volatility and correlation analysis"),
                        html.Li("Tail risk analysis and extreme value statistics"),
                        html.Li("Underwater drawdown plots with recovery periods"),
                        html.Li("Monte Carlo simulation and scenario analysis"),
                        html.Li("Volatility regime detection and classification")
                    ], style={
                        'color': COLORS['tertiary'],
                        'textAlign': 'left',
                        'fontSize': '14px',
                        'lineHeight': '1.6'
                    })
                ], style={'textAlign': 'center'}),
                'full-width'
            )
            
        ], style={
            'padding': '40px',
            'backgroundColor': COLORS['page_bg'],
            'minHeight': 'calc(100vh - 90px)',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center',
            'alignItems': 'center'
        })


def create_risk_analysis_content(strategy: str = None, version: str = None, 
                                start_date: datetime = None, end_date: datetime = None) -> html.Div:
    """
    Factory function to create Risk Analysis tab content.
    
    Args:
        strategy: Selected strategy name
        version: Selected version identifier
        start_date: Filter start date
        end_date: Filter end date
        
    Returns:
        Dash HTML Div containing the Risk Analysis tab content
    """
    from .DashboardDataService import DashboardDataService
    
    data_service = DashboardDataService()
    risk_analysis_tab = RiskAnalysisTab(data_service)
    
    return risk_analysis_tab.create_tab_content(strategy, version, start_date, end_date)