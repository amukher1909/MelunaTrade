"""
Portfolio Overview Tab - Interactive Performance Analytics

This module implements the Portfolio Overview tab with comprehensive performance 
analytics and interactive charts integrating real data from PortfolioMetrics module.

Features:
- Interactive equity curve with benchmark comparison
- Performance metrics (CAGR, Sharpe Ratio, Maximum Drawdown)
- Risk-adjusted ratios (Calmar, Sortino, Information Ratio) 
- Rolling window analysis
- Returns distribution analysis
- Drawdown underwater plot visualization
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

from dash import dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate

from .DashboardComponents import (
    create_kpi_card, create_chart_container, create_card_grid, 
    create_card_group, create_base_card, COLORS, CARD_SIZES
)
from .DashboardDataService import DashboardDataService, CachedAnalyticsData
from ..metrics.PortfolioMetrics import PortfolioMetrics
from .InteractiveEquityCurve import create_interactive_equity_curve_component
from .UnifiedPortfolioStatisticsTable import create_unified_portfolio_statistics_table

logger = logging.getLogger(__name__)


class PortfolioOverviewTab:
    """
    Portfolio Overview tab component providing comprehensive performance analytics.
    
    Integrates with PortfolioMetrics module to display real backtest data through
    interactive charts and KPI cards with professional styling.
    """
    
    def __init__(self, data_service: DashboardDataService):
        """
        Initialize Portfolio Overview tab.
        
        Args:
            data_service: Dashboard data service for loading analytics data
        """
        self.data_service = data_service
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def create_tab_content(self, strategy: str = None, version: str = None, 
                          start_date: datetime = None, end_date: datetime = None) -> html.Div:
        """
        Create Portfolio Overview tab content with real data integration.
        
        Args:
            strategy: Selected strategy name
            version: Selected version identifier  
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            Dash HTML Div containing the complete Portfolio Overview tab
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
        
        metrics = analytics_data.portfolio_metrics
        equity_curve = analytics_data.equity_curve
        
        # Create KPI cards with real data
        kpi_cards = self._create_kpi_cards(metrics)
        
        # Create charts with real data
        charts = self._create_charts(analytics_data)
        
        return html.Div([
            # Tab Header
            html.Div([
                html.H2("Portfolio Overview", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P("Comprehensive portfolio performance metrics and interactive analytics", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '20px'}),
            
            # Key Performance Indicators
            create_card_group("Key Performance Indicators", kpi_cards),
            
            # Primary Interactive Equity Curve Chart (Issue #32 - Primary Feature)
            html.Div([
                charts['equity_curve']
            ], style={'marginBottom': '40px'}),
            
            # Secondary Charts Section
            html.Div([
                create_card_grid([
                    charts['returns_distribution']
                ])
            ], style={'marginBottom': '30px'}),
            
            # Rolling Metrics Analysis
            html.Div([
                create_card_grid([
                    charts['rolling_metrics']
                ])
            ], style={'marginBottom': '30px'}),
            
            # Unified Portfolio Statistics Table (Issue #61 - Replaces fragmented sections)
            html.Div([
                create_unified_portfolio_statistics_table(analytics_data)
            ], style={'marginBottom': '30px'})
            
        ], style={
            'padding': '20px',
            'backgroundColor': COLORS['page_bg'],
            'minHeight': 'calc(100vh - 90px)'
        })
    
    def _create_placeholder_content(self) -> html.Div:
        """Create placeholder content when no data is available."""
        
        placeholder_cards = [
            create_kpi_card(
                title="Total Return", 
                value="--", 
                subtitle="Select strategy and version", 
                color=COLORS['tertiary'], 
                icon="fas fa-chart-line",
                size="small",
                loading=False
            ),
            create_kpi_card(
                title="Sharpe Ratio", 
                value="--", 
                subtitle="Risk-adjusted returns", 
                color=COLORS['tertiary'], 
                icon="fas fa-balance-scale",
                size="small"
            ),
            create_kpi_card(
                title="Max Drawdown", 
                value="--", 
                subtitle="Maximum loss from peak", 
                color=COLORS['tertiary'], 
                icon="fas fa-arrow-down",
                size="small"
            ),
            create_kpi_card(
                title="Volatility", 
                value="--", 
                subtitle="Annualized volatility", 
                color=COLORS['tertiary'], 
                icon="fas fa-wave-square",
                size="small"
            )
        ]
        
        # Placeholder chart
        placeholder_chart = create_chart_container(
            "portfolio-equity-curve",
            "Portfolio Equity Curve",
            "400px",
            'chart'
        )
        
        return html.Div([
            # Tab Header
            html.Div([
                html.H2("Portfolio Overview", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P("Select a strategy and version to view portfolio performance analytics", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '20px'}),
            
            # Placeholder KPI Cards
            create_card_group("Key Performance Indicators", placeholder_cards),
            
            # Placeholder Chart
            html.Div([placeholder_chart], style={'marginBottom': '40px'}),
            
            # Instructions
            create_base_card(
                html.Div([
                    html.H4("Getting Started", style={
                        'color': COLORS['primary'],
                        'marginBottom': '15px',
                        'fontSize': '18px',
                        'fontWeight': '600'
                    }),
                    html.P("To view portfolio performance analytics:", style={
                        'color': COLORS['tertiary'], 
                        'marginBottom': '10px'
                    }),
                    html.Ol([
                        html.Li("Select a strategy from the dropdown in the header"),
                        html.Li("Choose a version to analyze"),
                        html.Li("Optionally set a date range filter"),
                        html.Li("View comprehensive performance metrics and interactive charts")
                    ], style={'color': COLORS['tertiary'], 'marginLeft': '20px'})
                ]),
                'full-width'
            )
            
        ], style={
            'padding': '20px', 
            'backgroundColor': COLORS['page_bg'],
            'minHeight': 'calc(100vh - 90px)'
        })
    
    def _create_kpi_cards(self, metrics: Dict[str, Any]) -> List[html.Div]:
        """Create KPI cards with real portfolio metrics."""
        
        # Safe value extraction with defaults
        total_return = metrics.get('Total Return (%)', 0)
        cagr = metrics.get('CAGR (%)', 0) 
        sharpe_ratio = metrics.get('Sharpe Ratio', 0)
        max_drawdown = metrics.get('Max Drawdown (%)', 0)
        volatility = metrics.get('Annualized Volatility (%)', 0)
        sortino_ratio = metrics.get('Sortino Ratio', 0)
        
        # Determine trend directions
        total_return_trend = 'up' if total_return > 0 else 'down' if total_return < 0 else 'neutral'
        sharpe_trend = 'up' if sharpe_ratio > 1.0 else 'neutral' if sharpe_ratio > 0 else 'down'
        drawdown_trend = 'neutral'  # Drawdown is always negative, so neutral for display
        volatility_trend = 'down' if volatility < 15 else 'neutral' if volatility < 25 else 'up'
        
        return [
            create_kpi_card(
                title="Total Return",
                value=f"{total_return:+.1f}%",
                subtitle=f"CAGR: {cagr:.1f}%",
                color=COLORS['profit'] if total_return > 0 else COLORS['loss'],
                icon="fas fa-chart-line",
                trend=total_return_trend,
                size="small"
            ),
            create_kpi_card(
                title="Sharpe Ratio", 
                value=f"{sharpe_ratio:.2f}",
                subtitle="Risk-adjusted returns",
                color=COLORS['info'] if sharpe_ratio > 1 else COLORS['caution'],
                icon="fas fa-balance-scale",
                trend=sharpe_trend,
                size="small"
            ),
            create_kpi_card(
                title="Max Drawdown",
                value=f"{max_drawdown:.1f}%",
                subtitle=f"Sortino: {sortino_ratio:.2f}",
                color=COLORS['loss'],
                icon="fas fa-arrow-down", 
                trend=drawdown_trend,
                size="small"
            ),
            create_kpi_card(
                title="Volatility",
                value=f"{volatility:.1f}%",
                subtitle="Annualized",
                color=COLORS['caution'],
                icon="fas fa-wave-square",
                trend=volatility_trend,
                size="small"
            )
        ]
    
    def _create_charts(self, analytics_data: CachedAnalyticsData) -> Dict[str, html.Div]:
        """Create interactive charts with real data."""
        
        charts = {}
        
        # Enhanced Interactive Equity curve chart (primary feature from issue #32)
        charts['equity_curve'] = create_interactive_equity_curve_component(
            self.data_service, analytics_data, selected_benchmark='SPY', selected_time_range='All'
        )
        
        # Returns distribution chart  
        charts['returns_distribution'] = self._create_returns_distribution_chart(analytics_data)
        
        # Rolling metrics chart
        charts['rolling_metrics'] = self._create_rolling_metrics_chart(analytics_data)
        
        return charts
    
    def _create_equity_curve_chart(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create interactive equity curve chart with benchmark comparison."""
        
        equity_curve = analytics_data.equity_curve
        if equity_curve is None or equity_curve.empty:
            return create_base_card(
                html.P("No equity curve data available", style={'textAlign': 'center'}),
                size='chart'
            )
        
        # Create figure
        fig = go.Figure()
        
        # Add portfolio equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Portfolio',
            line=dict(color=COLORS['info'], width=2),
            hovertemplate='<b>%{y:₹,.0f}</b><br>%{x}<extra></extra>'
        ))
        
        # Add benchmark (simulated S&P 500 for demonstration)
        if len(equity_curve) > 1:
            start_value = equity_curve.iloc[0]
            benchmark_returns = np.random.normal(0.0005, 0.015, len(equity_curve))  # ~13% annual, 15% vol
            benchmark_curve = start_value * np.cumprod(1 + benchmark_returns)
            
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=benchmark_curve,
                mode='lines', 
                name='Benchmark (S&P 500)',
                line=dict(color=COLORS['tertiary'], width=2, dash='dash'),
                hovertemplate='<b>%{y:₹,.0f}</b><br>%{x}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Portfolio Equity Curve vs Benchmark',
                'font': {'size': 18, 'color': COLORS['primary']},
                'x': 0
            },
            xaxis_title='Date',
            yaxis_title='Portfolio Value (₹)',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        # Style axes
        fig.update_xaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False,
            tickformat='₹,.0f'
        )
        
        # Create chart container
        content = html.Div([
            html.Div([
                html.H3("Portfolio Equity Curve", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'flex': '1'
                }),
                html.Div([
                    html.Button([
                        html.I(className="fas fa-expand-alt")
                    ], style={
                        'background': 'none',
                        'border': 'none',
                        'color': COLORS['tertiary'],
                        'cursor': 'pointer',
                        'padding': '8px',
                        'fontSize': '14px'
                    }, title="Fullscreen")
                ])
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'marginBottom': '20px',
                'paddingBottom': '12px',
                'borderBottom': f'1px solid {COLORS["border"]}'
            }),
            dcc.Graph(
                id='portfolio-equity-curve',
                figure=fig,
                style={'height': '400px'}
            )
        ])
        
        return create_base_card(content, size='chart', card_type='chart')
    
    def _create_returns_distribution_chart(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create returns distribution histogram with normal distribution overlay."""
        
        equity_curve = analytics_data.equity_curve
        if equity_curve is None or equity_curve.empty:
            return create_base_card(
                html.P("No returns data available", style={'textAlign': 'center'}),
                size='large'
            )
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna() * 100  # Convert to percentage
        
        if returns.empty:
            return create_base_card(
                html.P("Insufficient data for returns analysis", style={'textAlign': 'center'}),
                size='large'
            )
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=30,
                name='Returns Distribution',
                marker_color=COLORS['info'],
                opacity=0.7,
                histnorm='probability density'
            )
        )
        
        # Add normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1/np.sqrt(2*np.pi*returns.var())) * np.exp(-0.5*((x_range - returns.mean())**2)/returns.var())
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Distribution',
                line=dict(color=COLORS['loss'], width=2, dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Daily Returns Distribution',
                'font': {'size': 18, 'color': COLORS['primary']},
                'x': 0
            },
            xaxis_title='Daily Returns (%)',
            yaxis_title='Probability Density',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right", 
                x=1
            ),
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        # Style axes
        fig.update_xaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=True,
            zerolinecolor=COLORS['border']
        )
        fig.update_yaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False
        )
        
        # Create chart container
        content = html.Div([
            html.Div([
                html.H3("Returns Distribution", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'flex': '1'
                }),
                html.Div([
                    html.Span(f"Mean: {returns.mean():.2f}%", style={
                        'marginRight': '15px', 
                        'fontSize': '12px',
                        'color': COLORS['tertiary']
                    }),
                    html.Span(f"Std: {returns.std():.2f}%", style={
                        'marginRight': '15px',
                        'fontSize': '12px', 
                        'color': COLORS['tertiary']
                    }),
                    html.Span(f"Skewness: {returns.skew():.2f}", style={
                        'fontSize': '12px',
                        'color': COLORS['tertiary']
                    })
                ])
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'marginBottom': '20px',
                'paddingBottom': '12px',
                'borderBottom': f'1px solid {COLORS["border"]}'
            }),
            dcc.Graph(
                id='returns-distribution',
                figure=fig,
                style={'height': '300px'}
            )
        ])
        
        return create_base_card(content, size='large', card_type='chart')
    
    def _create_rolling_metrics_chart(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create rolling Sharpe ratio and volatility charts."""
        
        equity_curve = analytics_data.equity_curve
        if equity_curve is None or equity_curve.empty or len(equity_curve) < 252:
            return create_base_card(
                html.P("Insufficient data for rolling analysis (requires 252+ observations)", 
                      style={'textAlign': 'center'}),
                size='large'
            )
        
        # Calculate rolling metrics
        portfolio_metrics = PortfolioMetrics(equity_curve, self.risk_free_rate)
        rolling_data = portfolio_metrics.get_rolling_metrics(window_days=60)  # 60-day rolling window
        
        if not rolling_data:
            return create_base_card(
                html.P("Unable to calculate rolling metrics", style={'textAlign': 'center'}),
                size='large'
            )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio (60-day)', 'Rolling Volatility (60-day)'),
            vertical_spacing=0.1
        )
        
        # Rolling Sharpe ratio
        if 'Rolling Sharpe Ratio' in rolling_data:
            sharpe_data = rolling_data['Rolling Sharpe Ratio'].dropna()
            fig.add_trace(
                go.Scatter(
                    x=sharpe_data.index,
                    y=sharpe_data.values,
                    mode='lines',
                    name='Sharpe Ratio',
                    line=dict(color=COLORS['info'], width=2),
                    hovertemplate='<b>%{y:.2f}</b><br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Rolling volatility
        if 'Rolling Volatility' in rolling_data:
            vol_data = rolling_data['Rolling Volatility'].dropna() * 100  # Convert to percentage
            fig.add_trace(
                go.Scatter(
                    x=vol_data.index,
                    y=vol_data.values,
                    mode='lines',
                    name='Volatility',
                    line=dict(color=COLORS['caution'], width=2),
                    hovertemplate='<b>%{y:.1f}%</b><br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        # Style axes
        fig.update_xaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False
        )
        
        # Create chart container
        content = html.Div([
            html.Div([
                html.H3("Rolling Performance Metrics", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary']
                })
            ], style={
                'marginBottom': '20px',
                'paddingBottom': '12px',
                'borderBottom': f'1px solid {COLORS["border"]}'
            }),
            dcc.Graph(
                id='rolling-metrics',
                figure=fig,
                style={'height': '400px'}
            )
        ])
        
        return create_base_card(content, size='chart', card_type='chart')
    
    
    def _create_performance_attribution_card(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create performance attribution and statistical significance display."""
        
        metrics = analytics_data.portfolio_metrics
        
        # Extract key metrics
        cagr = metrics.get('CAGR (%)', 0)
        volatility = metrics.get('Annualized Volatility (%)', 0)
        sharpe = metrics.get('Sharpe Ratio', 0)
        sortino = metrics.get('Sortino Ratio', 0)
        calmar = cagr / abs(metrics.get('Max Drawdown (%)', 1)) if metrics.get('Max Drawdown (%)') else 0
        
        var_95 = metrics.get('VaR 95% (%)', 0)
        cvar_95 = metrics.get('CVaR 95% (%)', 0)
        skewness = metrics.get('Skewness', 0)
        kurtosis = metrics.get('Kurtosis', 0)
        
        gain_to_pain = metrics.get('Gain-to-Pain Ratio', 0)
        
        content = html.Div([
            html.H4("Performance Attribution & Risk Analysis", style={
                'color': COLORS['primary'],
                'marginBottom': '20px',
                'fontSize': '18px',
                'fontWeight': '600'
            }),
            
            # Risk-Adjusted Ratios
            html.Div([
                html.Div([
                    html.H5("Risk-Adjusted Ratios", style={
                        'color': COLORS['secondary'],
                        'marginBottom': '15px',
                        'fontSize': '16px'
                    }),
                    html.Div([
                        html.Div([
                            html.Span("Sharpe Ratio: ", style={'fontWeight': '500'}),
                            html.Span(f"{sharpe:.2f}", style={
                                'color': COLORS['profit'] if sharpe > 1 else COLORS['caution'],
                                'fontWeight': '600'
                            })
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("Sortino Ratio: ", style={'fontWeight': '500'}),
                            html.Span(f"{sortino:.2f}", style={
                                'color': COLORS['profit'] if sortino > 1.5 else COLORS['caution'],
                                'fontWeight': '600'
                            })
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("Calmar Ratio: ", style={'fontWeight': '500'}),
                            html.Span(f"{calmar:.2f}", style={
                                'color': COLORS['profit'] if calmar > 1 else COLORS['caution'],
                                'fontWeight': '600'
                            })
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("Gain-to-Pain: ", style={'fontWeight': '500'}),
                            html.Span(f"{gain_to_pain:.2f}", style={
                                'color': COLORS['profit'] if gain_to_pain > 2 else COLORS['caution'],
                                'fontWeight': '600'
                            })
                        ])
                    ])
                ], style={'flex': '1', 'marginRight': '30px'}),
                
                # Tail Risk Metrics
                html.Div([
                    html.H5("Tail Risk Analysis", style={
                        'color': COLORS['secondary'],
                        'marginBottom': '15px',
                        'fontSize': '16px'
                    }),
                    html.Div([
                        html.Div([
                            html.Span("VaR (95%): ", style={'fontWeight': '500'}),
                            html.Span(f"{var_95:.2f}%", style={
                                'color': COLORS['loss'],
                                'fontWeight': '600'
                            })
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("CVaR (95%): ", style={'fontWeight': '500'}),
                            html.Span(f"{cvar_95:.2f}%", style={
                                'color': COLORS['loss'],
                                'fontWeight': '600'
                            })
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("Skewness: ", style={'fontWeight': '500'}),
                            html.Span(f"{skewness:.2f}", style={
                                'color': COLORS['profit'] if skewness > 0 else COLORS['loss'],
                                'fontWeight': '600'
                            })
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("Kurtosis: ", style={'fontWeight': '500'}),
                            html.Span(f"{kurtosis:.2f}", style={
                                'color': COLORS['caution'] if abs(kurtosis) > 1 else COLORS['info'],
                                'fontWeight': '600'
                            })
                        ])
                    ])
                ], style={'flex': '1'})
                
            ], style={'display': 'flex', 'marginBottom': '20px'}),
            
            # Statistical Significance
            html.Div([
                html.H5("Statistical Significance", style={
                    'color': COLORS['secondary'],
                    'marginBottom': '15px',
                    'fontSize': '16px'
                }),
                self._create_significance_indicator(analytics_data)
            ])
            
        ])
        
        return create_base_card(content, 'full-width')
    
    def _create_significance_indicator(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create statistical significance indicator."""
        
        equity_curve = analytics_data.equity_curve
        if equity_curve is None or equity_curve.empty:
            return html.P("Insufficient data for statistical analysis", 
                         style={'color': COLORS['tertiary']})
        
        returns = equity_curve.pct_change().dropna()
        n_observations = len(returns)
        
        # Simple statistical significance indicators
        significance_tests = []
        
        # Sample size adequacy
        if n_observations >= 500:
            significance_tests.append(("Sample Size", "Strong", COLORS['profit']))
        elif n_observations >= 100:
            significance_tests.append(("Sample Size", "Adequate", COLORS['caution']))
        else:
            significance_tests.append(("Sample Size", "Limited", COLORS['loss']))
        
        # Sharpe ratio significance (rough approximation)
        sharpe = analytics_data.portfolio_metrics.get('Sharpe Ratio', 0)
        sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / n_observations)  # Approximate standard error
        sharpe_t_stat = sharpe / sharpe_se if sharpe_se > 0 else 0
        
        if abs(sharpe_t_stat) > 2.58:  # 99% confidence
            significance_tests.append(("Sharpe Significance", "High (99%)", COLORS['profit']))
        elif abs(sharpe_t_stat) > 1.96:  # 95% confidence
            significance_tests.append(("Sharpe Significance", "Medium (95%)", COLORS['caution']))
        else:
            significance_tests.append(("Sharpe Significance", "Low", COLORS['loss']))
        
        # Time series stability (coefficient of variation of rolling Sharpe)
        if n_observations >= 252:  # Need enough data for rolling analysis
            try:
                portfolio_metrics = PortfolioMetrics(equity_curve)
                rolling_data = portfolio_metrics.get_rolling_metrics(60)
                if 'Rolling Sharpe Ratio' in rolling_data:
                    rolling_sharpe = rolling_data['Rolling Sharpe Ratio'].dropna()
                    if len(rolling_sharpe) > 10:
                        sharpe_stability = rolling_sharpe.std() / abs(rolling_sharpe.mean()) if rolling_sharpe.mean() != 0 else float('inf')
                        if sharpe_stability < 0.5:
                            significance_tests.append(("Performance Stability", "High", COLORS['profit']))
                        elif sharpe_stability < 1.0:
                            significance_tests.append(("Performance Stability", "Medium", COLORS['caution']))
                        else:
                            significance_tests.append(("Performance Stability", "Low", COLORS['loss']))
            except Exception:
                significance_tests.append(("Performance Stability", "Unknown", COLORS['tertiary']))
        
        # Create indicator badges
        indicators = []
        for test_name, result, color in significance_tests:
            indicators.append(
                html.Div([
                    html.Span(f"{test_name}: ", style={'fontWeight': '500'}),
                    html.Span(result, style={
                        'backgroundColor': color,
                        'color': 'white',
                        'padding': '2px 8px',
                        'borderRadius': '12px',
                        'fontSize': '12px',
                        'fontWeight': '500'
                    })
                ], style={'marginBottom': '8px'})
            )
        
        return html.Div([
            html.P(f"Analysis based on {n_observations:,} observations", style={
                'color': COLORS['tertiary'],
                'fontSize': '12px',
                'marginBottom': '15px'
            }),
            html.Div(indicators)
        ])


def create_portfolio_overview_content(data_service: DashboardDataService, 
                                    strategy: str = None, version: str = None,
                                    start_date: datetime = None, end_date: datetime = None) -> html.Div:
    """
    Factory function to create Portfolio Overview tab content.
    
    Args:
        data_service: Dashboard data service instance
        strategy: Selected strategy name
        version: Selected version identifier
        start_date: Filter start date
        end_date: Filter end date
        
    Returns:
        Portfolio Overview tab content
    """
    overview_tab = PortfolioOverviewTab(data_service)
    return overview_tab.create_tab_content(strategy, version, start_date, end_date)