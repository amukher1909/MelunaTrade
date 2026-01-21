"""
Interactive Equity Curve Chart Component

This module implements the primary interactive equity curve chart as the centerpiece 
of the dashboard, featuring benchmark overlay, advanced interactivity, and professional 
presentation quality.

Features:
- Primary strategy equity curve with cumulative returns
- Benchmark comparison overlay (S&P 500, custom benchmark)
- Interactive features (zoom, pan, hover details)
- Drawdown periods highlighted
- Time range selector (1M, 3M, 6M, YTD, 1Y, All)
- Professional styling suitable for executive presentation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass

from dash import dcc, html, Input, Output, callback, State
from dash.exceptions import PreventUpdate

from .DashboardComponents import COLORS, CARD_SIZES, create_base_card
from .DashboardDataService import DashboardDataService, CachedAnalyticsData

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkData:
    """Container for benchmark data and metadata."""
    name: str
    data: pd.Series
    description: str
    color: str
    line_style: str = 'dash'


class BenchmarkDataService:
    """Service for generating and managing benchmark data."""
    
    def __init__(self):
        """Initialize benchmark data service."""
        self.available_benchmarks = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ-100 ETF', 
            'IWM': 'Russell 2000 ETF',
            'VTI': 'Total Stock Market ETF',
            'CUSTOM': 'Custom Benchmark'
        }
    
    def generate_synthetic_benchmark(self, start_date: datetime, end_date: datetime,
                                   benchmark_type: str = 'SPY',
                                   initial_value: float = 100000.0) -> BenchmarkData:
        """
        Generate synthetic benchmark data for demonstration purposes.
        
        In a production environment, this would connect to a real data provider
        like Yahoo Finance, Alpha Vantage, or Bloomberg API.
        
        Args:
            start_date: Start date for benchmark
            end_date: End date for benchmark
            benchmark_type: Type of benchmark ('SPY', 'QQQ', etc.)
            initial_value: Starting portfolio value
            
        Returns:
            BenchmarkData object with synthetic data
        """
        # Create date range matching strategy data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Benchmark characteristics (annualized)
        benchmark_params = {
            'SPY': {'return': 0.10, 'volatility': 0.16, 'color': COLORS['info']},
            'QQQ': {'return': 0.12, 'volatility': 0.22, 'color': COLORS['tertiary']},
            'IWM': {'return': 0.08, 'volatility': 0.24, 'color': COLORS['caution']},
            'VTI': {'return': 0.09, 'volatility': 0.17, 'color': COLORS['profit']},
            'CUSTOM': {'return': 0.07, 'volatility': 0.12, 'color': COLORS['secondary']}
        }
        
        params = benchmark_params.get(benchmark_type, benchmark_params['SPY'])
        
        # Generate synthetic returns using geometric Brownian motion
        np.random.seed(42)  # For consistent demo data
        dt = 1/252  # Daily time step
        n_periods = len(date_range)
        
        # Generate correlated returns with some realistic characteristics
        drift = (params['return'] - 0.5 * params['volatility']**2) * dt
        diffusion = params['volatility'] * np.sqrt(dt) * np.random.normal(0, 1, n_periods)
        
        # Add some market regime changes for realism
        regime_changes = np.random.choice([0, 1], size=n_periods, p=[0.95, 0.05])
        stress_factor = np.where(regime_changes, -0.03, 0)  # 3% stress on regime change days
        
        log_returns = drift + diffusion + stress_factor
        
        # Convert to price series
        prices = initial_value * np.exp(np.cumsum(log_returns))
        benchmark_series = pd.Series(prices, index=date_range)
        
        return BenchmarkData(
            name=self.available_benchmarks[benchmark_type],
            data=benchmark_series,
            description=f"Synthetic {self.available_benchmarks[benchmark_type]} data for demonstration",
            color=params['color'],
            line_style='dash'
        )
    
    def get_available_benchmarks(self) -> Dict[str, str]:
        """Get list of available benchmarks."""
        return self.available_benchmarks.copy()


class InteractiveEquityCurveChart:
    """
    Advanced interactive equity curve chart component for the dashboard.
    
    This component serves as the primary visualization in the Portfolio Overview tab,
    providing comprehensive equity curve analysis with benchmark comparison and
    interactive features.
    """
    
    def __init__(self, data_service: DashboardDataService):
        """
        Initialize the interactive equity curve chart.
        
        Args:
            data_service: Dashboard data service for loading analytics data
        """
        self.data_service = data_service
        self.benchmark_service = BenchmarkDataService()
        
        # Chart configuration
        self.default_height = 600
        self.time_range_options = [
            {'label': '1M', 'value': '1M'},
            {'label': '3M', 'value': '3M'},
            {'label': '6M', 'value': '6M'},
            {'label': 'YTD', 'value': 'YTD'},
            {'label': '1Y', 'value': '1Y'},
            {'label': 'All', 'value': 'All'}
        ]
        
        # Performance optimization settings
        self.max_points_display = 2000  # Limit for interactive performance
        self.decimation_factor = 0.1  # Factor for data decimation
        
        # Mobile responsiveness breakpoints
        self.mobile_breakpoint = 768
        self.tablet_breakpoint = 1024
    
    def create_equity_curve_card(self, analytics_data: CachedAnalyticsData = None,
                                selected_benchmark: str = 'SPY',
                                selected_time_range: str = 'All') -> html.Div:
        """
        Create the complete interactive equity curve card with controls.
        
        Args:
            analytics_data: Portfolio analytics data
            selected_benchmark: Selected benchmark identifier
            selected_time_range: Selected time range filter
            
        Returns:
            Complete card component with chart and controls
        """
        if analytics_data is None or analytics_data.equity_curve is None:
            return self._create_placeholder_card()
        
        # Create the main chart
        chart_figure = self._create_equity_curve_figure(
            analytics_data, selected_benchmark, selected_time_range
        )
        
        # Create chart controls
        controls = self._create_chart_controls(selected_benchmark, selected_time_range)
        
        # Create chart container
        content = html.Div([
            # Chart Header with Controls
            html.Div([
                html.Div([
                    html.H3("Portfolio Equity Curve", style={
                        'margin': '0',
                        'fontSize': '24px',
                        'fontWeight': '600',
                        'color': COLORS['primary']
                    }),
                    html.P("Interactive performance analysis with benchmark comparison", style={
                        'margin': '5px 0 0 0',
                        'fontSize': '14px',
                        'color': COLORS['tertiary']
                    })
                ], style={'flex': '1'}),
                
                # Chart action buttons
                html.Div([
                    html.Button([
                        html.I(className="fas fa-download", style={'marginRight': '8px'}),
                        "Export"
                    ], 
                    id='export-equity-curve-btn',
                    style={
                        'background': COLORS['info'],
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '6px',
                        'padding': '8px 16px',
                        'cursor': 'pointer',
                        'fontSize': '14px',
                        'marginRight': '10px'
                    }),
                    html.Button([
                        html.I(className="fas fa-expand-alt")
                    ], 
                    id='fullscreen-equity-curve-btn',
                    style={
                        'background': 'none',
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '6px',
                        'color': COLORS['tertiary'],
                        'cursor': 'pointer',
                        'padding': '8px 12px',
                        'fontSize': '14px'
                    }, 
                    title="Fullscreen")
                ])
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'flex-start',
                'marginBottom': '20px',
                'paddingBottom': '15px',
                'borderBottom': f'1px solid {COLORS["border"]}'
            }),
            
            # Chart Controls
            controls,
            
            # Main Chart
            dcc.Graph(
                id='interactive-equity-curve',
                figure=chart_figure,
                style={'height': f'{self.default_height}px'},
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['pan2d', 'lasso2d'],
                    'modeBarButtonsToRemove': ['autoScale2d', 'resetScale2d'],
                    'scrollZoom': True,
                    'doubleClick': 'reset+autosize'
                }
            ),
            
            # Chart Statistics Summary
            self._create_chart_statistics(analytics_data, selected_benchmark)
            
        ])
        
        return create_base_card(content, size='chart', card_type='chart')
    
    def _create_placeholder_card(self) -> html.Div:
        """Create placeholder card when no data is available."""
        content = html.Div([
            html.Div([
                html.H3("Portfolio Equity Curve", style={
                    'margin': '0',
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'color': COLORS['primary']
                })
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.I(className="fas fa-chart-line", style={
                    'fontSize': '64px',
                    'color': COLORS['border'],
                    'marginBottom': '20px'
                }),
                html.H4("No Data Available", style={
                    'color': COLORS['tertiary'],
                    'marginBottom': '10px'
                }),
                html.P("Select a strategy and version to view the interactive equity curve analysis.", 
                      style={'color': COLORS['tertiary'], 'textAlign': 'center'})
            ], style={
                'textAlign': 'center',
                'padding': '60px 20px',
                'color': COLORS['tertiary']
            })
        ])
        
        return create_base_card(content, size='chart', card_type='chart')
    
    def _create_chart_controls(self, selected_benchmark: str, 
                              selected_time_range: str) -> html.Div:
        """Create chart control panel with time range and benchmark selectors."""
        return html.Div([
            # Time Range Selector
            html.Div([
                html.Label("Time Range:", style={
                    'fontWeight': '500',
                    'color': COLORS['primary'],
                    'marginBottom': '8px',
                    'display': 'block',
                    'fontSize': '14px'
                }),
                html.Div([
                    html.Button(
                        option['label'],
                        id={'type': 'time-range-btn', 'index': option['value']},
                        style={
                            'background': COLORS['info'] if option['value'] == selected_time_range else 'white',
                            'color': 'white' if option['value'] == selected_time_range else COLORS['primary'],
                            'border': f'1px solid {COLORS["info"]}',
                            'borderRadius': '4px',
                            'padding': '6px 12px',
                            'margin': '0 4px 0 0',
                            'cursor': 'pointer',
                            'fontSize': '12px',
                            'fontWeight': '500'
                        }
                    ) for option in self.time_range_options
                ], style={'display': 'flex', 'flexWrap': 'wrap'})
            ], style={'flex': '1', 'marginRight': '30px'}),
            
            # Benchmark Selector
            html.Div([
                html.Label("Benchmark:", style={
                    'fontWeight': '500',
                    'color': COLORS['primary'],
                    'marginBottom': '8px',
                    'display': 'block',
                    'fontSize': '14px'
                }),
                dcc.Dropdown(
                    id='benchmark-selector',
                    options=[
                        {'label': name, 'value': code} 
                        for code, name in self.benchmark_service.get_available_benchmarks().items()
                    ],
                    value=selected_benchmark,
                    style={'minWidth': '200px'},
                    clearable=False
                )
            ], style={'flex': '0 0 auto'})
            
        ], style={
            'display': 'flex',
            'alignItems': 'flex-end',
            'marginBottom': '25px',
            'padding': '15px',
            'backgroundColor': COLORS['page_bg'],
            'borderRadius': '6px',
            'border': f'1px solid {COLORS["border"]}',
            'flexDirection': 'row',
            '@media (max-width: 768px)': {
                'flexDirection': 'column',
                'alignItems': 'stretch'
            }
        })
    
    def _create_equity_curve_figure(self, analytics_data: CachedAnalyticsData,
                                   benchmark_type: str = 'SPY',
                                   time_range: str = 'All') -> go.Figure:
        """
        Create the main interactive equity curve figure with benchmark overlay.
        
        Args:
            analytics_data: Portfolio analytics data
            benchmark_type: Type of benchmark to overlay
            time_range: Time range filter to apply
            
        Returns:
            Plotly figure object
        """
        equity_curve = analytics_data.equity_curve.copy()
        
        if equity_curve is None or equity_curve.empty:
            return self._create_empty_figure()
        
        # Apply time range filter
        filtered_equity = self._apply_time_range_filter(equity_curve, time_range)
        
        if filtered_equity.empty:
            return self._create_empty_figure()
        
        # Apply performance optimization for large datasets
        filtered_equity = self._optimize_data_for_performance(filtered_equity)
        
        # Create figure with secondary y-axis for drawdown
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Portfolio Value & Benchmark Comparison', 'Drawdown Analysis'),
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Add portfolio equity curve
        fig.add_trace(
            go.Scatter(
                x=filtered_equity.index,
                y=filtered_equity.values,
                mode='lines',
                name='Portfolio',
                line=dict(color=COLORS['info'], width=3),
                hovertemplate=(
                    '<b>Portfolio Value</b><br>' +
                    'Date: %{x}<br>' +
                    'Value: ₹%{y:,.0f}<br>' +
                    'Return: %{customdata:.2f}%<br>' +
                    '<extra></extra>'
                ),
                customdata=self._calculate_cumulative_returns(filtered_equity)
            ),
            row=1, col=1
        )
        
        # Generate and add benchmark data
        benchmark_data = self.benchmark_service.generate_synthetic_benchmark(
            start_date=filtered_equity.index[0],
            end_date=filtered_equity.index[-1],
            benchmark_type=benchmark_type,
            initial_value=filtered_equity.iloc[0]
        )
        
        fig.add_trace(
            go.Scatter(
                x=benchmark_data.data.index,
                y=benchmark_data.data.values,
                mode='lines',
                name=benchmark_data.name,
                line=dict(color=benchmark_data.color, width=2, dash=benchmark_data.line_style),
                hovertemplate=(
                    f'<b>{benchmark_data.name}</b><br>' +
                    'Date: %{x}<br>' +
                    'Value: ₹%{y:,.0f}<br>' +
                    'Return: %{customdata:.2f}%<br>' +
                    '<extra></extra>'
                ),
                customdata=self._calculate_cumulative_returns(benchmark_data.data)
            ),
            row=1, col=1
        )
        
        # Add drawdown visualization
        drawdown_data = self._calculate_drawdown(filtered_equity)
        
        # Highlight major drawdown periods
        drawdown_periods = self._identify_drawdown_periods(drawdown_data, threshold=-0.05)
        
        # Add drawdown area chart
        fig.add_trace(
            go.Scatter(
                x=drawdown_data.index,
                y=drawdown_data.values * 100,  # Convert to percentage
                mode='lines',
                name='Drawdown',
                line=dict(color=COLORS['loss'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.3)',
                hovertemplate=(
                    '<b>Drawdown</b><br>' +
                    'Date: %{x}<br>' +
                    'Drawdown: %{y:.2f}%<br>' +
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
        
        # Add drawdown period annotations
        for period in drawdown_periods:
            fig.add_vrect(
                x0=period['start'], x1=period['end'],
                fillcolor="rgba(255, 68, 68, 0.1)",
                layer="below", line_width=0,
                row=1, col=1
            )
        
        # Get responsive layout configuration
        responsive_config = self._create_responsive_layout_config()
        
        # Update layout with responsive design
        fig.update_layout(
            height=self.default_height,
            font=responsive_config['font']['desktop'],
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=COLORS['border'],
                borderwidth=1
            ),
            margin=responsive_config['margin']['desktop'],
            showlegend=True,
            autosize=True
        )
        
        # Update x-axes
        fig.update_xaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False,
            showspikes=True,
            spikethickness=1,
            spikecolor=COLORS['tertiary'],
            spikesnap="cursor",
            spikemode="across"
        )
        
        # Update y-axes for main chart
        fig.update_yaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False,
            tickformat='₹,.0f',
            title_text="Portfolio Value (₹)",
            row=1, col=1
        )
        
        # Update y-axes for drawdown chart
        fig.update_yaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=True,
            zerolinecolor=COLORS['border'],
            zerolinewidth=2,
            tickformat='.1f',
            title_text="Drawdown (%)",
            range=[min(drawdown_data.values * 100) * 1.1, 1],
            row=2, col=1
        )
        
        return fig
    
    def _apply_time_range_filter(self, data: pd.Series, time_range: str) -> pd.Series:
        """Apply time range filter to data series."""
        if time_range == 'All':
            return data
        
        end_date = data.index[-1]
        
        if time_range == '1M':
            start_date = end_date - timedelta(days=30)
        elif time_range == '3M':
            start_date = end_date - timedelta(days=90)
        elif time_range == '6M':
            start_date = end_date - timedelta(days=180)
        elif time_range == '1Y':
            start_date = end_date - timedelta(days=365)
        elif time_range == 'YTD':
            start_date = datetime(end_date.year, 1, 1)
        else:
            return data
        
        return data[data.index >= start_date]
    
    def _calculate_cumulative_returns(self, data: pd.Series) -> np.ndarray:
        """Calculate cumulative returns as percentage."""
        if data.empty:
            return np.array([])
        
        initial_value = data.iloc[0]
        return ((data / initial_value) - 1) * 100
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        if equity_curve.empty:
            return pd.Series()
        
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def _identify_drawdown_periods(self, drawdown: pd.Series, 
                                  threshold: float = -0.05) -> List[Dict[str, Any]]:
        """Identify significant drawdown periods."""
        if drawdown.empty:
            return []
        
        periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd_value in drawdown.items():
            if dd_value <= threshold and not in_drawdown:
                # Start of drawdown period
                in_drawdown = True
                start_date = date
            elif dd_value > threshold and in_drawdown:
                # End of drawdown period
                in_drawdown = False
                if start_date:
                    periods.append({
                        'start': start_date,
                        'end': date,
                        'max_drawdown': drawdown.loc[start_date:date].min()
                    })
                start_date = None
        
        # Handle case where we end in drawdown
        if in_drawdown and start_date:
            periods.append({
                'start': start_date,
                'end': drawdown.index[-1],
                'max_drawdown': drawdown.loc[start_date:].min()
            })
        
        return periods
    
    def _optimize_data_for_performance(self, data: pd.Series) -> pd.Series:
        """
        Optimize data for performance by applying decimation for large datasets.
        
        Args:
            data: Input data series
            
        Returns:
            Optimized data series
        """
        if len(data) <= self.max_points_display:
            return data
        
        # Calculate decimation step to reduce data points to target size
        target_size = self.max_points_display - 1  # Reserve one for the last point
        step = max(1, len(data) // target_size)
        
        # Create optimized data by sampling at regular intervals
        indices = list(range(0, len(data), step))
        
        # Always include the last index if not already included
        if indices[-1] != len(data) - 1:
            indices.append(len(data) - 1)
        
        # Limit to max_points_display to ensure we don't exceed the limit
        if len(indices) > self.max_points_display:
            indices = indices[:self.max_points_display-1] + [indices[-1]]
        
        optimized_data = data.iloc[indices].copy()
        
        logger.info(f"Optimized data from {len(data)} to {len(optimized_data)} points for performance")
        return optimized_data
    
    def _create_responsive_layout_config(self) -> Dict[str, Any]:
        """
        Create responsive layout configuration for mobile and tablet devices.
        
        Returns:
            Layout configuration dictionary
        """
        return {
            'margin': {
                'mobile': dict(l=40, r=20, t=60, b=40),
                'tablet': dict(l=50, r=20, t=70, b=40),
                'desktop': dict(l=60, r=20, t=80, b=40)
            },
            'font': {
                'mobile': dict(family='Inter, sans-serif', size=10),
                'tablet': dict(family='Inter, sans-serif', size=11),
                'desktop': dict(family='Inter, sans-serif', size=12)
            },
            'legend': {
                'mobile': dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01),
                'tablet': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                'desktop': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            }
        }
    
    def _create_empty_figure(self) -> go.Figure:
        """Create empty figure placeholder."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected time range",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS['tertiary'])
        )
        fig.update_layout(
            height=self.default_height,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif')
        )
        return fig
    
    def _create_chart_statistics(self, analytics_data: CachedAnalyticsData,
                                benchmark_type: str) -> html.Div:
        """Create summary statistics panel below the chart."""
        if not analytics_data or not analytics_data.portfolio_metrics:
            return html.Div()
        
        metrics = analytics_data.portfolio_metrics
        
        # Calculate benchmark statistics for comparison
        equity_curve = analytics_data.equity_curve
        if equity_curve is not None and not equity_curve.empty:
            benchmark_data = self.benchmark_service.generate_synthetic_benchmark(
                start_date=equity_curve.index[0],
                end_date=equity_curve.index[-1],
                benchmark_type=benchmark_type,
                initial_value=equity_curve.iloc[0]
            )
            
            benchmark_return = ((benchmark_data.data.iloc[-1] / benchmark_data.data.iloc[0]) - 1) * 100
            portfolio_return = metrics.get('Total Return (%)', 0)
            excess_return = portfolio_return - benchmark_return
        else:
            benchmark_return = 0
            excess_return = 0
        
        return html.Div([
            html.H5("Performance Summary", style={
                'color': COLORS['primary'],
                'marginBottom': '15px',
                'fontSize': '16px',
                'fontWeight': '600'
            }),
            
            html.Div([
                # Portfolio metrics
                html.Div([
                    html.Div([
                        html.Span("Total Return:", style={'fontWeight': '500'}),
                        html.Span(f"{metrics.get('Total Return (%)', 0):.2f}%", style={
                            'marginLeft': '8px',
                            'color': COLORS['profit'] if metrics.get('Total Return (%)', 0) > 0 else COLORS['loss'],
                            'fontWeight': '600'
                        })
                    ], style={'marginBottom': '5px'}),
                    
                    html.Div([
                        html.Span("Sharpe Ratio:", style={'fontWeight': '500'}),
                        html.Span(f"{metrics.get('Sharpe Ratio', 0):.2f}", style={
                            'marginLeft': '8px',
                            'color': COLORS['info'],
                            'fontWeight': '600'
                        })
                    ], style={'marginBottom': '5px'}),
                    
                    html.Div([
                        html.Span("Max Drawdown:", style={'fontWeight': '500'}),
                        html.Span(f"{metrics.get('Max Drawdown (%)', 0):.2f}%", style={
                            'marginLeft': '8px',
                            'color': COLORS['loss'],
                            'fontWeight': '600'
                        })
                    ])
                ], style={'flex': '1'}),
                
                # Benchmark comparison
                html.Div([
                    html.Div([
                        html.Span("Benchmark Return:", style={'fontWeight': '500'}),
                        html.Span(f"{benchmark_return:.2f}%", style={
                            'marginLeft': '8px',
                            'color': COLORS['tertiary'],
                            'fontWeight': '600'
                        })
                    ], style={'marginBottom': '5px'}),
                    
                    html.Div([
                        html.Span("Excess Return:", style={'fontWeight': '500'}),
                        html.Span(f"{excess_return:+.2f}%", style={
                            'marginLeft': '8px',
                            'color': COLORS['profit'] if excess_return > 0 else COLORS['loss'],
                            'fontWeight': '600'
                        })
                    ], style={'marginBottom': '5px'}),
                    
                    html.Div([
                        html.Span("Volatility:", style={'fontWeight': '500'}),
                        html.Span(f"{metrics.get('Annualized Volatility (%)', 0):.2f}%", style={
                            'marginLeft': '8px',
                            'color': COLORS['caution'],
                            'fontWeight': '600'
                        })
                    ])
                ], style={'flex': '1'})
                
            ], style={'display': 'flex', 'gap': '40px'})
            
        ], style={
            'marginTop': '20px',
            'padding': '20px',
            'backgroundColor': COLORS['page_bg'],
            'borderRadius': '6px',
            'border': f'1px solid {COLORS["border"]}'
        })


def create_interactive_equity_curve_component(data_service: DashboardDataService,
                                            analytics_data: CachedAnalyticsData = None,
                                            selected_benchmark: str = 'SPY',
                                            selected_time_range: str = 'All') -> html.Div:
    """
    Factory function to create the interactive equity curve component.
    
    Args:
        data_service: Dashboard data service instance
        analytics_data: Portfolio analytics data
        selected_benchmark: Selected benchmark identifier
        selected_time_range: Selected time range filter
        
    Returns:
        Interactive equity curve card component
    """
    chart = InteractiveEquityCurveChart(data_service)
    return chart.create_equity_curve_card(analytics_data, selected_benchmark, selected_time_range)