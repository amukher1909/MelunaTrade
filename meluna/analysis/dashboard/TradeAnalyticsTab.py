"""
Trade Analytics Tab - MFE/MAE Analysis and Trade-Level Performance

This module implements the Trade Analytics tab with comprehensive trade-level analysis
including MFE/MAE visualization, profit factor breakdown, and alpha decay patterns.

Features:
- Interactive MFE/MAE scatter plot analysis with trade selection
- Trade duration vs profitability categorization charts
- Profit factor breakdown by trade characteristics
- Alpha decay analysis showing returns degradation over time
- Trade timing efficiency metrics and visualizations
- Interactive trade tables with filtering and sorting capabilities
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

from dash import dcc, html, Input, Output, callback, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from .DashboardComponents import (
    create_kpi_card, create_chart_container, create_card_grid, 
    create_card_group, create_base_card, COLORS, CARD_SIZES
)
from .DashboardDataService import DashboardDataService, CachedAnalyticsData
from ..metrics.TradeAnalyzer import TradeAnalyzer, TradeMetrics

logger = logging.getLogger(__name__)


class TradeAnalyticsTab:
    """
    Trade Analytics tab component providing comprehensive trade-level analysis.
    
    Integrates with TradeAnalyzer module to display detailed trade performance,
    MFE/MAE analysis, and efficiency metrics through interactive charts and tables.
    """
    
    def __init__(self, data_service: DashboardDataService):
        """
        Initialize Trade Analytics tab.
        
        Args:
            data_service: Dashboard data service for loading analytics data
        """
        self.data_service = data_service
        
    def create_tab_content(self, strategy: str = None, version: str = None, 
                          start_date: datetime = None, end_date: datetime = None) -> html.Div:
        """
        Create Trade Analytics tab content with real data integration.
        
        Args:
            strategy: Selected strategy name
            version: Selected version identifier
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            Dash HTML Div containing the complete Trade Analytics tab
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
        if analytics_data and analytics_data.trade_log is not None and not analytics_data.trade_log.empty:
            return self._create_content_with_data(analytics_data)
        else:
            return self._create_placeholder_content()
    
    def _create_content_with_data(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create tab content with real trade data."""
        
        # Initialize TradeAnalyzer with trade data
        try:
            trade_analyzer = TradeAnalyzer(
                trade_log=analytics_data.trade_log,
                price_data=getattr(analytics_data, 'price_data', None)
            )
            
            # Calculate comprehensive trade metrics
            trade_metrics = trade_analyzer.calculate_all_metrics()
            
        except Exception as e:
            logger.error(f"Error initializing TradeAnalyzer: {e}")
            return self._create_error_content(str(e))
        
        # Create KPI cards with trade metrics
        kpi_cards = self._create_trade_kpi_cards(trade_metrics, trade_analyzer)
        
        # Create charts and visualizations
        charts = self._create_trade_charts(trade_analyzer, analytics_data)
        
        # Create interactive trade table
        trade_table = self._create_interactive_trade_table(trade_analyzer)
        
        return html.Div([
            # Tab Header
            html.Div([
                html.H2("Trade Analytics", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P("Detailed trade-level performance and efficiency analysis with MFE/MAE insights", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '20px'}),
            
            # Trade Performance KPIs
            create_card_group("Trade Performance Metrics", kpi_cards[:4]),
            
            # MFE/MAE Analysis (Primary Feature)
            html.Div([
                charts['mfe_mae_scatter']
            ], style={'marginBottom': '30px'}),
            
            # Duration Analysis - Professional Side-by-Side Layout
            html.Div([
                html.Div([
                    html.H3("Trade Duration Analysis", style={
                        'margin': '0',
                        'fontSize': '20px',
                        'fontWeight': '600',
                        'color': COLORS['primary'],
                        'marginBottom': '15px'
                    }),
                    html.P("Dual-plot analysis showing duration vs profitability patterns", style={
                        'color': COLORS['tertiary'],
                        'fontSize': '14px',
                        'marginBottom': '25px',
                        'fontStyle': 'italic'
                    })
                ], className='duration-analysis-header'),
                html.Div([
                    charts['duration_pnl_chart'],
                    charts['duration_count_chart']
                ], className='duration-analysis-grid')
            ], className='duration-analysis-container'),
            
            # Profit Factor Breakdown Analysis
            html.Div([
                charts['profit_factor_breakdown']
            ], style={'marginBottom': '30px'}),
            
            # Alpha Decay and Timing Analysis
            html.Div([
                create_card_grid([
                    charts['alpha_decay'],
                    charts['timing_efficiency']
                ])
            ], style={'marginBottom': '30px'}),
            
            # Interactive Trade Table
            create_card_group("Individual Trade Details", [trade_table]),
            
            # Advanced MFE/MAE Metrics
            create_card_group("MFE/MAE Efficiency Metrics", kpi_cards[4:])
            
        ], style={
            'padding': '20px',
            'backgroundColor': COLORS['page_bg'],
            'minHeight': 'calc(100vh - 90px)'
        })
    
    def _create_trade_kpi_cards(self, trade_metrics: TradeMetrics, trade_analyzer: TradeAnalyzer) -> List[html.Div]:
        """Create KPI cards for trade-level metrics."""
        
        # Get additional metrics from analyzer
        core_metrics = trade_analyzer.metrics
        
        cards = [
            # Primary Trade Metrics
            create_kpi_card(
                title="Win Rate",
                value=f"{(core_metrics.get('win_count', 0) / max(1, core_metrics.get('total_trades', 1))) * 100:.1f}%",
                subtitle=f"{core_metrics.get('win_count', 0)} of {core_metrics.get('total_trades', 0)} trades",
                color=COLORS['profit'] if (core_metrics.get('win_count', 0) / max(1, core_metrics.get('total_trades', 1))) > 0.5 else COLORS['caution'],
                icon="fas fa-trophy",
                trend="up" if (core_metrics.get('win_count', 0) / max(1, core_metrics.get('total_trades', 1))) > 0.5 else "neutral",
                size="small"
            ),
            create_kpi_card(
                title="Profit Factor",
                value=f"{trade_metrics.profit_factor:.2f}",
                subtitle="Gross profit / Gross loss",
                color=COLORS['profit'] if trade_metrics.profit_factor > 1.0 else COLORS['loss'],
                icon="fas fa-calculator",
                trend="up" if trade_metrics.profit_factor > 1.0 else "down",
                size="small"
            ),
            create_kpi_card(
                title="Expectancy",
                value=f"₹{trade_metrics.expectancy:.2f}",
                subtitle="Average expected P/L per trade",
                color=COLORS['profit'] if trade_metrics.expectancy > 0 else COLORS['loss'],
                icon="fas fa-chart-line",
                trend="up" if trade_metrics.expectancy > 0 else "down",
                size="small"
            ),
            create_kpi_card(
                title="Payoff Ratio",
                value=f"{trade_metrics.payoff_ratio:.2f}",
                subtitle="Average win / Average loss",
                color=COLORS['info'] if trade_metrics.payoff_ratio > 1.0 else COLORS['caution'],
                icon="fas fa-balance-scale",
                trend="up" if trade_metrics.payoff_ratio > 1.0 else "neutral",
                size="small"
            ),
            
            # MFE/MAE Efficiency Metrics
            create_kpi_card(
                title="MFE Efficiency",
                value=f"{trade_metrics.mfe_efficiency * 100:.1f}%",
                subtitle="Favorable excursion captured",
                color=COLORS['profit'] if trade_metrics.mfe_efficiency > 0.7 else COLORS['caution'],
                icon="fas fa-arrow-up",
                trend="up" if trade_metrics.mfe_efficiency > 0.7 else "neutral",
                size="small"
            ),
            create_kpi_card(
                title="MAE Control",
                value=f"{abs(trade_metrics.mae_efficiency) * 100:.1f}%",
                subtitle="Adverse excursion minimized",
                color=COLORS['profit'] if abs(trade_metrics.mae_efficiency) > 0.7 else COLORS['caution'],
                icon="fas fa-shield-alt",
                trend="up" if abs(trade_metrics.mae_efficiency) > 0.7 else "neutral",
                size="small"
            ),
            create_kpi_card(
                title="Avg MFE",
                value=f"₹{trade_metrics.avg_mfe:.2f}",
                subtitle="Maximum favorable excursion",
                color=COLORS['info'],
                icon="fas fa-arrow-circle-up",
                size="small"
            ),
            create_kpi_card(
                title="Avg MAE",
                value=f"₹{trade_metrics.avg_mae:.2f}",
                subtitle="Maximum adverse excursion",
                color=COLORS['loss'],
                icon="fas fa-arrow-circle-down",
                size="small"
            )
        ]
        
        return cards
    
    def _create_trade_charts(self, trade_analyzer: TradeAnalyzer, analytics_data: CachedAnalyticsData) -> Dict[str, html.Div]:
        """Create comprehensive trade analysis charts."""
        
        charts = {}
        
        # MFE/MAE Scatter Plot
        charts['mfe_mae_scatter'] = self._create_mfe_mae_scatter_chart(trade_analyzer)
        
        # Trade Duration Analysis - Split into side-by-side charts
        charts['duration_pnl_chart'] = self._create_duration_pnl_chart(trade_analyzer)
        charts['duration_count_chart'] = self._create_duration_count_chart(trade_analyzer)
        
        # Profit Factor Breakdown
        charts['profit_factor_breakdown'] = self._create_profit_factor_breakdown_chart(trade_analyzer)
        
        # Alpha Decay Analysis
        charts['alpha_decay'] = self._create_alpha_decay_chart(trade_analyzer)
        
        # Trade Timing Efficiency
        charts['timing_efficiency'] = self._create_timing_efficiency_chart(trade_analyzer)
        
        return charts
    
    def _create_mfe_mae_scatter_chart(self, trade_analyzer: TradeAnalyzer) -> html.Div:
        """Create interactive MFE/MAE scatter plot."""
        
        # Get trade data with MFE/MAE
        trade_data = trade_analyzer.trade_log.copy()
        
        # Ensure MFE/MAE data is available
        if 'mfe' not in trade_data.columns or 'mae' not in trade_data.columns:
            trade_analyzer.calculate_mfe_mae_metrics()
            trade_data = trade_analyzer.trade_log.copy()
        
        # Create scatter plot
        fig = go.Figure()
        
        # Winning trades
        winners = trade_data[trade_data['pnl'] > 0]
        if not winners.empty:
            fig.add_trace(go.Scatter(
                x=winners['mfe'],
                y=winners['mae'],
                mode='markers',
                name='Winning Trades',
                marker=dict(
                    color=COLORS['profit'],
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[f"Trade: {row['symbol']}<br>Entry: {row['entry_timestamp']}<br>P/L: ₹{row['pnl']:.2f}<br>Duration: {row['duration_days']:.1f}d<br>MFE: ₹{row['mfe']:.2f}<br>MAE: ₹{row['mae']:.2f}" 
                      for _, row in winners.iterrows()],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Losing trades
        losers = trade_data[trade_data['pnl'] <= 0]
        if not losers.empty:
            fig.add_trace(go.Scatter(
                x=losers['mfe'],
                y=losers['mae'],
                mode='markers',
                name='Losing Trades',
                marker=dict(
                    color=COLORS['loss'],
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[f"Trade: {row['symbol']}<br>Entry: {row['entry_timestamp']}<br>P/L: ₹{row['pnl']:.2f}<br>Duration: {row['duration_days']:.1f}d<br>MFE: ₹{row['mfe']:.2f}<br>MAE: ₹{row['mae']:.2f}" 
                      for _, row in losers.iterrows()],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Add quadrant lines
        x_range = [trade_data['mfe'].min(), trade_data['mfe'].max()]
        y_range = [trade_data['mae'].min(), trade_data['mae'].max()]
        
        # Efficiency reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            xaxis_title="Maximum Favorable Excursion (MFE) ₹",
            yaxis_title="Maximum Adverse Excursion (MAE) ₹",
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
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
        
        return create_base_card(
            html.Div([
                html.H3("MFE/MAE Analysis", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '500px'}
                )
            ]),
            'chart',
            'chart'
        )
    
    def _create_duration_pnl_chart(self, trade_analyzer: TradeAnalyzer) -> html.Div:
        """Create duration vs P/L analysis chart with native Dash styling."""
        
        # Get duration analysis data
        duration_analysis = trade_analyzer.detailed_analysis.get('duration_analysis')
        
        if duration_analysis is None:
            return create_base_card(
                html.Div([
                    html.H4("Average P/L by Holding Period", style={
                        'textAlign': 'center',
                        'color': COLORS['primary'],
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'marginBottom': '20px'
                    }),
                    html.Div("No P/L duration data available", style={
                        'textAlign': 'center',
                        'color': COLORS['tertiary'],
                        'fontSize': '14px',
                        'fontStyle': 'italic',
                        'padding': '40px'
                    })
                ], style={
                    'backgroundColor': '#FFFFFF',
                    'padding': '20px',
                    'borderRadius': '8px'
                }),
                'large',
                'default'
            )
        
        # Create individual P/L chart with native Dash styling
        duration_bins = duration_analysis.index.tolist()
        avg_pnl = duration_analysis['avg_pnl'].tolist()
        
        # Color code by profitability
        colors = [COLORS['profit'] if pnl > 0 else COLORS['loss'] for pnl in avg_pnl]
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=duration_bins,
                y=avg_pnl,
                marker=dict(
                    color=colors,
                    line=dict(width=0)  # Remove border
                ),
                text=[f'₹{pnl:.2f}' for pnl in avg_pnl],
                textposition='outside',
                textfont=dict(size=11, color=COLORS['primary']),
                hovertemplate='<b>%{x}</b><br>Avg P/L: ₹%{y:.2f}<extra></extra>',
                showlegend=False
            )
        )
        
        fig.update_layout(
            xaxis=dict(
                title="Holding Period",
                titlefont=dict(size=12, color=COLORS['tertiary']),
                tickfont=dict(size=11, color=COLORS['primary']),
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                zeroline=False,
                showgrid=True
            ),
            yaxis=dict(
                title="Average P/L (₹)",
                titlefont=dict(size=12, color=COLORS['tertiary']),
                tickfont=dict(size=11, color=COLORS['primary']),
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.3)',
                showgrid=True
            ),
            plot_bgcolor='#FFFFFF',  # White background
            paper_bgcolor='#FFFFFF',  # White background
            font=dict(family="Inter, sans-serif", size=11),
            margin=dict(l=60, r=30, t=80, b=60),
            height=450,
            showlegend=False
        )
        
        return create_base_card(
            html.Div([
                dcc.Graph(
                    figure=fig,
                    style={
                        'height': '450px',
                        'backgroundColor': '#FFFFFF'
                    },
                    config={
                        'displayModeBar': False,  # Hide plotly toolbar
                        'staticPlot': False
                    }
                )
            ], style={
                'backgroundColor': '#FFFFFF',
                'padding': '15px',
                'borderRadius': '8px',
                'border': '1px solid #E9ECEF'
            }),
            'large',
            'default'
        )
    
    def _create_duration_count_chart(self, trade_analyzer: TradeAnalyzer) -> html.Div:
        """Create duration vs trade count chart with native Dash styling."""
        
        # Get duration analysis data
        duration_analysis = trade_analyzer.detailed_analysis.get('duration_analysis')
        
        if duration_analysis is None:
            return create_base_card(
                html.Div([
                    html.H4("Trade Count by Duration", style={
                        'textAlign': 'center',
                        'color': COLORS['primary'],
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'marginBottom': '20px'
                    }),
                    html.Div("No trade count data available", style={
                        'textAlign': 'center',
                        'color': COLORS['tertiary'],
                        'fontSize': '14px',
                        'fontStyle': 'italic',
                        'padding': '40px'
                    })
                ], style={
                    'backgroundColor': '#FFFFFF',
                    'padding': '20px',
                    'borderRadius': '8px'
                }),
                'large',
                'default'
            )
        
        # Create individual trade count chart with native Dash styling
        duration_bins = duration_analysis.index.tolist()
        trade_counts = duration_analysis['trade_count'].tolist()
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=duration_bins,
                y=trade_counts,
                marker=dict(
                    color=COLORS['info'],
                    line=dict(width=0)  # Remove border
                ),
                text=trade_counts,
                textposition='outside',
                textfont=dict(size=11, color=COLORS['primary']),
                hovertemplate='<b>%{x}</b><br>Trade Count: %{y}<extra></extra>',
                showlegend=False
            )
        )
        
        fig.update_layout(
            xaxis=dict(
                title="Holding Period",
                titlefont=dict(size=12, color=COLORS['tertiary']),
                tickfont=dict(size=11, color=COLORS['primary']),
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                zeroline=False,
                showgrid=True
            ),
            yaxis=dict(
                title="Number of Trades",
                titlefont=dict(size=12, color=COLORS['tertiary']),
                tickfont=dict(size=11, color=COLORS['primary']),
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.3)',
                showgrid=True
            ),
            plot_bgcolor='#FFFFFF',  # White background
            paper_bgcolor='#FFFFFF',  # White background
            font=dict(family="Inter, sans-serif", size=11),
            margin=dict(l=60, r=30, t=80, b=60),
            height=450,
            showlegend=False
        )
        
        return create_base_card(
            html.Div([
                dcc.Graph(
                    figure=fig,
                    style={
                        'height': '450px',
                        'backgroundColor': '#FFFFFF'
                    },
                    config={
                        'displayModeBar': False,  # Hide plotly toolbar
                        'staticPlot': False
                    }
                )
            ], style={
                'backgroundColor': '#FFFFFF',
                'padding': '15px',
                'borderRadius': '8px',
                'border': '1px solid #E9ECEF'
            }),
            'large',
            'default'
        )
    
    def _create_profit_factor_breakdown_chart(self, trade_analyzer: TradeAnalyzer) -> html.Div:
        """Create profit factor breakdown by trade characteristics."""
        
        # Get categorized trades
        categorized_trades = trade_analyzer.categorize_trades()
        
        # Calculate profit factor by category
        pf_by_size = categorized_trades.groupby('size_category', observed=True)['pnl'].agg(['sum']).reset_index()
        pf_by_size['gross_profit'] = categorized_trades[categorized_trades['pnl'] > 0].groupby('size_category', observed=True)['pnl'].sum().reindex(pf_by_size['size_category']).fillna(0).values
        pf_by_size['gross_loss'] = abs(categorized_trades[categorized_trades['pnl'] < 0].groupby('size_category', observed=True)['pnl'].sum().reindex(pf_by_size['size_category']).fillna(0).values)
        pf_by_size['profit_factor'] = pf_by_size['gross_profit'] / pf_by_size['gross_loss'].replace(0, 1)
        
        # Create pie chart for profit contribution by size
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=['Profit Contribution by Trade Size', 'Profit Factor by Size Category']
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=pf_by_size['size_category'],
                values=pf_by_size['gross_profit'],
                name="Profit Contribution",
                marker_colors=[COLORS['profit'], COLORS['info'], COLORS['secondary']]
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=pf_by_size['size_category'],
                y=pf_by_size['profit_factor'],
                name="Profit Factor",
                marker_color=[COLORS['profit'] if pf >= 1.0 else COLORS['loss'] for pf in pf_by_size['profit_factor']],
                text=[f'{pf:.2f}' for pf in pf_by_size['profit_factor']],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        return create_base_card(
            html.Div([
                html.H3("Profit Factor Analysis", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '400px'}
                )
            ]),
            'large',
            'chart'
        )
    
    def _create_alpha_decay_chart(self, trade_analyzer: TradeAnalyzer) -> html.Div:
        """Create alpha decay analysis chart."""
        
        # Get alpha decay metrics
        alpha_metrics = trade_analyzer.detect_alpha_decay()
        trade_data = trade_analyzer.trade_log.copy()
        
        # Calculate daily returns if not available
        if 'daily_return' not in trade_data.columns:
            trade_data['daily_return'] = trade_data['pnl'] / trade_data['duration_days']
        
        # Create rolling window analysis
        trade_data_sorted = trade_data.sort_values('duration_days')
        
        fig = go.Figure()
        
        # Scatter plot of daily returns vs duration
        fig.add_trace(go.Scatter(
            x=trade_data_sorted['duration_days'],
            y=trade_data_sorted['daily_return'],
            mode='markers',
            name='Daily Returns',
            marker=dict(
                color=COLORS['info'],
                size=6,
                opacity=0.6
            ),
            text=[f"Trade: {row['symbol']}<br>Duration: {row['duration_days']:.1f}d<br>Daily Return: ₹{row['daily_return']:.2f}" 
                  for _, row in trade_data_sorted.iterrows()],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        # Add trend line if alpha decay coefficient is available
        if not np.isnan(alpha_metrics.get('alpha_decay_coefficient', np.nan)):
            x_trend = np.linspace(trade_data['duration_days'].min(), trade_data['duration_days'].max(), 100)
            # Simple polynomial fit for visualization
            try:
                coeffs = np.polyfit(trade_data_sorted['duration_days'], trade_data_sorted['daily_return'], deg=2)
                y_trend = np.polyval(coeffs, x_trend)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name='Alpha Decay Trend',
                    line=dict(color=COLORS['loss'], width=2, dash='dash')
                ))
            except:
                pass
        
        # Mark optimal holding period if available
        optimal_period = alpha_metrics.get('optimal_holding_period')
        if optimal_period and not np.isnan(optimal_period):
            fig.add_vline(
                x=optimal_period,
                line_dash="dot",
                line_color=COLORS['profit'],
                annotation_text=f"Optimal: {optimal_period:.1f}d"
            )
        
        fig.update_layout(
            xaxis_title="Holding Period (Days)",
            yaxis_title="Daily Return (₹)",
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
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
        
        return create_base_card(
            html.Div([
                html.H3("Alpha Decay Analysis", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '350px'}
                )
            ]),
            'large',
            'chart'
        )
    
    def _create_timing_efficiency_chart(self, trade_analyzer: TradeAnalyzer) -> html.Div:
        """Create trade timing efficiency metrics chart."""
        
        trade_data = trade_analyzer.trade_log.copy()
        
        # Calculate timing efficiency metrics
        if 'mfe' in trade_data.columns and 'mae' in trade_data.columns:
            # Entry timing efficiency: How close to optimal entry (minimize MAE)
            trade_data['entry_efficiency'] = 1 - abs(trade_data['mae'] / trade_data['pnl'].abs().replace(0, 1))
            
            # Exit timing efficiency: How much of MFE was captured
            trade_data['exit_efficiency'] = trade_data['pnl'] / trade_data['mfe'].replace(0, 1)
            
            # Clean data
            trade_data['entry_efficiency'] = trade_data['entry_efficiency'].clip(-2, 2)
            trade_data['exit_efficiency'] = trade_data['exit_efficiency'].clip(-2, 2)
            
            # Create timing efficiency plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Entry Timing Efficiency Distribution', 'Exit Timing Efficiency Distribution'],
                vertical_spacing=0.15
            )
            
            # Entry efficiency histogram
            fig.add_trace(
                go.Histogram(
                    x=trade_data['entry_efficiency'],
                    name='Entry Efficiency',
                    marker_color=COLORS['info'],
                    opacity=0.7,
                    nbinsx=20
                ),
                row=1, col=1
            )
            
            # Exit efficiency histogram
            fig.add_trace(
                go.Histogram(
                    x=trade_data['exit_efficiency'],
                    name='Exit Efficiency',
                    marker_color=COLORS['secondary'],
                    opacity=0.7,
                    nbinsx=20
                ),
                row=2, col=1
            )
            
            # Add reference lines for good efficiency (>0.7)
            fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS['profit'], row=1, col=1)
            fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS['profit'], row=2, col=1)
            
        else:
            # Create placeholder if MFE/MAE not available
            fig = go.Figure()
            fig.add_annotation(
                text="MFE/MAE data required for timing analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(l=50, r=50, t=40, b=50)
        )
        
        fig.update_xaxes(title_text="Efficiency Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        return create_base_card(
            html.Div([
                html.H3("Timing Efficiency", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    style={'height': '350px'}
                )
            ]),
            'large',
            'chart'
        )
    
    def _create_interactive_trade_table(self, trade_analyzer: TradeAnalyzer) -> html.Div:
        """Create interactive trade table with filtering and sorting."""
        
        # Get categorized trade data
        trade_data = trade_analyzer.categorize_trades()
        
        # Prepare data for table
        table_data = trade_data[[
            'symbol', 'entry_timestamp', 'exit_timestamp', 'entry_price', 'exit_price',
            'quantity', 'pnl', 'duration_days', 'size_category', 'outcome', 'duration_bin'
        ]].copy()
        
        # Add MFE/MAE if available
        if 'mfe' in trade_data.columns:
            table_data['mfe'] = trade_data['mfe']
            table_data['mae'] = trade_data['mae']
        
        # Format data for display
        table_data['entry_timestamp'] = pd.to_datetime(table_data['entry_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        table_data['exit_timestamp'] = pd.to_datetime(table_data['exit_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        table_data['pnl'] = table_data['pnl'].round(2)
        table_data['entry_price'] = table_data['entry_price'].round(2)
        table_data['exit_price'] = table_data['exit_price'].round(2)
        table_data['duration_days'] = table_data['duration_days'].round(1)
        
        # Define columns
        columns = [
            {'name': 'Symbol', 'id': 'symbol', 'type': 'text'},
            {'name': 'Entry Time', 'id': 'entry_timestamp', 'type': 'datetime'},
            {'name': 'Exit Time', 'id': 'exit_timestamp', 'type': 'datetime'},
            {'name': 'Entry Price', 'id': 'entry_price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Exit Price', 'id': 'exit_price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric'},
            {'name': 'P/L (₹)', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Duration (d)', 'id': 'duration_days', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Size', 'id': 'size_category', 'type': 'text'},
            {'name': 'Outcome', 'id': 'outcome', 'type': 'text'},
            {'name': 'Duration Bin', 'id': 'duration_bin', 'type': 'text'}
        ]
        
        # Add MFE/MAE columns if available
        if 'mfe' in table_data.columns:
            columns.extend([
                {'name': 'MFE (₹)', 'id': 'mfe', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'MAE (₹)', 'id': 'mae', 'type': 'numeric', 'format': {'specifier': '.2f'}}
            ])
        
        # Style data conditionally
        style_data_conditional = [
            {
                'if': {'filter_query': '{outcome} = Winner'},
                'backgroundColor': f'{COLORS["profit"]}20',
                'color': COLORS['primary'],
            },
            {
                'if': {'filter_query': '{outcome} = Loser'},
                'backgroundColor': f'{COLORS["loss"]}20',
                'color': COLORS['primary'],
            }
        ]
        
        return create_base_card(
            html.Div([
                html.H3("Individual Trade Details", style={
                    'margin': '0',
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '20px'
                }),
                html.P(f"Interactive table of {len(table_data)} trades with filtering and sorting capabilities", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '14px',
                    'marginBottom': '20px'
                }),
                dash_table.DataTable(
                    data=table_data.to_dict('records'),
                    columns=columns,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    page_current=0,
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'center',
                        'fontFamily': 'Inter, sans-serif',
                        'fontSize': '12px',
                        'padding': '8px'
                    },
                    style_header={
                        'backgroundColor': COLORS['primary'],
                        'color': 'white',
                        'fontWeight': '600',
                        'textAlign': 'center'
                    },
                    style_data_conditional=style_data_conditional,
                    export_format="csv",
                    export_headers="display"
                )
            ]),
            'full-width',
            'default'
        )
    
    def _create_placeholder_content(self) -> html.Div:
        """Create placeholder content when no data is available."""
        
        return html.Div([
            html.Div([
                html.H2("Trade Analytics", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P("Detailed trade-level performance and efficiency analysis", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '40px'}),
            
            create_base_card(
                html.Div([
                    html.I(className="fas fa-chart-bar", style={
                        'fontSize': '64px',
                        'color': COLORS['border'],
                        'marginBottom': '20px'
                    }),
                    html.H3("No Trade Data Available", style={
                        'color': COLORS['primary'],
                        'marginBottom': '15px',
                        'fontSize': '24px',
                        'fontWeight': '600'
                    }),
                    html.P("Please select a strategy and version to view comprehensive trade analytics including:", style={
                        'color': COLORS['tertiary'],
                        'marginBottom': '20px',
                        'fontSize': '16px'
                    }),
                    html.Ul([
                        html.Li("MFE/MAE scatter plot analysis"),
                        html.Li("Trade duration vs profitability breakdown"),
                        html.Li("Profit factor analysis by trade characteristics"),
                        html.Li("Alpha decay patterns over holding periods"),
                        html.Li("Trade timing efficiency metrics"),
                        html.Li("Interactive trade table with filtering")
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
    
    def _create_error_content(self, error_message: str) -> html.Div:
        """Create error content when trade analysis fails."""
        
        return html.Div([
            html.Div([
                html.H2("Trade Analytics", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P("Error loading trade analytics", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '40px'}),
            
            create_base_card(
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={
                        'fontSize': '64px',
                        'color': COLORS['caution'],
                        'marginBottom': '20px'
                    }),
                    html.H3("Error Loading Trade Data", style={
                        'color': COLORS['primary'],
                        'marginBottom': '15px',
                        'fontSize': '24px',
                        'fontWeight': '600'
                    }),
                    html.P(f"Unable to analyze trade data: {error_message}", style={
                        'color': COLORS['tertiary'],
                        'marginBottom': '20px',
                        'fontSize': '16px'
                    }),
                    html.P("Please check that the selected strategy contains valid trade log data.", style={
                        'color': COLORS['tertiary'],
                        'fontSize': '14px'
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


def create_trade_analytics_content(strategy: str = None, version: str = None, 
                                  start_date: datetime = None, end_date: datetime = None) -> html.Div:
    """
    Factory function to create Trade Analytics tab content.
    
    Args:
        strategy: Selected strategy name
        version: Selected version identifier
        start_date: Filter start date
        end_date: Filter end date
        
    Returns:
        Dash HTML Div containing the Trade Analytics tab content
    """
    from .DashboardDataService import DashboardDataService
    
    data_service = DashboardDataService()
    trade_analytics_tab = TradeAnalyticsTab(data_service)
    
    return trade_analytics_tab.create_tab_content(strategy, version, start_date, end_date)