"""
Unified Portfolio Statistics Table Component

This module provides a comprehensive, professionally styled statistics table that consolidates
Performance Attribution and Risk Analysis sections into a single unified presentation.

Features:
- Single comprehensive table with all key portfolio metrics
- Logical groupings: Performance, Risk, Distribution Statistics  
- Professional institutional-grade styling
- Responsive design with appropriate width sizing
- Clear metric labels and formatted values
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dash import html, dcc
import logging

from .DashboardComponents import create_base_card, COLORS
from .DashboardDataService import CachedAnalyticsData

logger = logging.getLogger(__name__)


class UnifiedPortfolioStatisticsTable:
    """
    Unified portfolio statistics table component that consolidates all portfolio
    metrics into a single, professional, well-organized table.
    """
    
    def __init__(self):
        """Initialize the unified statistics table component."""
        pass
    
    def create_unified_statistics_table(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """
        Create unified comprehensive statistics table.
        
        Args:
            analytics_data: Cached analytics data containing portfolio metrics
            
        Returns:
            HTML Div containing the unified statistics table
        """
        if not analytics_data or not analytics_data.portfolio_metrics:
            return self._create_placeholder_table()
        
        # Extract metrics
        metrics = analytics_data.portfolio_metrics
        
        # Organize metrics into logical groups
        metric_groups = self._organize_metrics_into_groups(metrics)
        
        # Create table content
        table_content = self._create_table_content(metric_groups)
        
        # Wrap in styled container
        return self._create_table_container(table_content)
    
    def _organize_metrics_into_groups(self, metrics: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize metrics into logical groups for table display.
        
        Args:
            metrics: Portfolio metrics dictionary
            
        Returns:
            Dictionary with grouped metrics
        """
        # Performance Metrics Group
        performance_metrics = [
            {
                'label': 'Total Return',
                'value': f"{metrics.get('Total Return (%)', 0):+.2f}%",
                'description': 'Cumulative portfolio return',
                'trend': 'positive' if metrics.get('Total Return (%)', 0) > 0 else 'negative'
            },
            {
                'label': 'CAGR',
                'value': f"{metrics.get('CAGR (%)', 0):.2f}%",
                'description': 'Compound Annual Growth Rate',
                'trend': 'positive' if metrics.get('CAGR (%)', 0) > 0 else 'negative'
            },
            {
                'label': 'Annualized Volatility',
                'value': f"{metrics.get('Annualized Volatility (%)', 0):.2f}%",
                'description': 'Annual volatility of returns',
                'trend': 'neutral'
            },
            {
                'label': 'Net TRI',
                'value': f"{metrics.get('Net TRI (Start=100)', 100):.1f}",
                'description': 'Total Return Index (Start=100)',
                'trend': 'positive' if metrics.get('Net TRI (Start=100)', 100) > 100 else 'negative'
            },
            {
                'label': 'Gain-to-Pain Ratio',
                'value': f"{metrics.get('Gain-to-Pain Ratio', 0):.2f}",
                'description': 'Ratio of positive to negative returns',
                'trend': 'positive' if metrics.get('Gain-to-Pain Ratio', 0) > 2 else 'neutral'
            }
        ]
        
        # Risk Metrics Group
        risk_metrics = [
            {
                'label': 'Sharpe Ratio',
                'value': f"{metrics.get('Sharpe Ratio', 0):.3f}",
                'description': 'Risk-adjusted return measure',
                'trend': 'positive' if metrics.get('Sharpe Ratio', 0) > 1 else 'neutral'
            },
            {
                'label': 'Sortino Ratio',
                'value': f"{metrics.get('Sortino Ratio', 0):.3f}",
                'description': 'Downside risk-adjusted return',
                'trend': 'positive' if metrics.get('Sortino Ratio', 0) > 1.5 else 'neutral'
            },
            {
                'label': 'Maximum Drawdown',
                'value': f"{metrics.get('Max Drawdown (%)', 0):.2f}%",
                'description': 'Largest peak-to-trough decline',
                'trend': 'negative'
            },
            {
                'label': 'VaR (95%)',
                'value': f"{metrics.get('VaR 95% (%)', 0):.2f}%",
                'description': 'Value at Risk at 95% confidence',
                'trend': 'negative'
            },
            {
                'label': 'CVaR (95%)',
                'value': f"{metrics.get('CVaR 95% (%)', 0):.2f}%",
                'description': 'Conditional Value at Risk',
                'trend': 'negative'
            }
        ]
        
        # Distribution Statistics Group
        distribution_metrics = [
            {
                'label': 'Skewness',
                'value': f"{metrics.get('Skewness', 0):.3f}",
                'description': 'Return distribution asymmetry',
                'trend': 'positive' if metrics.get('Skewness', 0) > 0 else 'negative'
            },
            {
                'label': 'Kurtosis',
                'value': f"{metrics.get('Kurtosis', 0):.3f}",
                'description': 'Return distribution tail thickness',
                'trend': 'neutral'
            },
            {
                'label': 'Downside Deviation',
                'value': f"{metrics.get('Downside Deviation (%)', 0):.2f}%",
                'description': 'Volatility of negative returns',
                'trend': 'neutral'
            },
            {
                'label': 'Longest Drawdown',
                'value': f"{metrics.get('Longest Drawdown Duration (days)', 0):.0f} days",
                'description': 'Maximum drawdown duration',
                'trend': 'neutral'
            },
            {
                'label': 'Number of Drawdowns',
                'value': f"{metrics.get('Number of Drawdown Periods', 0):.0f}",
                'description': 'Total number of drawdown periods',
                'trend': 'neutral'
            }
        ]
        
        # Horizon Returns Group
        horizon_returns = self._create_horizon_returns_metrics(metrics)
        
        return {
            'Performance Metrics': performance_metrics,
            'Horizon Returns': horizon_returns,
            'Risk Metrics': risk_metrics,
            'Distribution Statistics': distribution_metrics
        }
    
    def _create_horizon_returns_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create horizon returns metrics for the table.
        
        Args:
            metrics: Portfolio metrics dictionary
            
        Returns:
            List of horizon return metrics
        """
        horizon_periods = ['1M', '3M', '6M', '1Y', '2Y', '3Y']
        horizon_metrics = []
        
        for period in horizon_periods:
            return_key = f'{period} Return (%)'
            sharpe_key = f'{period} Sharpe Ratio'
            
            # Get return value
            return_value = metrics.get(return_key)
            sharpe_value = metrics.get(sharpe_key)
            
            if return_value is not None:
                # Format return value
                return_str = f"{return_value:+.2f}%"
                trend = 'positive' if return_value > 0 else 'negative'
                
                # Add Sharpe info if available
                if sharpe_value is not None:
                    description = f"{period} period return (Sharpe: {sharpe_value:.2f})"
                else:
                    description = f"{period} period return"
            else:
                return_str = "N/A"
                trend = 'neutral'
                description = f"{period} period return (insufficient data)"
            
            horizon_metrics.append({
                'label': f'{period} Return',
                'value': return_str,
                'description': description,
                'trend': trend
            })
        
        return horizon_metrics
    
    def _create_table_content(self, metric_groups: Dict[str, List[Dict[str, Any]]]) -> html.Div:
        """
        Create the main table content with grouped metrics.
        
        Args:
            metric_groups: Organized metric groups
            
        Returns:
            HTML table content
        """
        table_rows = []
        
        for group_name, metrics in metric_groups.items():
            # Add group header row
            table_rows.append(
                html.Tr([
                    html.Td(
                        html.Strong(group_name),
                        colSpan=3,
                        className="group-header",
                        style={
                            'backgroundColor': COLORS['background'],
                            'color': COLORS['primary'],
                            'fontWeight': '600',
                            'fontSize': '14px',
                            'padding': '12px 16px',
                            'borderTop': f'2px solid {COLORS["info"]}',
                            'borderBottom': f'1px solid {COLORS["border"]}'
                        }
                    )
                ])
            )
            
            # Add metric rows
            for i, metric in enumerate(metrics):
                # Determine value color based on trend
                value_color = self._get_trend_color(metric['trend'])
                
                # Alternating row background
                row_bg = COLORS['background'] if i % 2 == 0 else 'white'
                
                table_rows.append(
                    html.Tr([
                        # Metric Label
                        html.Td(
                            metric['label'],
                            style={
                                'padding': '10px 16px',
                                'backgroundColor': row_bg,
                                'borderBottom': f'1px solid {COLORS["border"]}',
                                'fontWeight': '500',
                                'color': COLORS['secondary'],
                                'fontSize': '13px',
                                'width': '40%'
                            }
                        ),
                        # Metric Value
                        html.Td(
                            metric['value'],
                            style={
                                'padding': '10px 16px',
                                'backgroundColor': row_bg,
                                'borderBottom': f'1px solid {COLORS["border"]}',
                                'fontWeight': '600',
                                'color': value_color,
                                'fontSize': '13px',
                                'textAlign': 'right',
                                'width': '25%',
                                'fontFamily': 'JetBrains Mono, monospace'
                            }
                        ),
                        # Metric Description
                        html.Td(
                            metric['description'],
                            style={
                                'padding': '10px 16px',
                                'backgroundColor': row_bg,
                                'borderBottom': f'1px solid {COLORS["border"]}',
                                'color': COLORS['tertiary'],
                                'fontSize': '12px',
                                'width': '35%'
                            }
                        )
                    ], className="metric-row")
                )
        
        # Create table
        table = html.Table(
            children=[
                # Table header
                html.Thead([
                    html.Tr([
                        html.Th("Metric", style={
                            'padding': '12px 16px',
                            'backgroundColor': COLORS['primary'],
                            'color': 'white',
                            'fontWeight': '600',
                            'fontSize': '13px',
                            'borderBottom': 'none',
                            'textAlign': 'left'
                        }),
                        html.Th("Value", style={
                            'padding': '12px 16px',
                            'backgroundColor': COLORS['primary'],
                            'color': 'white',
                            'fontWeight': '600',
                            'fontSize': '13px',
                            'borderBottom': 'none',
                            'textAlign': 'right'
                        }),
                        html.Th("Description", style={
                            'padding': '12px 16px',
                            'backgroundColor': COLORS['primary'],
                            'color': 'white',
                            'fontWeight': '600',
                            'fontSize': '13px',
                            'borderBottom': 'none',
                            'textAlign': 'left'
                        })
                    ])
                ]),
                # Table body
                html.Tbody(table_rows)
            ],
            style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'backgroundColor': 'white',
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '6px',
                'overflow': 'hidden',
                'fontSize': '13px',
                'fontFamily': 'Inter, sans-serif'
            }
        )
        
        return table
    
    def _create_table_container(self, table_content: html.Table) -> html.Div:
        """
        Create styled container for the statistics table.
        
        Args:
            table_content: HTML table element
            
        Returns:
            Styled container with table
        """
        content = html.Div([
            # Section header
            html.Div([
                html.H3("Portfolio Statistics", style={
                    'margin': '0',
                    'fontSize': '20px',
                    'fontWeight': '600',
                    'color': COLORS['primary'],
                    'marginBottom': '8px'
                }),
                html.P("Comprehensive performance, risk, and distribution metrics", style={
                    'margin': '0',
                    'fontSize': '14px',
                    'color': COLORS['tertiary'],
                    'marginBottom': '20px'
                })
            ]),
            
            # Table wrapper with controlled width
            html.Div([
                table_content
            ], style={
                'overflowX': 'auto',
                'maxWidth': '100%'
            })
        ])
        
        return create_base_card(content, size='full-width', card_type='table')
    
    def _create_placeholder_table(self) -> html.Div:
        """Create placeholder table when no data is available."""
        
        placeholder_content = html.Div([
            html.Div([
                html.I(className="fas fa-table", style={
                    'fontSize': '48px',
                    'color': COLORS['border'],
                    'marginBottom': '20px'
                }),
                html.H4("No Statistics Available", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px'
                }),
                html.P("Select a strategy and version to view portfolio statistics", style={
                    'color': COLORS['tertiary'],
                    'textAlign': 'center'
                })
            ], style={
                'textAlign': 'center',
                'padding': '60px 20px'
            })
        ])
        
        return create_base_card(placeholder_content, size='full-width')
    
    def _get_trend_color(self, trend: str) -> str:
        """
        Get color based on metric trend.
        
        Args:
            trend: Trend indicator ('positive', 'negative', 'neutral')
            
        Returns:
            Color string
        """
        if trend == 'positive':
            return COLORS['profit']
        elif trend == 'negative':
            return COLORS['loss']
        else:
            return COLORS['secondary']


def create_unified_portfolio_statistics_table(analytics_data: CachedAnalyticsData) -> html.Div:
    """
    Factory function to create unified portfolio statistics table.
    
    Args:
        analytics_data: Cached analytics data with portfolio metrics
        
    Returns:
        Unified statistics table component
    """
    table_component = UnifiedPortfolioStatisticsTable()
    return table_component.create_unified_statistics_table(analytics_data)