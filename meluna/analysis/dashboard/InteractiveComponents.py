"""
Interactive Dashboard Components - Enhanced UI Elements

This module provides enhanced interactive components including rich tooltips,
interactive KPI cards, synchronized charts, and performance-optimized elements
for creating sophisticated user experiences.

Features:
- Enhanced KPI cards with drill-down capabilities
- Interactive time series charts with synchronization
- Rich tooltip components with contextual information
- Performance-optimized chart components with debouncing
- Responsive modal components for detailed analysis
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import dcc, html, Input, Output, State, callback, clientside_callback, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from .DashboardComponents import COLORS, create_base_card
from .DashboardDataService import DashboardDataService, CachedAnalyticsData

logger = logging.getLogger(__name__)


class InteractiveKPICard:
    """Enhanced KPI card with drill-down and interactive capabilities."""
    
    @staticmethod
    def create_enhanced_kpi_card(
        title: str,
        value: str, 
        subtitle: str = '',
        color: str = COLORS['info'],
        icon: str = 'fas fa-chart-line',
        trend: str = 'neutral',
        size: str = 'small',
        drill_down_enabled: bool = True,
        card_id: str = None
    ) -> html.Div:
        """
        Create enhanced interactive KPI card with drill-down capability.
        
        Args:
            title: Card title
            value: Main value display
            subtitle: Secondary information
            color: Card accent color
            icon: FontAwesome icon class
            trend: Trend indicator ('up', 'down', 'neutral')
            size: Card size ('small', 'medium', 'large')
            drill_down_enabled: Enable click for drill-down
            card_id: Unique identifier for the card
            
        Returns:
            Enhanced interactive KPI card component
        """
        
        if not card_id:
            card_id = f"kpi-{title.lower().replace(' ', '-')}"
        
        # Trend indicator
        trend_icon = {
            'up': 'fas fa-arrow-up',
            'down': 'fas fa-arrow-down', 
            'neutral': 'fas fa-minus'
        }.get(trend, 'fas fa-minus')
        
        trend_color = {
            'up': COLORS['profit'],
            'down': COLORS['loss'],
            'neutral': COLORS['tertiary']
        }.get(trend, COLORS['tertiary'])
        
        # Size configurations
        size_config = {
            'small': {'height': '120px', 'padding': '15px'},
            'medium': {'height': '150px', 'padding': '20px'},
            'large': {'height': '180px', 'padding': '25px'}
        }
        
        config = size_config.get(size, size_config['small'])
        
        # Enhanced styling with hover effects
        card_style = {
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["border"]}',
            'borderLeft': f'4px solid {color}',
            'borderRadius': '8px',
            'padding': config['padding'],
            'height': config['height'],
            'cursor': 'pointer' if drill_down_enabled else 'default',
            'transition': 'all 0.3s ease',
            'position': 'relative',
            'overflow': 'hidden'
        }
        
        # Interactive hover styles (added via CSS class)
        hover_class = 'interactive-kpi-card' if drill_down_enabled else 'static-kpi-card'
        
        card_content = html.Div([
            # Header with icon and trend
            html.Div([
                html.I(className=icon, style={
                    'fontSize': '24px',
                    'color': color,
                    'marginRight': '10px'
                }),
                html.Div([
                    html.I(className=trend_icon, style={
                        'fontSize': '12px',
                        'color': trend_color
                    })
                ], style={'float': 'right'})
            ], style={'marginBottom': '10px'}),
            
            # Main value
            html.H3(value, style={
                'color': color,
                'fontWeight': '700',
                'fontSize': '28px',
                'margin': '0',
                'lineHeight': '1.2'
            }),
            
            # Title
            html.H5(title, style={
                'color': COLORS['primary'],
                'fontWeight': '600',
                'fontSize': '14px',
                'margin': '5px 0',
                'textTransform': 'uppercase',
                'letterSpacing': '0.5px'
            }),
            
            # Subtitle
            html.P(subtitle, style={
                'color': COLORS['tertiary'],
                'fontSize': '12px',
                'margin': '0',
                'lineHeight': '1.4'
            }),
            
            # Drill-down indicator
            html.Div([
                html.I(className='fas fa-external-link-alt', style={
                    'fontSize': '10px',
                    'color': COLORS['tertiary'],
                    'opacity': '0.7'
                })
            ], style={
                'position': 'absolute',
                'bottom': '10px',
                'right': '10px',
                'display': 'block' if drill_down_enabled else 'none'
            })
            
        ], style=card_style, className=hover_class, id={'type': 'kpi-card', 'index': card_id})
        
        return card_content
    
    @staticmethod
    def create_kpi_group_with_interactions(
        title: str,
        kpi_cards: List[html.Div],
        group_id: str = None
    ) -> html.Div:
        """Create a group of interactive KPI cards with coordinated interactions."""
        
        if not group_id:
            group_id = f"kpi-group-{title.lower().replace(' ', '-')}"
        
        return html.Div([
            html.H4(title, style={
                'color': COLORS['primary'],
                'marginBottom': '20px',
                'fontSize': '18px',
                'fontWeight': '600',
                'borderBottom': f'2px solid {COLORS["border"]}',
                'paddingBottom': '8px'
            }),
            html.Div(kpi_cards, style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))',
                'gap': '20px',
                'marginBottom': '30px'
            }, id=group_id)
        ])


class SynchronizedChart:
    """Chart component with cross-chart synchronization capabilities."""
    
    def __init__(self, chart_id: str, sync_group: str = 'default'):
        """
        Initialize synchronized chart.
        
        Args:
            chart_id: Unique identifier for the chart
            sync_group: Group for synchronization (charts in same group sync together)
        """
        self.chart_id = chart_id
        self.sync_group = sync_group
    
    def create_time_series_chart(
        self,
        data: pd.Series,
        title: str,
        y_title: str = 'Value',
        color: str = COLORS['info'],
        show_hover_line: bool = True,
        enable_zoom_sync: bool = True,
        enable_click_filter: bool = False
    ) -> dcc.Graph:
        """
        Create synchronized time series chart.
        
        Args:
            data: Time series data
            title: Chart title
            y_title: Y-axis title
            color: Line color
            show_hover_line: Enable hover synchronization line
            enable_zoom_sync: Enable zoom synchronization
            enable_click_filter: Enable click-to-filter functionality
            
        Returns:
            Interactive chart component
        """
        
        fig = go.Figure()
        
        # Add main data trace
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=title,
            line=dict(color=color, width=2),
            hovertemplate='<b>%{y:,.2f}</b><br>%{x}<br><extra></extra>'
        ))
        
        # Configure layout for synchronization
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 16, 'color': COLORS['primary']},
                'x': 0
            },
            xaxis_title='Date',
            yaxis_title=y_title,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            hovermode='x unified' if show_hover_line else 'closest',
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
        
        # Configure chart for interactions
        config = {
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'displaylogo': False
        }
        
        # Add sync and filter attributes
        chart_props = {
            'figure': fig,
            'config': config,
            'id': {'type': 'sync-chart', 'group': self.sync_group, 'id': self.chart_id}
        }
        
        if enable_click_filter:
            chart_props['id'] = {'type': 'filterable-chart', 'index': self.chart_id}
        
        return dcc.Graph(**chart_props)
    
    def create_scatter_chart(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str,
        x_title: str = 'X',
        y_title: str = 'Y',
        color_data: Optional[List[float]] = None,
        size_data: Optional[List[float]] = None,
        hover_data: Optional[Dict[str, List]] = None
    ) -> dcc.Graph:
        """
        Create interactive scatter chart with rich tooltips.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            title: Chart title
            x_title: X-axis title
            y_title: Y-axis title
            color_data: Optional color mapping data
            size_data: Optional size mapping data
            hover_data: Additional hover information
            
        Returns:
            Interactive scatter chart
        """
        
        fig = go.Figure()
        
        # Prepare hover template
        hover_template = f'<b>{title}</b><br>'
        hover_template += f'{x_title}: %{{x}}<br>'
        hover_template += f'{y_title}: %{{y}}<br>'
        
        if hover_data:
            for key, values in hover_data.items():
                hover_template += f'{key}: %{{customdata[{len(hover_template.split("customdata")) - 1}]}}<br>'
        
        hover_template += '<extra></extra>'
        
        # Create scatter trace
        scatter_trace = go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name=title,
            marker=dict(
                size=size_data if size_data else 8,
                color=color_data if color_data else COLORS['info'],
                colorscale='Viridis' if color_data else None,
                showscale=bool(color_data),
                opacity=0.7
            ),
            hovertemplate=hover_template,
            customdata=list(zip(*hover_data.values())) if hover_data else None
        )
        
        fig.add_trace(scatter_trace)
        
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 16, 'color': COLORS['primary']},
                'x': 0
            },
            xaxis_title=x_title,
            yaxis_title=y_title,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        config = {
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            'displaylogo': False
        }
        
        return dcc.Graph(
            figure=fig,
            config=config,
            id={'type': 'filterable-chart', 'index': self.chart_id}
        )


class RichTooltipComponent:
    """Enhanced tooltip component with rich contextual information."""
    
    @staticmethod
    def create_info_tooltip(
        content: str,
        tooltip_text: str,
        placement: str = 'top'
    ) -> dbc.Tooltip:
        """
        Create informational tooltip with rich content.
        
        Args:
            content: Main content to display tooltip on
            tooltip_text: Tooltip text content
            placement: Tooltip placement ('top', 'bottom', 'left', 'right')
            
        Returns:
            Rich tooltip component
        """
        
        tooltip_id = f"tooltip-{hash(content + tooltip_text) % 10000}"
        
        return html.Div([
            html.Span([
                content,
                html.I(className='fas fa-info-circle', style={
                    'marginLeft': '5px',
                    'color': COLORS['info'],
                    'fontSize': '12px'
                })
            ], id=tooltip_id),
            dbc.Tooltip(
                tooltip_text,
                target=tooltip_id,
                placement=placement,
                style={'maxWidth': '300px', 'fontSize': '12px'}
            )
        ])
    
    @staticmethod
    def create_metric_tooltip(
        metric_name: str,
        current_value: str,
        methodology: str,
        benchmark_value: Optional[str] = None,
        interpretation: Optional[str] = None
    ) -> html.Div:
        """
        Create comprehensive metric tooltip with methodology and benchmarks.
        
        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            methodology: Calculation methodology
            benchmark_value: Optional benchmark comparison
            interpretation: Optional interpretation guidance
            
        Returns:
            Rich metric tooltip component
        """
        
        tooltip_content = html.Div([
            html.H6(metric_name, style={
                'color': COLORS['primary'],
                'marginBottom': '8px',
                'fontWeight': '600'
            }),
            html.P(f"Current Value: {current_value}", style={
                'margin': '4px 0',
                'fontWeight': '500'
            }),
            html.Hr(style={'margin': '8px 0'}),
            html.P([
                html.Strong("Methodology: "),
                methodology
            ], style={'margin': '4px 0', 'fontSize': '12px'}),
        ])
        
        if benchmark_value:
            tooltip_content.children.append(
                html.P([
                    html.Strong("Benchmark: "),
                    benchmark_value
                ], style={'margin': '4px 0', 'fontSize': '12px'})
            )
        
        if interpretation:
            tooltip_content.children.extend([
                html.Hr(style={'margin': '8px 0'}),
                html.P([
                    html.Strong("Interpretation: "),
                    interpretation
                ], style={'margin': '4px 0', 'fontSize': '12px'})
            ])
        
        return tooltip_content


class PerformanceOptimizedChart:
    """Chart component optimized for performance with large datasets."""
    
    def __init__(self, max_points: int = 2000, debounce_ms: int = 100):
        """
        Initialize performance-optimized chart.
        
        Args:
            max_points: Maximum points to display (decimation threshold)
            debounce_ms: Debounce time for interactions in milliseconds
        """
        self.max_points = max_points
        self.debounce_ms = debounce_ms
    
    def decimate_data(self, data: pd.Series) -> pd.Series:
        """
        Intelligently decimate data for performance while preserving key features.
        
        Args:
            data: Time series data to decimate
            
        Returns:
            Decimated data maintaining important features
        """
        if len(data) <= self.max_points:
            return data
        
        # Use intelligent decimation preserving peaks and valleys
        step = len(data) // self.max_points
        
        # Always include first and last points
        indices = [0]
        
        # Add points at regular intervals
        for i in range(step, len(data) - step, step):
            indices.append(i)
        
        # Add local extrema (peaks and valleys) in each segment
        for i in range(0, len(data) - step, step):
            segment = data.iloc[i:i+step]
            if len(segment) > 2:
                max_idx = segment.idxmax()
                min_idx = segment.idxmin()
                
                # Add indices of local extrema
                if max_idx not in indices:
                    indices.append(data.index.get_loc(max_idx))
                if min_idx not in indices:
                    indices.append(data.index.get_loc(min_idx))
        
        # Always include last point
        indices.append(len(data) - 1)
        
        # Sort indices and remove duplicates
        indices = sorted(list(set(indices)))
        
        return data.iloc[indices]
    
    def create_optimized_time_series(
        self,
        data: pd.Series,
        title: str,
        chart_id: str,
        color: str = COLORS['info']
    ) -> dcc.Graph:
        """
        Create performance-optimized time series chart.
        
        Args:
            data: Time series data
            title: Chart title
            chart_id: Chart identifier
            color: Line color
            
        Returns:
            Optimized chart component
        """
        
        # Decimate data if necessary
        optimized_data = self.decimate_data(data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=optimized_data.index,
            y=optimized_data.values,
            mode='lines',
            name=title,
            line=dict(color=color, width=2),
            hovertemplate='<b>%{y:,.2f}</b><br>%{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f"{title} (Optimized: {len(optimized_data):,} of {len(data):,} points)",
                'font': {'size': 16, 'color': COLORS['primary']},
                'x': 0
            },
            xaxis_title='Date',
            yaxis_title='Value',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(l=60, r=20, t=60, b=40)
        )
        
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'responsive': True
        }
        
        return dcc.Graph(
            figure=fig,
            config=config,
            id=chart_id
        )


class InteractiveBookmarkBar:
    """Bookmark bar component for saving and restoring analysis states."""
    
    @staticmethod
    def create_bookmark_bar() -> html.Div:
        """Create interactive bookmark bar with save/restore functionality."""
        
        return html.Div([
            html.Div([
                html.H6("Bookmarks", style={
                    'color': COLORS['primary'],
                    'margin': '0',
                    'fontSize': '14px',
                    'fontWeight': '600'
                }),
                html.Div([
                    dbc.Button([
                        html.I(className='fas fa-bookmark'),
                        " Save State"
                    ], 
                    color='info', 
                    size='sm', 
                    id='create-bookmark-btn',
                    style={'marginRight': '10px'}),
                    
                    dcc.Dropdown(
                        id='bookmark-selector',
                        placeholder='Select saved state...',
                        style={'width': '200px', 'display': 'inline-block'},
                        clearable=True
                    ),
                    
                    dbc.Button([
                        html.I(className='fas fa-share-alt')
                    ], 
                    color='secondary', 
                    size='sm', 
                    id='share-bookmark-btn',
                    style={'marginLeft': '10px'},
                    title='Share analysis state')
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'padding': '10px 20px',
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '4px',
                'marginBottom': '20px'
            })
        ])


def create_css_enhancements() -> str:
    """Generate CSS for enhanced interactive components."""
    
    return f"""
    /* Interactive KPI Card Enhancements */
    .interactive-kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        border-color: {COLORS['info']} !important;
    }}
    
    .static-kpi-card {{
        cursor: default;
    }}
    
    /* Chart synchronization highlights */
    .sync-highlight {{
        background-color: rgba(0, 123, 255, 0.1);
        transition: background-color 0.2s ease;
    }}
    
    /* Modal enhancements */
    .modal-content {{
        border-radius: 8px;
        border: none;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }}
    
    /* Tooltip enhancements */
    .tooltip {{
        font-size: 12px;
        line-height: 1.4;
    }}
    
    /* Performance indicators */
    .performance-indicator {{
        font-size: 10px;
        color: {COLORS['tertiary']};
        opacity: 0.8;
    }}
    
    /* Hover synchronization line */
    .hover-sync-line {{
        pointer-events: none;
        z-index: 10;
    }}
    
    /* Bookmark bar styling */
    .bookmark-bar {{
        position: sticky;
        top: 90px;
        z-index: 100;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    """