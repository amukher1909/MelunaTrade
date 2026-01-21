"""
Individual Trade Tab - Interactive Single Trade Analysis

This module implements the Individual Trade tab with comprehensive single-trade
visualization and analysis integrating with the existing trade visualization system.

Features:
- Interactive trade selection and navigation
- Candlestick charts with trade markers and annotations
- Risk management level visualization (stop loss, take profit)
- Context window controls for chart zoom and focus
- Trade metrics and performance analysis
- Collapsible side panels for controls and information
- Responsive layout following dashboard theme
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

from dash import dcc, html, Input, Output, State, callback, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from .DashboardComponents import (
    create_kpi_card, create_chart_container, create_card_grid, 
    create_card_group, create_base_card, COLORS, CARD_SIZES
)
from .DashboardDataService import DashboardDataService, CachedAnalyticsData
from ..trade_visualization.visualizations.TradeVisualizationTab import TradeVisualizationTab

logger = logging.getLogger(__name__)


class IndividualTradeTab:
    """
    Individual Trade tab component providing comprehensive single-trade analysis.
    
    Integrates with the trade visualization system to display detailed trade analysis
    with interactive charts, controls, and metrics display.
    """
    
    def __init__(self, data_service: DashboardDataService):
        """
        Initialize Individual Trade tab.
        
        Args:
            data_service: Dashboard data service for loading analytics data
        """
        self.data_service = data_service
        self.trade_viz_tab = TradeVisualizationTab(data_service)
        
    def create_tab_content(self, strategy: str = None, version: str = None, 
                          start_date: datetime = None, end_date: datetime = None) -> html.Div:
        """
        Create Individual Trade tab content with real data integration.
        
        Args:
            strategy: Selected strategy name
            version: Selected version identifier  
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            Dash HTML Div containing the complete Individual Trade tab
        """
        
        # Load analytics data to check availability
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
        if analytics_data and hasattr(analytics_data, 'trade_log') and analytics_data.trade_log is not None:
            # Store current strategy/version in data service for TradeVisualizationTab
            self.data_service._current_strategy = strategy
            self.data_service._current_version = version
            return self._create_content_with_data(analytics_data, strategy, version, start_date, end_date)
        else:
            return self._create_placeholder_content()
    
    def _create_content_with_data(self, analytics_data: CachedAnalyticsData,
                                 strategy: str, version: str,
                                 start_date: datetime = None, end_date: datetime = None) -> html.Div:
        """Create tab content with real trade data."""
        
        # Get trade count for header info
        trade_count = len(analytics_data.trade_log) if analytics_data.trade_log is not None else 0
        
        # Create main layout with responsive design
        return html.Div([
            # Tab Header
            html.Div([
                html.H2("Individual Trade Analysis", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P(
                    f"Detailed single-trade visualization and analysis for {strategy} {version} "
                    f"({trade_count} trades available)",
                    style={
                        'color': COLORS['tertiary'],
                        'fontSize': '16px',
                        'marginBottom': '30px',
                        'lineHeight': '1.6'
                    }
                )
            ], style={'marginBottom': '20px'}),
            
            # Use enhanced TradeVisualizationTab with complete functionality
            self.trade_viz_tab.create_tab_content(strategy, version, start_date, end_date),
            
            # State stores
            dcc.Store(id='individual-trade-state', data={
                'current_trade_index': 0,
                'bars_before': 50,
                'bars_after': 20,
                'show_volume': True,
                'show_indicators': True
            }),
            dcc.Store(id='trade-data-store', data={}),
            
        ], style={
            'padding': '20px',
            'backgroundColor': COLORS['page_bg'],
            'minHeight': 'calc(100vh - 90px)',
            'position': 'relative'
        })
    
    def _create_placeholder_content(self) -> html.Div:
        """Create placeholder content when no data is available."""
        
        return html.Div([
            # Tab Header
            html.Div([
                html.H2("Individual Trade Analysis", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P("Select a strategy and version to view individual trade analysis", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '20px'}),
            
            # Placeholder layout
            dbc.Container([
                dbc.Row([
                    # Left sidebar placeholder
                    dbc.Col([
                        create_base_card(
                            html.Div([
                                html.H5("Trade Navigation", style={
                                    'color': COLORS['primary'],
                                    'marginBottom': '15px'
                                }),
                                html.P("No trades available", style={
                                    'color': COLORS['tertiary'],
                                    'fontSize': '14px',
                                    'textAlign': 'center'
                                })
                            ]),
                            size='full-width'
                        )
                    ], width=12, md=3),
                    
                    # Main chart placeholder
                    dbc.Col([
                        create_base_card(
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-chart-candlestick", style={
                                        'fontSize': '64px',
                                        'color': COLORS['border'],
                                        'marginBottom': '20px'
                                    }),
                                    html.H4("Trade Chart", style={
                                        'color': COLORS['primary'],
                                        'marginBottom': '15px'
                                    }),
                                    html.P("Select a strategy and version to view trade charts", style={
                                        'color': COLORS['tertiary'],
                                        'fontSize': '16px'
                                    })
                                ], style={
                                    'textAlign': 'center',
                                    'padding': '80px 20px'
                                })
                            ]),
                            size='chart'
                        )
                    ], width=12, md=9)
                ])
            ], fluid=True),
            
            # Instructions
            html.Div([
                create_base_card(
                    html.Div([
                        html.H4("Getting Started", style={
                            'color': COLORS['primary'],
                            'marginBottom': '15px',
                            'fontSize': '18px',
                            'fontWeight': '600'
                        }),
                        html.P("To analyze individual trades:", style={
                            'color': COLORS['tertiary'], 
                            'marginBottom': '10px'
                        }),
                        html.Ol([
                            html.Li("Select a strategy from the dropdown in the header"),
                            html.Li("Choose a version to analyze"),
                            html.Li("Use the trade navigation controls to browse individual trades"),
                            html.Li("Adjust context window settings to focus on specific periods"),
                            html.Li("View detailed trade metrics and performance analysis")
                        ], style={'color': COLORS['tertiary'], 'marginLeft': '20px'})
                    ]),
                    'full-width'
                )
            ], style={'marginTop': '40px'})
            
        ], style={
            'padding': '20px', 
            'backgroundColor': COLORS['page_bg'],
            'minHeight': 'calc(100vh - 90px)'
        })
    
    def _create_trade_navigation_panel(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create trade navigation and selection panel."""
        
        trade_log = analytics_data.trade_log
        trade_options = []
        
        if trade_log is not None and len(trade_log) > 0:
            for i, (_, trade) in enumerate(trade_log.iterrows()):
                symbol = trade.get('symbol', 'UNKNOWN')
                pnl = trade.get('pnl', 0)
                entry_time = trade.get('entry_timestamp', '')
                
                # Format label
                label = f"Trade {i+1:03d} - {symbol} ({pnl:+.2f})"
                if entry_time:
                    try:
                        entry_date = pd.to_datetime(entry_time).strftime('%m/%d')
                        label = f"Trade {i+1:03d} - {symbol} {entry_date} ({pnl:+.2f})"
                    except:
                        pass
                
                trade_options.append({'label': label, 'value': i})
        
        return create_base_card(
            html.Div([
                html.H5("Trade Navigation", style={
                    'color': COLORS['primary'],
                    'marginBottom': '15px',
                    'fontSize': '16px',
                    'fontWeight': '600'
                }),
                
                # Trade selection dropdown
                html.Label("Select Trade:", style={
                    'color': COLORS['primary'],
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'marginBottom': '5px'
                }),
                dcc.Dropdown(
                    id='individual-trade-dropdown',
                    options=trade_options,
                    value=0 if trade_options else None,
                    placeholder="No trades available" if not trade_options else "Select trade...",
                    style={'marginBottom': '15px'},
                    clearable=False
                ),
                
                # Navigation buttons
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="fas fa-step-backward"), " First"],
                            id='first-trade-btn',
                            color='outline-secondary',
                            size='sm',
                            disabled=len(trade_options) == 0
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-chevron-left"), " Prev"],
                            id='prev-trade-btn',
                            color='outline-secondary',
                            size='sm',
                            disabled=len(trade_options) == 0
                        ),
                        dbc.Button(
                            ["Next ", html.I(className="fas fa-chevron-right")],
                            id='next-trade-btn',
                            color='outline-secondary',
                            size='sm',
                            disabled=len(trade_options) == 0
                        ),
                        dbc.Button(
                            ["Last ", html.I(className="fas fa-step-forward")],
                            id='last-trade-btn',
                            color='outline-secondary',
                            size='sm',
                            disabled=len(trade_options) == 0
                        )
                    ], style={'width': '100%'})
                ], style={'marginBottom': '15px'}),
                
                # Trade count indicator
                html.P(
                    f"Showing {len(trade_options)} trades" if trade_options else "No trades available",
                    style={
                        'color': COLORS['tertiary'],
                        'fontSize': '12px',
                        'textAlign': 'center',
                        'margin': '0'
                    }
                )
            ]),
            size='full-width'
        )
    
    def _create_context_controls_panel(self) -> html.Div:
        """Create context window controls panel."""
        
        return create_base_card(
            html.Div([
                html.H5("Context Window", style={
                    'color': COLORS['primary'],
                    'marginBottom': '15px',
                    'fontSize': '16px',
                    'fontWeight': '600'
                }),
                
                # Bars before entry
                html.Label("Bars Before Entry:", style={
                    'color': COLORS['primary'],
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'marginBottom': '5px'
                }),
                dcc.Slider(
                    id='individual-bars-before-slider',
                    min=20,
                    max=200,
                    step=10,
                    value=50,
                    marks={20: '20', 50: '50', 100: '100', 200: '200'},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),
                
                html.Br(),
                
                # Bars after exit
                html.Label("Bars After Exit:", style={
                    'color': COLORS['primary'],
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'marginBottom': '5px'
                }),
                dcc.Slider(
                    id='individual-bars-after-slider',
                    min=10,
                    max=100,
                    step=5,
                    value=20,
                    marks={10: '10', 20: '20', 50: '50', 100: '100'},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ]),
            size='full-width'
        )
    
    def _create_chart_options_panel(self) -> html.Div:
        """Create chart display options panel."""
        
        return create_base_card(
            html.Div([
                html.H5("Chart Options", style={
                    'color': COLORS['primary'],
                    'marginBottom': '15px',
                    'fontSize': '16px',
                    'fontWeight': '600'
                }),
                
                # Chart display options
                dbc.Checklist(
                    id='individual-chart-options',
                    options=[
                        {"label": " Show Volume", "value": "volume"},
                        {"label": " Show Technical Indicators", "value": "indicators"},
                        {"label": " Show Risk Levels", "value": "risk_levels"},
                        {"label": " Show Trade Annotations", "value": "annotations"}
                    ],
                    value=["volume", "indicators", "risk_levels", "annotations"],
                    style={'marginBottom': '15px'}
                ),
                
                # Reset button
                dbc.Button(
                    "Reset to Default",
                    id='individual-reset-options-btn',
                    color='outline-primary',
                    size='sm',
                    style={'width': '100%'}
                )
            ]),
            size='full-width'
        )
    
    def _create_main_chart_area(self) -> html.Div:
        """Create the main chart display area."""
        
        return create_base_card(
            html.Div([
                # Chart header with controls
                html.Div([
                    html.H4("Trade Chart", style={
                        'color': COLORS['primary'],
                        'margin': '0',
                        'fontSize': '18px',
                        'fontWeight': '600'
                    }),
                    
                    # Chart action buttons
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button(
                                [html.I(className="fas fa-home"), " Reset Zoom"],
                                id='individual-reset-zoom-btn',
                                color='outline-secondary',
                                size='sm'
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-expand"), " Fullscreen"],
                                id='individual-fullscreen-btn',
                                color='outline-secondary',
                                size='sm'
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-download"), " Export"],
                                id='individual-export-btn',
                                color='outline-primary',
                                size='sm'
                            )
                        ])
                    ])
                ], style={
                    'display': 'flex',
                    'justifyContent': 'space-between',
                    'alignItems': 'center',
                    'marginBottom': '20px',
                    'paddingBottom': '12px',
                    'borderBottom': f'1px solid {COLORS["border"]}'
                }),
                
                # Loading wrapper
                dcc.Loading(
                    id="individual-trade-chart-loading",
                    children=[
                        dcc.Graph(
                            id='individual-trade-chart',
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': [
                                    'pan2d', 'lasso2d', 'autoScale2d'
                                ]
                            },
                            style={'height': '600px'}
                        )
                    ],
                    type="default",
                    color=COLORS['info']
                )
            ]),
            size='chart'
        )
    
    def _create_trade_summary_cards(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create trade summary KPI cards."""
        
        # Get summary statistics from trade log
        if analytics_data.trade_log is not None and len(analytics_data.trade_log) > 0:
            trade_log = analytics_data.trade_log
            
            total_trades = len(trade_log)
            winning_trades = len(trade_log[trade_log['pnl'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = trade_log['pnl'].mean()
            total_pnl = trade_log['pnl'].sum()
            
        else:
            total_trades = 0
            winning_trades = 0
            win_rate = 0
            avg_pnl = 0
            total_pnl = 0
        
        summary_cards = [
            create_kpi_card(
                title="Total Trades",
                value=f"{total_trades}",
                subtitle=f"{winning_trades} winning",
                color=COLORS['info'],
                icon="fas fa-list-ol",
                size="small"
            ),
            create_kpi_card(
                title="Win Rate",
                value=f"{win_rate:.1f}%",
                subtitle="Successful trades",
                color=COLORS['profit'] if win_rate > 50 else COLORS['caution'],
                icon="fas fa-percentage",
                size="small"
            ),
            create_kpi_card(
                title="Average P&L",
                value=f"{avg_pnl:+.2f}",
                subtitle="Per trade",
                color=COLORS['profit'] if avg_pnl > 0 else COLORS['loss'],
                icon="fas fa-chart-line",
                size="small"
            ),
            create_kpi_card(
                title="Total P&L",
                value=f"{total_pnl:+.2f}",
                subtitle="All trades",
                color=COLORS['profit'] if total_pnl > 0 else COLORS['loss'],
                icon="fas fa-wallet",
                size="small"
            )
        ]
        
        return create_card_group("Trade Summary", summary_cards)
    
    def _create_trade_metrics_panel(self, analytics_data: CachedAnalyticsData) -> html.Div:
        """Create trade metrics sidebar panel."""
        
        return create_base_card(
            html.Div([
                html.H5("Trade Details", style={
                    'color': COLORS['primary'],
                    'marginBottom': '15px',
                    'fontSize': '16px',
                    'fontWeight': '600'
                }),
                
                # Trade details will be populated by callback
                html.Div(id='individual-trade-details', children=[
                    html.P("Select a trade to view details", style={
                        'color': COLORS['tertiary'],
                        'fontSize': '14px',
                        'textAlign': 'center',
                        'margin': '20px 0'
                    })
                ])
            ]),
            size='full-width'
        )


def create_individual_trade_content(data_service: DashboardDataService, 
                                  strategy: str = None, version: str = None,
                                  start_date: datetime = None, end_date: datetime = None) -> html.Div:
    """
    Factory function to create Individual Trade tab content.
    
    Args:
        data_service: Dashboard data service instance
        strategy: Selected strategy name
        version: Selected version identifier
        start_date: Filter start date
        end_date: Filter end date
        
    Returns:
        Individual Trade tab content
    """
    individual_trade_tab = IndividualTradeTab(data_service)
    return individual_trade_tab.create_tab_content(strategy, version, start_date, end_date)


# Callback for sidebar toggle (mobile responsiveness)
@callback(
    Output("controls-sidebar-collapse", "is_open"),
    [Input("toggle-controls-sidebar", "n_clicks")],
    [State("controls-sidebar-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_controls_sidebar(n_clicks, is_open):
    """Toggle the controls sidebar for mobile devices."""
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("metrics-sidebar-collapse", "is_open"),
    [Input("toggle-metrics-sidebar", "n_clicks")],
    [State("metrics-sidebar-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_metrics_sidebar(n_clicks, is_open):
    """Toggle the metrics sidebar for mobile devices."""
    if n_clicks:
        return not is_open
    return is_open


# Main chart update callback
@callback(
    [Output('individual-trade-chart', 'figure'),
     Output('individual-trade-details', 'children')],
    [Input('individual-trade-dropdown', 'value'),
     Input('individual-bars-before-slider', 'value'),
     Input('individual-bars-after-slider', 'value'),
     Input('individual-chart-options', 'value')],
    [State('trade-data-store', 'data')],
    prevent_initial_call=True
)
def update_individual_trade_chart(selected_trade_index, bars_before, bars_after, chart_options, trade_data):
    """Update the individual trade chart based on selections."""
    
    # Generate placeholder data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.uniform(0.5, 2.0, 100)
    low_prices = close_prices - np.random.uniform(0.5, 2.0, 100)
    open_prices = close_prices + np.random.uniform(-1.0, 1.0, 100)
    volumes = np.random.uniform(10000, 50000, 100)
    
    # Determine subplot configuration based on volume option
    show_volume = chart_options and 'volume' in chart_options
    
    if show_volume:
        # Create subplots: OHLCV chart (80% height) and Volume chart (20% height) below
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.8, 0.2],
            subplot_titles=('Price', 'Volume')
        )
        
        # Add candlestick chart to the top subplot ONLY
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=open_prices,
                high=high_prices,
                low=low_prices,
                close=close_prices,
                name="Price",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add volume bars to the bottom subplot with color coding
        volume_colors = ['green' if close_prices[i] >= open_prices[i] else 'red' for i in range(len(dates))]
        
        fig.add_trace(
            go.Bar(
                x=dates,
                y=volumes,
                name="Volume",
                marker_color=volume_colors,
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
    else:
        # Create single plot for OHLCV only
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=dates,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name="Price"
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Trade {selected_trade_index + 1 if selected_trade_index is not None else 'N/A'}",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12),
        height=600,
        margin=dict(l=60, r=20, t=60, b=40),
        showlegend=False if show_volume else True,
        xaxis_rangeslider_visible=False  # This removes the smaller candlestick chart below
    )
    
    # Update axis styling - handle both single chart and subplots
    if show_volume:
        # Style axes for subplots
        fig.update_xaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False,
            row=1, col=1
        )
        fig.update_xaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False,
            title_text="Date",
            row=2, col=1
        )
        fig.update_yaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False,
            row=1, col=1
        )
        fig.update_yaxes(
            gridcolor=COLORS['border'],
            gridwidth=1,
            zeroline=False,
            row=2, col=1
        )
    else:
        # Style axes for single chart
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price ($)"
        )
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
    
    # Create trade details
    if selected_trade_index is not None:
        trade_details = html.Div([
            html.H6(f"Trade {selected_trade_index + 1}", style={'color': COLORS['primary']}),
            html.Hr(),
            html.P([
                html.Strong("Symbol: "), "DEMO"
            ], style={'margin': '5px 0'}),
            html.P([
                html.Strong("Entry: "), f"${open_prices[50]:.2f}"
            ], style={'margin': '5px 0'}),
            html.P([
                html.Strong("Exit: "), f"${close_prices[70]:.2f}"
            ], style={'margin': '5px 0'}),
            html.P([
                html.Strong("P&L: "), f"${(close_prices[70] - open_prices[50]):.2f}"
            ], style={
                'margin': '5px 0',
                'color': COLORS['profit'] if close_prices[70] > open_prices[50] else COLORS['loss'],
                'fontWeight': 'bold'
            })
        ])
    else:
        trade_details = html.P("Select a trade to view details", style={'color': COLORS['tertiary']})
    
    return fig, trade_details


# Navigation button callbacks
@callback(
    Output('individual-trade-dropdown', 'value', allow_duplicate=True),
    [Input('first-trade-btn', 'n_clicks'),
     Input('prev-trade-btn', 'n_clicks'),
     Input('next-trade-btn', 'n_clicks'),
     Input('last-trade-btn', 'n_clicks')],
    [State('individual-trade-dropdown', 'options'),
     State('individual-trade-dropdown', 'value')],
    prevent_initial_call=True
)
def navigate_trades(first_clicks, prev_clicks, next_clicks, last_clicks, options, current_value):
    """Handle trade navigation button clicks."""
    
    if not options:
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'first-trade-btn':
        return 0
    elif button_id == 'last-trade-btn':
        return len(options) - 1
    elif button_id == 'prev-trade-btn':
        if current_value is not None and current_value > 0:
            return current_value - 1
    elif button_id == 'next-trade-btn':
        if current_value is not None and current_value < len(options) - 1:
            return current_value + 1
    
    raise PreventUpdate


# Reset controls callback
@callback(
    [Output('individual-bars-before-slider', 'value'),
     Output('individual-bars-after-slider', 'value'),
     Output('individual-chart-options', 'value')],
    [Input('individual-reset-options-btn', 'n_clicks')],
    prevent_initial_call=True
)
def reset_individual_options(n_clicks):
    """Reset chart options to default values."""
    if n_clicks:
        return 50, 20, ["volume", "indicators", "risk_levels", "annotations"]
    raise PreventUpdate