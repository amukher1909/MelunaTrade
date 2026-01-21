# meluna/analysis/trade_visualization/visualizations/TradeVisualizationTab.py

"""
Individual Trade Visualization Tab Component.

This module implements the Individual Trade Visualization tab for the Meluna dashboard,
providing comprehensive single-trade analysis with candlestick charts, trade markers,
risk management levels, and context window controls.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .TradeChart import TradeChart
from ..base_interfaces import TradeVisualizationConfig, ChartAnnotation
from ..TradeDataExtractor import TradeDataExtractor
from ..calculations.TradeMetrics import TradeVisualizationMetrics, TradeMetricsCalculator
from ..calculations.PriceActionMetrics import PriceActionAnalysis, PriceActionCalculator
from ...dashboard.DashboardDataService import DashboardDataService
from ...dashboard.DashboardComponents import create_base_card, create_card_group, COLORS

logger = logging.getLogger(__name__)


class TradeVisualizationTab:
    """
    Individual Trade Visualization Tab component for the Meluna dashboard.
    
    Provides comprehensive single-trade analysis with:
    - Interactive candlestick chart with trade markers
    - Risk management level visualization
    - Trade navigation and selection controls
    - Context window adjustment
    - Technical indicator overlays
    - Trade metrics display
    """
    
    def __init__(self, data_service: DashboardDataService):
        """
        Initialize the Trade Visualization Tab.
        
        Args:
            data_service: Dashboard data service for accessing backtest results
        """
        self.data_service = data_service
        self.trade_extractor = TradeDataExtractor(data_service.results_directory)
        # Note: calculators will be initialized when needed with specific trade data
        self.metrics_calculator = None
        self.price_action_calculator = None
        
        # Default configuration
        self.config = TradeVisualizationConfig(
            chart_height=800,
            chart_width=1200,
            bars_before_entry=50,
            bars_after_exit=20,
            show_volume=True,
            show_indicators=True
        )
        
        # Current state
        self.current_trade_data = None
        self.available_trades = []
        self.current_trade_index = 0
        
        # Register callbacks for chart updates
        self._register_callbacks()
        
    def create_tab_content(self, strategy: str = None, version: str = None,
                          start_date: datetime = None, end_date: datetime = None) -> html.Div:
        """
        Create Individual Trade Visualization tab content with real data integration.
        
        Args:
            strategy: Selected strategy name
            version: Selected version
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            Dash HTML Div containing the trade visualization tab content
        """
        try:
            # Load available trades
            self._load_available_trades(strategy, version, start_date, end_date)
            
            # Create trade chart
            trade_chart_component = self._create_trade_chart_component()
            
            # Create control panels
            trade_controls = self._create_trade_controls()
            context_controls = self._create_context_controls()
            
            # Create metrics panels
            trade_metrics_panel = self._create_trade_metrics_panel()
            
            return html.Div([
                # Tab Header
                self._create_tab_header(),
                
                # Main content layout using Bootstrap container
                dbc.Container([
                    dbc.Row([
                        # Left sidebar with controls and metrics
                        dbc.Col([
                            trade_controls,
                            html.Br(),
                            context_controls,
                            html.Br(),
                            trade_metrics_panel
                        ], width=3, style={'paddingRight': '15px'}),
                        
                        # Main chart area
                        dbc.Col([
                            trade_chart_component
                        ], width=9)
                    ], className="g-0")
                ], fluid=True),
                
                # Store components for state management
                dcc.Store(id='trade-visualization-state', data={'trades_loaded': len(self.available_trades)}),
                dcc.Store(id='available-trades-store', data=self.available_trades),
                dcc.Store(id='current-trade-store', data={}),
                
            ], style={
                'padding': '20px',
                'backgroundColor': COLORS['page_bg'],
                'minHeight': 'calc(100vh - 200px)'
            })
            
        except Exception as e:
            logger.error(f"Error creating trade visualization tab content: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._create_error_content(str(e))
            
    def _create_tab_header(self) -> html.Div:
        """Create the tab header with title and description."""
        return html.Div([
            html.H2("Individual Trade Visualization", style={
                'color': COLORS['primary'],
                'marginBottom': '10px',
                'fontSize': '28px',
                'fontWeight': '600'
            }),
            html.P(
                "Detailed individual trade analysis with candlestick charts, trade markers, "
                "risk management levels, and technical context visualization.",
                style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                }
            )
        ], style={'marginBottom': '20px'})
        
    def _create_trade_controls(self) -> html.Div:
        """Create trade navigation and selection controls."""
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
                    id='trade-selection-dropdown',
                    options=[],  # Will be populated by callback
                    value=None,  # Will be set by callback
                    placeholder="Loading trades...",
                    style={'marginBottom': '15px'}
                ),
                
                # Navigation buttons
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-chevron-left"), " Previous"],
                        id='prev-trade-btn',
                        color='secondary',
                        size='sm',
                        outline=True,
                        disabled=True,
                        style={'marginRight': '10px'}
                    ),
                    dbc.Button(
                        ["Next ", html.I(className="fas fa-chevron-right")],
                        id='next-trade-btn',
                        color='secondary',
                        size='sm',
                        outline=True,
                        disabled=True
                    )
                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                
                # Trade count indicator
                html.P(
                    id='trade-count-indicator',
                    children=f"Showing {len(self.available_trades)} trades" if self.available_trades else "No trades available",
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
        
    def _create_context_controls(self) -> html.Div:
        """Create context window adjustment controls."""
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
                    id='bars-before-slider',
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
                    id='bars-after-slider',
                    min=10,
                    max=100,
                    step=5,
                    value=20,
                    marks={10: '10', 20: '20', 50: '50', 100: '100'},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),
                
                html.Br(),
                
                # Reset button
                dbc.Button(
                    "Reset to Default",
                    id='reset-context-btn',
                    color='outline-primary',
                    size='sm',
                    style={'width': '100%'}
                )
            ]),
            size='full-width'
        )
        
    def _create_trade_metrics_panel(self) -> html.Div:
        """Create trade metrics display panel."""
        return create_base_card(
            html.Div([
                html.H5("Trade Metrics", style={
                    'color': COLORS['primary'],
                    'marginBottom': '15px',
                    'fontSize': '16px',
                    'fontWeight': '600'
                }),
                
                # Metrics will be populated by callback
                html.Div(id='trade-metrics-content', children=[
                    html.P("Select a trade to view metrics", style={
                        'color': COLORS['tertiary'],
                        'fontSize': '14px',
                        'textAlign': 'center',
                        'margin': '20px 0'
                    })
                ])
            ]),
            size='full-width'
        )
        
    def _create_trade_chart_component(self) -> html.Div:
        """Create the main trade chart component."""
        return create_base_card(
            html.Div([
                # Chart container
                dcc.Loading(
                    id="trade-chart-loading",
                    children=[
                        dcc.Graph(
                            id='trade-chart',
                            figure=self._create_empty_chart(),  # Initial figure
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': [
                                    'pan2d', 'lasso2d', 'autoScale2d', 'hoverClosestCartesian'
                                ]
                            },
                            style={'height': '800px', 'width': '100%'}
                        )
                    ],
                    type="default",
                    color=COLORS['info']
                ),
                
                # Chart controls
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="fas fa-home"), " Reset Zoom"],
                            id='reset-zoom-btn',
                            color='outline-secondary',
                            size='sm'
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-download"), " Export"],
                            id='export-chart-btn',
                            color='outline-primary',
                            size='sm'
                        )
                    ])
                ], style={
                    'textAlign': 'center',
                    'marginTop': '10px'
                })
            ]),
            size='full-width'
        )
        
    def _create_error_content(self, error_message: str) -> html.Div:
        """Create error content when tab fails to load."""
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={
                    'fontSize': '48px',
                    'color': COLORS['loss'],
                    'marginBottom': '20px'
                }),
                html.H3("Error Loading Trade Visualization", style={
                    'color': COLORS['primary'],
                    'marginBottom': '15px'
                }),
                html.P(f"An error occurred while loading the trade visualization: {error_message}", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '20px'
                }),
                dbc.Button(
                    "Retry",
                    id='retry-load-btn',
                    color='primary',
                    size='lg'
                )
            ], style={
                'textAlign': 'center',
                'padding': '60px 20px'
            })
        ], style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["border"]}',
            'borderRadius': '8px',
            'margin': '20px',
            'minHeight': '400px',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center'
        })
        
    def _load_available_trades(self, strategy: str = None, version: str = None,
                              start_date: datetime = None, end_date: datetime = None) -> None:
        """Load available trades from trade logs based on filter criteria."""
        try:
            # Load real trade data from data service
            analytics_data = self.data_service.load_analytics_data(
                strategy or 'test_strategy', 
                version or 'v1'
            )
            
            if analytics_data and hasattr(analytics_data, 'trade_log') and analytics_data.trade_log is not None:
                trade_log = analytics_data.trade_log
                
                # Convert trade log to our format
                self.available_trades = []
                for index, trade in trade_log.iterrows():
                    trade_dict = {
                        'trade_id': f"TRADE_{index + 1:03d}",
                        'symbol': trade.get('symbol', 'UNKNOWN'),
                        'entry_time': trade.get('entry_timestamp', ''),
                        'exit_time': trade.get('exit_timestamp', ''),
                        'entry_price': float(trade.get('entry_price', 0)),
                        'exit_price': float(trade.get('exit_price', 0)),
                        'quantity': int(trade.get('quantity', 0)),
                        'pnl': float(trade.get('pnl', 0)),
                        'pnl_pct': float(trade.get('pnl_percent', 0))
                    }
                    self.available_trades.append(trade_dict)
                
                logger.info(f"Loaded {len(self.available_trades)} trades from trade logs")
                
                if self.available_trades:
                    self.current_trade_index = 0
                    self.current_trade_data = self.available_trades[0]
                else:
                    logger.warning("No trades found in trade logs")
                    self.available_trades = []
                    self.current_trade_data = None
            else:
                logger.warning("No trade log found in analytics data")
                self.available_trades = []
                self.current_trade_data = None
                
        except Exception as e:
            logger.error(f"Error loading trades from trade logs: {e}")
            # Fallback to empty state
            self.available_trades = []
            self.current_trade_data = None
    
    def _register_callbacks(self):
        """Register Dash callbacks for chart updates and interactions."""
        
        @callback(
            [Output('trade-selection-dropdown', 'options'),
             Output('trade-selection-dropdown', 'value')],
            [Input('trade-visualization-state', 'data')],
            prevent_initial_call=False
        )
        def update_dropdown_options(state_data):
            """Update dropdown options based on available trades."""
            if not self.available_trades:
                return [], None
            
            options = [
                {'label': f"{trade['trade_id']} - {trade['symbol']} ({trade['pnl']:+.2f})", 
                 'value': i} 
                for i, trade in enumerate(self.available_trades)
            ]
            return options, 0  # Select first trade by default
        
        @callback(
            Output('trade-chart', 'figure'),
            [Input('trade-selection-dropdown', 'value'),
             Input('bars-before-slider', 'value'),
             Input('bars-after-slider', 'value')],
            prevent_initial_call=False
        )
        def update_trade_chart(selected_trade_index, bars_before, bars_after):
            """Update the trade chart based on selected trade and context window."""
            try:
                if not self.available_trades or selected_trade_index is None:
                    return self._create_empty_chart()
                
                if selected_trade_index >= len(self.available_trades):
                    return self._create_empty_chart()
                
                # Get selected trade
                trade_info = self.available_trades[selected_trade_index]
                
                # Extract trade data with context window
                from ..TradeDataExtractor import ContextWindowConfig
                context_config = ContextWindowConfig(
                    bars_before_entry=bars_before or 50,
                    bars_after_exit=bars_after or 20
                )
                
                # Get current strategy/version from callback context or default
                strategy = getattr(self.data_service, '_current_strategy', 'test_strategy')
                version = getattr(self.data_service, '_current_version', 'v1')
                
                extraction_result = self.trade_extractor.extract_trade_data(
                    strategy=strategy,
                    version=version,
                    trade_id=str(trade_info['trade_id']),
                    symbol=trade_info['symbol'],
                    context_config=context_config
                )
                
                if not extraction_result.success:
                    logger.error(f"Failed to extract trade data: {extraction_result.error_message}")
                    return self._create_error_chart(extraction_result.error_message)
                
                # Calculate trade metrics with P&L curve
                trade_window = extraction_result.trade_window
                metrics_calc = TradeMetricsCalculator(
                    price_data=trade_window.context_ohlcv,
                    trade_data={
                        'trade_id': trade_window.trade_id,
                        'symbol': trade_window.symbol,
                        'entry_timestamp': trade_window.entry_timestamp,
                        'exit_timestamp': trade_window.exit_timestamp,
                        'entry_price': trade_window.entry_price,
                        'exit_price': trade_window.exit_price,
                        'quantity': trade_window.quantity,
                        'stop_loss': trade_info.get('stop_loss'),
                        'target': trade_info.get('target')
                    }
                )
                
                # Calculate comprehensive metrics
                trade_metrics = metrics_calc.calculate_metrics()
                
                # Create price action analysis
                price_action_calc = PriceActionCalculator(trade_window.context_ohlcv)
                price_action = price_action_calc.calculate_analysis()
                
                # Create TradeChart with enhanced P&L evolution subplot
                chart = TradeChart(self.config)
                chart.prepare_data(trade_metrics, price_action)
                figure = chart.create_plot()
                
                logger.info(f"Successfully created chart for trade {trade_window.trade_id} with P&L evolution")
                return figure
                
            except Exception as e:
                logger.error(f"Error updating trade chart: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return self._create_error_chart(f"Chart update failed: {str(e)}")
    
    def _create_empty_chart(self):
        """Create an empty placeholder chart."""
        fig = go.Figure()
        fig.add_annotation(
            text="Select a trade to view chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    def _create_error_chart(self, error_message: str):
        """Create an error chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color='red')
        )
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig


def create_trade_visualization_content(data_service: DashboardDataService,
                                     strategy: str = None, version: str = None,
                                     start_date: datetime = None, end_date: datetime = None) -> html.Div:
    """
    Factory function to create Individual Trade Visualization tab content.
    
    Args:
        data_service: Dashboard data service instance
        strategy: Selected strategy name
        version: Selected version
        start_date: Filter start date
        end_date: Filter end date
        
    Returns:
        Dash HTML Div containing the Individual Trade Visualization tab content
    """
    trade_viz_tab = TradeVisualizationTab(data_service)
    return trade_viz_tab.create_tab_content(strategy, version, start_date, end_date)