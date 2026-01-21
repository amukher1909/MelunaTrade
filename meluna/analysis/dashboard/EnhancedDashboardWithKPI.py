# meluna/analysis/EnhancedDashboardWithKPI.py

"""
Enhanced Trading Analytics Dashboard with Real KPI Cards Integration

This module creates a comprehensive dashboard that integrates real KPI cards
with live data from PortfolioMetrics and TradeAnalyzer modules, providing
executive-level performance insights with interactive controls.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

# Add project root to Python path if needed
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate

from .DashboardComponents import (
    create_header_bar, create_navigation_tabs, create_main_layout,
    create_real_kpi_cards_from_data, create_loading_kpi_cards,
    COLORS, DASHBOARD_CSS
)
from .DashboardDataService import DashboardDataService, CachedAnalyticsData, LoadingState
from .KPICards import KPICardGenerator, KPICardRenderer

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class EnhancedKPIDashboard:
    """
    Enhanced dashboard with real KPI cards integration providing comprehensive
    trading analytics with executive-level summary views.
    """
    
    def __init__(self, results_directory: str = "results", 
                 app_name: str = "MELUNA Analytics Dashboard",
                 host: str = "127.0.0.1", port: int = 9999):
        """
        Initialize the enhanced KPI dashboard.
        
        Args:
            results_directory: Path to backtest results directory
            app_name: Application name for browser title
            host: Server host address
            port: Server port number
        """
        self.results_directory = results_directory
        self.app_name = app_name
        self.host = host
        self.port = port
        
        # Initialize data service
        self.data_service = DashboardDataService(
            results_directory=results_directory,
            cache_size=10,
            enable_threading=True
        )
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://use.fontawesome.com/releases/v5.15.4/css/all.css",
                "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
            ],
            title=app_name,
            update_title="Loading...",
            suppress_callback_exceptions=True
        )
        
        # Set up layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info(f"Enhanced KPI Dashboard initialized for {results_directory}")
    
    def _setup_layout(self):
        """Set up the main dashboard layout."""
        self.app.layout = html.Div([
            # CSS injection
            html.Style(DASHBOARD_CSS),
            
            # Data stores for state management
            dcc.Store(id='current-analytics-data', storage_type='session'),
            dcc.Store(id='loading-state', storage_type='memory'),
            dcc.Store(id='error-state', storage_type='memory'),
            
            # Header bar with controls
            create_header_bar(self.results_directory),
            
            # Navigation tabs
            create_navigation_tabs(),
            
            # Main content area
            html.Div(id='dashboard-main-content', children=[
                create_loading_kpi_cards()  # Default loading state
            ], style={
                'marginTop': '90px',
                'backgroundColor': COLORS['page_bg'],
                'minHeight': 'calc(100vh - 90px)'
            }),
            
            # Footer
            html.Div([
                html.P([
                    "Powered by MELUNA Analytics Engine | ",
                    html.A("Documentation", href="#", style={'color': COLORS['info']}),
                    " | ",
                    html.A("Support", href="#", style={'color': COLORS['info']})
                ], style={
                    'textAlign': 'center',
                    'color': COLORS['tertiary'],
                    'fontSize': '12px',
                    'margin': '0',
                    'padding': '20px'
                })
            ], style={
                'backgroundColor': COLORS['background'],
                'borderTop': f'1px solid {COLORS["border"]}'
            })
            
        ], id='main-dashboard-container')
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks for interactivity."""
        
        @self.app.callback(
            Output('version-dropdown', 'options'),
            Output('version-dropdown', 'value'),
            Input('strategy-dropdown', 'value'),
            prevent_initial_call=True
        )
        def update_version_dropdown(selected_strategy):
            """Update version dropdown based on selected strategy."""
            if not selected_strategy:
                return [], None
            
            try:
                strategies = self.data_service.discover_strategies()
                versions = strategies.get(selected_strategy, [])
                
                options = [{'label': version, 'value': version} for version in versions]
                default_value = versions[-1] if versions else None  # Latest version
                
                return options, default_value
            except Exception as e:
                logger.error(f"Error updating version dropdown: {e}")
                return [], None
        
        @self.app.callback(
            Output('date-range-picker', 'start_date'),
            Output('date-range-picker', 'end_date'),
            Input('strategy-dropdown', 'value'),
            Input('version-dropdown', 'value'),
            prevent_initial_call=True
        )
        def update_date_range(selected_strategy, selected_version):
            """Update date range picker based on selected strategy/version."""
            if not selected_strategy or not selected_version:
                return None, None
            
            try:
                version_info = self.data_service.get_strategy_version_info(
                    selected_strategy, selected_version
                )
                
                if version_info:
                    return version_info.start_date, version_info.end_date
                else:
                    return None, None
            except Exception as e:
                logger.error(f"Error updating date range: {e}")
                return None, None
        
        @self.app.callback(
            Output('current-analytics-data', 'data'),
            Output('loading-state', 'data'),
            Output('error-state', 'data'),
            Input('strategy-dropdown', 'value'),
            Input('version-dropdown', 'value'),
            Input('date-range-picker', 'start_date'),
            Input('date-range-picker', 'end_date'),
            prevent_initial_call=True
        )
        def load_analytics_data(selected_strategy, selected_version, start_date, end_date):
            """Load analytics data when strategy/version/dates change."""
            if not selected_strategy or not selected_version:
                return None, {'is_loading': False}, None
            
            try:
                # Update loading state
                loading_state = {'is_loading': True, 'progress': 0.1, 'message': 'Loading analytics data...'}
                
                # Parse dates if provided
                date_range = None
                if start_date and end_date:
                    date_range = (pd.to_datetime(start_date), pd.to_datetime(end_date))
                
                # Load analytics data
                analytics_data = self.data_service.load_analytics_data(
                    strategy=selected_strategy,
                    version=selected_version,
                    date_range=date_range,
                    force_reload=False
                )
                
                if analytics_data:
                    # Convert to serializable format
                    data_dict = {
                        'strategy': analytics_data.strategy,
                        'version': analytics_data.version,
                        'portfolio_metrics': analytics_data.portfolio_metrics,
                        'trade_metrics': analytics_data.trade_metrics.__dict__ if analytics_data.trade_metrics else None,
                        'cache_timestamp': analytics_data.cache_timestamp.isoformat(),
                        'has_equity_curve': analytics_data.equity_curve is not None,
                        'has_trade_log': analytics_data.trade_log is not None,
                        'equity_curve_length': len(analytics_data.equity_curve) if analytics_data.equity_curve is not None else 0,
                        'trade_log_length': len(analytics_data.trade_log) if analytics_data.trade_log is not None else 0
                    }
                    
                    # Store the actual analytics_data object for KPI generation
                    # Note: We'll need to recreate this in the KPI callback
                    
                    return data_dict, {'is_loading': False, 'progress': 1.0, 'message': 'Data loaded successfully'}, None
                else:
                    error_msg = f"Failed to load data for {selected_strategy}/{selected_version}"
                    return None, {'is_loading': False}, {'error': error_msg}
                    
            except Exception as e:
                error_msg = f"Error loading analytics data: {str(e)}"
                logger.error(error_msg)
                return None, {'is_loading': False}, {'error': error_msg}
        
        @self.app.callback(
            Output('dashboard-main-content', 'children'),
            Input('main-tabs', 'active_tab'),
            Input('current-analytics-data', 'data'),
            Input('loading-state', 'data'),
            Input('error-state', 'data'),
            State('strategy-dropdown', 'value'),
            State('version-dropdown', 'value'),
            prevent_initial_call=False
        )
        def update_dashboard_content(active_tab, analytics_data_dict, loading_state, 
                                   error_state, selected_strategy, selected_version):
            """Update dashboard content based on active tab and data state."""
            
            # Handle loading state
            if loading_state and loading_state.get('is_loading', False):
                return create_loading_kpi_cards()
            
            # Handle error state
            if error_state and error_state.get('error'):
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle", style={
                            'fontSize': '48px',
                            'color': COLORS['caution'],
                            'marginBottom': '20px'
                        }),
                        html.H3("Data Loading Error", style={
                            'color': COLORS['primary'],
                            'marginBottom': '15px'
                        }),
                        html.P(error_state['error'], style={
                            'color': COLORS['tertiary'],
                            'textAlign': 'center',
                            'fontSize': '16px'
                        }),
                        html.Button("Retry", id='retry-button', style={
                            'backgroundColor': COLORS['info'],
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'marginTop': '20px'
                        })
                    ], style={
                        'textAlign': 'center',
                        'padding': '60px',
                        'backgroundColor': COLORS['background'],
                        'borderRadius': '8px',
                        'border': f'1px solid {COLORS["border"]}',
                        'maxWidth': '500px',
                        'margin': '0 auto'
                    })
                ], style={
                    'padding': '60px 20px',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'minHeight': '400px'
                })
            
            # Handle case with no data
            if not analytics_data_dict or not selected_strategy or not selected_version:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-line", style={
                            'fontSize': '64px',
                            'color': COLORS['border'],
                            'marginBottom': '30px'
                        }),
                        html.H3("Select Strategy and Version", style={
                            'color': COLORS['primary'],
                            'marginBottom': '15px'
                        }),
                        html.P("Choose a strategy and version from the dropdown controls above to view performance metrics.", style={
                            'color': COLORS['tertiary'],
                            'textAlign': 'center',
                            'fontSize': '16px',
                            'lineHeight': '1.6'
                        })
                    ], style={
                        'textAlign': 'center',
                        'padding': '80px 40px',
                        'backgroundColor': COLORS['background'],
                        'borderRadius': '8px',
                        'border': f'1px solid {COLORS["border"]}',
                        'maxWidth': '600px',
                        'margin': '0 auto'
                    })
                ], style={
                    'padding': '60px 20px',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'minHeight': '400px'
                })
            
            try:
                # Recreate analytics data for KPI generation
                # This is a simplified recreation - in production, we'd want a more robust approach
                analytics_data = type('CachedAnalyticsData', (), {
                    'strategy': analytics_data_dict['strategy'],
                    'version': analytics_data_dict['version'],
                    'portfolio_metrics': analytics_data_dict['portfolio_metrics'],
                    'trade_metrics': type('TradeMetrics', (), analytics_data_dict['trade_metrics']) if analytics_data_dict['trade_metrics'] else None,
                    'equity_curve': None,  # Would need to reload for full functionality
                    'trade_log': None,     # Would need to reload for full functionality
                    'cache_timestamp': pd.to_datetime(analytics_data_dict['cache_timestamp'])
                })()
                
                # Create KPI cards based on active tab
                if active_tab == 'portfolio-overview':
                    return create_real_kpi_cards_from_data(
                        analytics_data, 
                        ['performance', 'risk']
                    )
                elif active_tab == 'trade-analytics':
                    return create_real_kpi_cards_from_data(
                        analytics_data, 
                        ['trade_efficiency', 'statistical']
                    )
                elif active_tab == 'risk-analysis':
                    return create_real_kpi_cards_from_data(
                        analytics_data, 
                        ['risk', 'statistical']
                    )
                else:
                    # Default: show all KPI categories
                    return create_real_kpi_cards_from_data(analytics_data)
                    
            except Exception as e:
                logger.error(f"Error creating dashboard content: {e}")
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-bug", style={
                            'fontSize': '48px',
                            'color': COLORS['caution'],
                            'marginBottom': '20px'
                        }),
                        html.H3("Content Generation Error", style={
                            'color': COLORS['primary'],
                            'marginBottom': '15px'
                        }),
                        html.P(f"Error generating dashboard content: {str(e)}", style={
                            'color': COLORS['tertiary'],
                            'textAlign': 'center',
                            'fontSize': '14px'
                        })
                    ], style={
                        'textAlign': 'center',
                        'padding': '40px',
                        'backgroundColor': COLORS['background'],
                        'borderRadius': '8px',
                        'border': f'1px solid {COLORS["border"]}'
                    })
                ], style={
                    'padding': '40px',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center'
                })
    
    def run(self, debug: bool = False, dev_tools_hot_reload: bool = True):
        """
        Run the enhanced KPI dashboard server.
        
        Args:
            debug: Enable debug mode
            dev_tools_hot_reload: Enable hot reload for development
        """
        try:
            logger.info(f"Starting Enhanced KPI Dashboard server...")
            logger.info(f"Dashboard will be available at: http://{self.host}:{self.port}")
            
            # Health check
            health_status = self.data_service.health_check()
            if health_status['status'] != 'healthy':
                logger.warning(f"Data service health check: {health_status['status']}")
                for issue in health_status.get('issues', []):
                    logger.warning(f"  - {issue}")
            
            # Cache stats
            cache_stats = self.data_service.get_cache_stats()
            logger.info(f"Cache initialized: {cache_stats['analytics_cache_size']} entries")
            
            # Start server
            self.app.run_server(
                host=self.host,
                port=self.port,
                debug=debug,
                dev_tools_hot_reload=dev_tools_hot_reload,
                dev_tools_silence_routes_logging=not debug
            )
            
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")
            raise
    
    def get_app(self):
        """Get the Dash app instance for external deployment."""
        return self.app


def main():
    """Main entry point for running the enhanced KPI dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MELUNA Enhanced KPI Analytics Dashboard")
    parser.add_argument('--results', '-r', default='results', 
                       help='Path to results directory (default: results)')
    parser.add_argument('--host', default='127.0.0.1', 
                       help='Host address (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', type=int, default=9999, 
                       help='Port number (default: 9999)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode')
    parser.add_argument('--no-reload', action='store_true', 
                       help='Disable hot reload')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run dashboard
    dashboard = EnhancedKPIDashboard(
        results_directory=args.results,
        host=args.host,
        port=args.port
    )
    
    dashboard.run(
        debug=args.debug,
        dev_tools_hot_reload=not args.no_reload
    )


if __name__ == "__main__":
    main()