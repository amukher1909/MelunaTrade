"""
Enhanced Trading Analytics Dashboard - Main Application

This module creates and runs the professional trading analytics dashboard,
integrating the core layout components with Plotly Dash framework and
providing the main application entry point.

Features:
- Master layout with fixed header and navigation
- Strategy and version selection with callbacks
- Date range filtering
- Responsive design for desktop and laptop screens
- Professional styling and color scheme
- Integration with existing backtest data structure
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import our dashboard components
from meluna.analysis.dashboard.DashboardComponents import (
    create_header_bar,
    create_navigation_tabs,
    create_main_layout,
    create_tab_specific_content,
    get_available_strategies,
    get_available_versions,
    COLORS,
    DASHBOARD_CSS
)
from meluna.analysis.dashboard.DashboardDataService import DashboardDataService
from meluna.analysis.dashboard.PortfolioOverviewTab import create_portfolio_overview_content
from meluna.analysis.dashboard.TradeAnalyticsTab import create_trade_analytics_content
from meluna.analysis.dashboard.RiskAnalysisTab import create_risk_analysis_content
from meluna.analysis.dashboard.IndividualTradeTab import create_individual_trade_content
from meluna.analysis.trade_visualization.visualizations.TradeVisualizationTab import create_trade_visualization_content
from meluna.analysis.dashboard.InteractiveDashboardManager import InteractiveDashboardManager, create_interactive_dashboard_manager
from meluna.analysis.dashboard.InteractiveComponents import create_css_enhancements

logger = logging.getLogger(__name__)


class EnhancedTradingDashboard:
    """
    Main dashboard application class that orchestrates the layout,
    callbacks, and data management for the trading analytics dashboard.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the dashboard application.
        
        Args:
            results_dir: Path to directory containing backtest results
        """
        self.results_dir = results_dir
        self.data_service = DashboardDataService(results_dir)
        self.interactive_manager = create_interactive_dashboard_manager(self.data_service)
        self.app = dash.Dash(
            __name__,
            title="Meluna Trading Analytics Dashboard",
            suppress_callback_exceptions=True,
            external_stylesheets=[
                'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap',
                'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap',
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
            ]
        )
        
        # Inject custom CSS
        self.app.index_string = f"""
        <!DOCTYPE html>
        <html>
            <head>
                {{%metas%}}
                <title>{{%title%}}</title>
                {{%favicon%}}
                {{%css%}}
                <style>
                    {DASHBOARD_CSS}
                    {create_css_enhancements()}
                </style>
            </head>
            <body>
                {{%app_entry%}}
                <footer>
                    {{%config%}}
                    {{%scripts%}}
                    {{%renderer%}}
                </footer>
            </body>
        </html>
        """
        
        self._setup_layout()
        self._setup_callbacks()
        self._setup_interactive_callbacks()
    
    def _setup_layout(self):
        """Set up the main application layout."""
        self.app.layout = html.Div([
            # Store components for data management and state persistence
            dcc.Store(id='current-data-store'),
            dcc.Store(id='strategy-data-store'),
            dcc.Store(id='tab-state-store', data={'current_tab': 'portfolio-overview'}),
            
            # Interactive state stores
            *self.interactive_manager.create_interaction_stores(),
            
            # Header bar
            create_header_bar(self.results_dir),
            
            # Navigation tabs
            create_navigation_tabs(),
            
            # Main content area
            html.Div(id='main-content'),
            
            # Modal components for interactive features
            *self.interactive_manager.create_modal_components(),
            
            # Hidden div for triggering callbacks
            html.Div(id='hidden-div', style={'display': 'none'})
            
        ], style={
            'margin': '0',
            'padding': '0',
            'fontFamily': 'Inter, sans-serif',
            'backgroundColor': COLORS['page_bg']
        })
    
    def _setup_callbacks(self):
        """Set up all dashboard callbacks for interactivity."""
        
        @self.app.callback(
            Output('version-dropdown', 'options'),
            Output('version-dropdown', 'value'),
            Input('strategy-dropdown', 'value'),
            prevent_initial_call=False
        )
        def update_version_dropdown(selected_strategy):
            """Update version dropdown based on selected strategy."""
            if not selected_strategy:
                return [], None
            
            versions = get_available_versions(selected_strategy, self.results_dir)
            options = [{'label': version, 'value': version} for version in versions]
            value = versions[-1] if versions else None  # Select latest version
            
            return options, value
        
        @self.app.callback(
            Output('date-range-picker', 'start_date'),
            Output('date-range-picker', 'end_date'),
            [Input('strategy-dropdown', 'value'),
             Input('version-dropdown', 'value')],
            prevent_initial_call=True
        )
        def update_date_range(selected_strategy, selected_version):
            """Update date range based on available data."""
            if not selected_strategy or not selected_version:
                raise PreventUpdate
            
            try:
                # Try to load equity curve data to get date range
                data_path = Path(self.results_dir) / selected_strategy / selected_version / 'equity_curve.parquet'
                if data_path.exists():
                    df = pd.read_parquet(data_path)
                    if 'Date' in df.columns:
                        start_date = df['Date'].min()
                        end_date = df['Date'].max()
                        return start_date, end_date
            except Exception as e:
                print(f"Error loading date range: {e}")
            
            # Fallback to default range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            return start_date, end_date
        
        @self.app.callback(
            [Output('active-tab-store', 'data'),
             Output('tab-button-portfolio-overview', 'style'),
             Output('tab-button-trade-analytics', 'style'),
             Output('tab-button-individual-trades', 'style'),
             Output('tab-button-risk-analysis', 'style')],
            [Input('tab-button-portfolio-overview', 'n_clicks'),
             Input('tab-button-trade-analytics', 'n_clicks'),
             Input('tab-button-individual-trades', 'n_clicks'),
             Input('tab-button-risk-analysis', 'n_clicks')],
            prevent_initial_call=False
        )
        def handle_horizontal_tab_clicks(portfolio_clicks, trade_clicks, individual_clicks, risk_clicks):
            """Handle horizontal tab button clicks and update active states."""
            from dash import callback_context
            
            # Default styles for inactive tabs
            inactive_style = {
                'backgroundColor': COLORS['background'],
                'color': COLORS['primary'],
                'border': f'2px solid {COLORS["border"]}',
                'borderRadius': '8px',
                'padding': '12px 24px',
                'margin': '0 8px',
                'minWidth': '180px',
                'height': '50px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                'fontFamily': 'Inter, sans-serif',
                'cursor': 'pointer',
                'textTransform': 'none'
            }
            
            # Active style
            active_style = {
                **inactive_style,
                'backgroundColor': COLORS['info'],
                'color': 'white',
                'border': f'2px solid {COLORS["info"]}',
                'boxShadow': '0 4px 12px rgba(0, 123, 255, 0.25)'
            }
            
            # Determine which button was clicked
            if callback_context.triggered:
                button_id = callback_context.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'tab-button-portfolio-overview':
                    return ('portfolio-overview', 
                           active_style, inactive_style, inactive_style, inactive_style)
                elif button_id == 'tab-button-trade-analytics':
                    return ('trade-analytics',
                           inactive_style, active_style, inactive_style, inactive_style)
                elif button_id == 'tab-button-individual-trades':
                    return ('individual-trades',
                           inactive_style, inactive_style, active_style, inactive_style)
                elif button_id == 'tab-button-risk-analysis':
                    return ('risk-analysis',
                           inactive_style, inactive_style, inactive_style, active_style)
            
            # Default: portfolio overview active
            return ('portfolio-overview',
                   active_style, inactive_style, inactive_style, inactive_style)
        
        @self.app.callback(
            Output('tab-state-store', 'data'),
            Input('active-tab-store', 'data'),
            State('tab-state-store', 'data'),
            prevent_initial_call=False
        )
        def update_tab_state(active_tab, current_state):
            """Update tab state for persistence across navigation."""
            if current_state is None:
                current_state = {}
            current_state['current_tab'] = active_tab or 'portfolio-overview'
            return current_state
        
        @self.app.callback(
            Output('main-content', 'children'),
            [Input('active-tab-store', 'data'),
             Input('strategy-dropdown', 'value'),
             Input('version-dropdown', 'value'),
             Input('date-range-picker', 'start_date'),
             Input('date-range-picker', 'end_date'),
             Input('hidden-div', 'children')],  # Add trigger for initial load
            State('tab-state-store', 'data'),
            prevent_initial_call=False
        )
        def update_main_content(active_tab, selected_strategy, selected_version, start_date, end_date, hidden_div_trigger, tab_state):
            """Update main content area based on tab selection and filters with state persistence."""
            
            # Debug logging
            print(f"[DEBUG] update_main_content called with:")
            print(f"  - active_tab: {active_tab}")
            print(f"  - selected_strategy: {selected_strategy}")
            print(f"  - selected_version: {selected_version}")
            print(f"  - start_date: {start_date}")
            print(f"  - end_date: {end_date}")
            print(f"  - tab_state: {tab_state}")
            
            # Use stored tab state if no active tab provided, default to portfolio-overview
            current_tab = active_tab or 'portfolio-overview'
            if not current_tab and tab_state:
                current_tab = tab_state.get('current_tab', 'portfolio-overview')
            
            # Convert date strings to datetime objects if provided
            parsed_start_date = None
            parsed_end_date = None
            if start_date:
                try:
                    parsed_start_date = pd.to_datetime(start_date)
                except (ValueError, TypeError):
                    parsed_start_date = None
            if end_date:
                try:
                    parsed_end_date = pd.to_datetime(end_date)
                except (ValueError, TypeError):
                    parsed_end_date = None
            
            # Return tab-specific content with real data integration
            tab = current_tab or 'portfolio-overview'
            
            if tab == 'portfolio-overview':
                return create_portfolio_overview_content(
                    self.data_service, 
                    selected_strategy, 
                    selected_version,
                    parsed_start_date,
                    parsed_end_date
                )
            elif tab == 'trade-analytics':
                return create_trade_analytics_content(
                    selected_strategy, 
                    selected_version,
                    parsed_start_date,
                    parsed_end_date
                )
            elif tab == 'individual-trades':
                return create_individual_trade_content(
                    self.data_service,
                    selected_strategy, 
                    selected_version,
                    parsed_start_date,
                    parsed_end_date
                )
            elif tab == 'risk-analysis':
                return create_risk_analysis_content(
                    selected_strategy, 
                    selected_version,
                    parsed_start_date,
                    parsed_end_date
                )
            else:
                # For other tabs, use existing placeholder content
                return create_tab_specific_content(tab)
        
        # Note: interactive-equity-curve callback is handled by the InteractiveEquityCurve module
        # to avoid duplicate callback outputs. The component manages its own updates.

        @self.app.callback(
            Output('placeholder-chart', 'figure'),
            [Input('strategy-dropdown', 'value'),
             Input('version-dropdown', 'value'),
             Input('date-range-picker', 'start_date'),
             Input('date-range-picker', 'end_date')],
            prevent_initial_call=True
        )
        def update_placeholder_chart(selected_strategy, selected_version, start_date, end_date):
            """Update placeholder chart with sample equity curve."""
            
            # Create sample data for demonstration
            dates = pd.date_range(
                start=start_date if start_date else '2024-01-01',
                end=end_date if end_date else '2024-12-31',
                freq='D'
            )
            
            # Generate sample equity curve
            import numpy as np
            np.random.seed(42)  # For consistent demo data
            returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual return, 20% volatility
            equity_curve = 100000 * np.cumprod(1 + returns)
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add equity curve
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity_curve,
                mode='lines',
                name='Portfolio Value',
                line=dict(color=COLORS['info'], width=2),
                hovertemplate='<b>%{y:₹,.0f}</b><br>%{x}<extra></extra>'
            ))
            
            # Add benchmark (simulated)
            benchmark = 100000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, len(dates)))
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark,
                mode='lines',
                name='Benchmark',
                line=dict(color=COLORS['tertiary'], width=2, dash='dash'),
                hovertemplate='<b>%{y:₹,.0f}</b><br>%{x}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f'Equity Curve - {selected_strategy or "Demo Strategy"} {selected_version or "v1"}',
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
            
            return fig
    
    def _setup_interactive_callbacks(self):
        """Set up interactive dashboard callbacks."""
        # TEMPORARILY DISABLED to fix dropdown rendering issues
        logger.info("Interactive callbacks temporarily disabled to fix dropdown issues")
        return
        
        try:
            # Set up all interactive callbacks through the manager
            self.interactive_manager.setup_interactive_callbacks(self.app)
            logger.info("Interactive callbacks setup completed successfully")
        except Exception as e:
            logger.warning(f"Interactive callbacks setup encountered issues: {e}")
            logger.info("Dashboard will continue with basic functionality")
            # Continue without advanced interactive features if setup fails
    
    def run(self, debug: bool = True, host: str = '127.0.0.1', port: int = 9999):
        """
        Run the dashboard application.
        
        Args:
            debug: Enable debug mode
            host: Host address
            port: Port number
        """
        print(f"\n{'='*60}")
        print("MELUNA ENHANCED TRADING DASHBOARD")
        print(f"{'='*60}")
        print(f"\nStarting dashboard server...")
        print(f"URL: http://{host}:{port}")
        print(f"Results directory: {self.results_dir}")
        print(f"Available strategies: {len(get_available_strategies(self.results_dir))}")
        print(f"\nEnhanced Features:")
        print("[+] Fixed header bar with controls")
        print("[+] Strategy and version selection")
        print("[+] Date range picker")
        print("[+] Navigation tabs (Portfolio/Trade/Risk)")
        print("[+] Responsive grid layout")
        print("[+] Professional styling")
        print("[+] Interactive charts with cross-chart linking")
        print("[+] Hover synchronization across time series")
        print("[+] Click-to-filter functionality")
        print("[+] KPI card drill-down capabilities")
        print("[+] Modal detail views")
        print("[+] Rich context-aware tooltips")
        print(f"\nPress Ctrl+C to stop the server")
        print(f"{'='*60}\n")
        
        try:
            self.app.run(
                debug=debug,
                host=host,
                port=port,
                dev_tools_ui=debug,
                dev_tools_props_check=debug
            )
        except KeyboardInterrupt:
            print("\nDashboard stopped by user")
        except Exception as e:
            print(f"\nError running dashboard: {e}")
            raise


def create_dashboard(results_dir: str = "results") -> EnhancedTradingDashboard:
    """
    Factory function to create and configure the dashboard.
    
    Args:
        results_dir: Path to directory containing backtest results
        
    Returns:
        Configured EnhancedTradingDashboard instance
    """
    return EnhancedTradingDashboard(results_dir=results_dir)


def validate_data_structure(results_dir: str = "results") -> Dict[str, any]:
    """
    Validate the data structure and return status information.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary containing validation results
    """
    results_path = Path(results_dir)
    validation = {
        'directory_exists': results_path.exists(),
        'strategies': [],
        'total_versions': 0,
        'valid_data_files': 0,
        'errors': []
    }
    
    if not validation['directory_exists']:
        validation['errors'].append(f"Results directory '{results_dir}' does not exist")
        return validation
    
    try:
        strategies = get_available_strategies(results_dir)
        validation['strategies'] = strategies
        
        for strategy in strategies:
            versions = get_available_versions(strategy, results_dir)
            validation['total_versions'] += len(versions)
            
            for version in versions:
                version_path = results_path / strategy / version
                
                # Check for required files
                required_files = ['trade_log.parquet', 'equity_curve.parquet']
                for file_name in required_files:
                    file_path = version_path / file_name
                    if file_path.exists():
                        validation['valid_data_files'] += 1
                    else:
                        validation['errors'].append(
                            f"Missing {file_name} in {strategy}/{version}"
                        )
    
    except Exception as e:
        validation['errors'].append(f"Error validating data structure: {e}")
    
    return validation


if __name__ == "__main__":
    """
    Direct execution for testing the dashboard.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Meluna Trading Analytics Dashboard')
    parser.add_argument('--results-dir', default='results', 
                        help='Path to results directory (default: results)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Host address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9985,
                        help='Port number (default: 9999)')
    parser.add_argument('--no-debug', action='store_true',
                        help='Disable debug mode')
    parser.add_argument('--validate', action='store_true',
                        help='Validate data structure and exit')
    
    args = parser.parse_args()
    
    if args.validate:
        print("Validating data structure...")
        validation = validate_data_structure(args.results_dir)
        
        print(f"\nValidation Results:")
        print(f"Directory exists: {validation['directory_exists']}")
        print(f"Strategies found: {len(validation['strategies'])}")
        print(f"Total versions: {validation['total_versions']}")
        print(f"Valid data files: {validation['valid_data_files']}")
        
        if validation['errors']:
            print(f"\nErrors found ({len(validation['errors'])}):")
            for error in validation['errors']:
                print(f"  - {error}")
        else:
            print("\n[SUCCESS] No errors found")
        
        sys.exit(0)
    
    # Create and run dashboard
    dashboard = create_dashboard(args.results_dir)
    dashboard.run(
        debug=not args.no_debug,
        host=args.host,
        port=args.port
    )