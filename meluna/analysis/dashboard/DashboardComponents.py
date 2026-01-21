"""
Professional Trading Analytics Dashboard - Core Layout Components

This module provides the foundational layout structure and components for the
Meluna trading analytics dashboard, implementing the master layout architecture
as defined in the dashboard design plan.

Key Components:
- Fixed header bar with strategy/version/date controls
- Navigation tabs structure 
- Responsive grid layout for main content area
- Professional styling following design color scheme
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to Python path if needed
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate

# Import KPI Cards functionality
# TODO: KPICards.py file missing - need to create or restore this file
from .KPICards import KPICardGenerator, KPICardRenderer, KPICardData
from .DashboardDataService import DashboardDataService, CachedAnalyticsData

# Professional Financial Color Palette
COLORS = {
    'primary': '#0D1B2A',      # Deep Navy Blue - Headers, primary text
    'secondary': '#1B2951',    # Slate Blue - Secondary elements
    'tertiary': '#415A77',     # Steel Gray - Tertiary elements
    'profit': '#00C851',       # Profit Green - Positive performance
    'loss': '#FF4444',         # Loss Red - Negative performance
    'caution': '#FF8800',      # Caution Orange - Warnings, neutral
    'info': '#007BFF',         # Insight Blue - Information, benchmarks
    'background': '#FFFFFF',   # Pure White - Card backgrounds
    'page_bg': '#F8F9FA',      # Light Gray - Page background
    'border': '#E9ECEF'        # Warm Gray - Dividers, borders
}

# CSS Styles
HEADER_STYLE = {
    'backgroundColor': COLORS['primary'],
    'color': 'white',
    'padding': '15px 30px',
    'position': 'fixed',
    'top': '0',
    'left': '0',
    'right': '0',
    'zIndex': '1000',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
    'fontFamily': 'Inter, sans-serif',
    'fontSize': '16px'
}

TAB_STYLE = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '12px 24px',
    'fontWeight': '500',
    'backgroundColor': COLORS['background'],
    'color': COLORS['primary'],
    'cursor': 'pointer',
    'transition': 'all 0.3s ease'
}

TAB_SELECTED_STYLE = {
    'borderTop': f'3px solid {COLORS["info"]}',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': COLORS['page_bg'],
    'color': COLORS['primary'],
    'padding': '12px 24px',
    'fontWeight': '600'
}

CONTENT_STYLE = {
    'marginTop': '0px',  # No margin since tabs are positioned below header
    'padding': '20px',
    'backgroundColor': COLORS['page_bg'],
    'minHeight': 'calc(100vh - 180px)'  # Account for header (90px) + horizontal tabs (90px)
}

CARD_STYLE = {
    'backgroundColor': COLORS['background'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '8px',
    'padding': '20px',
    'margin': '10px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
    'transition': 'box-shadow 0.3s ease'
}

# Enhanced Grid System Styles
GRID_CONTAINER_STYLE = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(auto-fit, minmax(300px, 1fr))',
    'gap': '20px',
    'padding': '20px 0'
}

# Card Grid System with Named Areas
CARD_GRID_STYLE = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(12, 1fr)',
    'gridGap': '20px',
    'padding': '20px',
    'gridAutoRows': 'minmax(120px, auto)'
}

# Card Size Categories
CARD_SIZES = {
    'small': {
        'gridColumn': 'span 3',
        'gridRow': 'span 1',
        'minHeight': '120px'
    },
    'medium': {
        'gridColumn': 'span 6',
        'gridRow': 'span 1',
        'minHeight': '120px'
    },
    'large': {
        'gridColumn': 'span 6',
        'gridRow': 'span 2',
        'minHeight': '280px'
    },
    'full-width': {
        'gridColumn': 'span 12',
        'gridRow': 'span 1',
        'minHeight': '120px'
    },
    'chart': {
        'gridColumn': 'span 12',
        'gridRow': 'span 3',
        'minHeight': '400px'
    }
}


def get_available_strategies(results_dir: str = "results") -> List[str]:
    """
    Get list of available trading strategies from results directory.
    
    Args:
        results_dir: Path to results directory containing backtest data
        
    Returns:
        List of strategy names (directory names)
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    
    strategies = []
    for item in results_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            strategies.append(item.name)
    
    return sorted(strategies)


def get_available_versions(strategy: str, results_dir: str = "results") -> List[str]:
    """
    Get list of available versions for a specific strategy.
    
    Args:
        strategy: Strategy name
        results_dir: Path to results directory
        
    Returns:
        List of version names (directory names starting with 'v')
    """
    if not strategy:
        return []
    
    strategy_path = Path(results_dir) / strategy
    if not strategy_path.exists():
        return []
    
    versions = []
    for item in strategy_path.iterdir():
        if item.is_dir() and item.name.startswith('v'):
            versions.append(item.name)
    
    # Sort versions numerically (v1, v2, v10, etc.)
    try:
        versions.sort(key=lambda x: int(x[1:]))
    except ValueError:
        versions.sort()
    
    return versions


def create_header_bar(results_dir: str = "results") -> html.Div:
    """
    Create the fixed header bar with brand, strategy selector, version selector,
    and date range picker.
    
    Args:
        results_dir: Path to results directory for populating dropdowns
        
    Returns:
        Dash HTML Div containing the header components
    """
    strategies = get_available_strategies(results_dir)
    
    # Create dropdown options
    dropdown_options = [{'label': strategy, 'value': strategy} for strategy in strategies]
    dropdown_value = strategies[0] if strategies else None
    
    return html.Div([
        html.Div([
            # Brand/Logo section
            html.Div([
                html.H2("MELUNA", style={
                    'margin': '0',
                    'fontWeight': '700',
                    'fontSize': '24px',
                    'letterSpacing': '1px'
                }),
                html.Span("ANALYTICS", style={
                    'fontSize': '12px',
                    'letterSpacing': '2px',
                    'opacity': '0.8'
                })
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start'}),
            
            # Controls section
            html.Div([
                # Strategy Selector
                html.Div([
                    html.Label("Strategy:", style={
                        'marginRight': '10px',
                        'fontSize': '14px',
                        'fontWeight': '500'
                    }),
                    dcc.Dropdown(
                        id='strategy-dropdown',
                        options=dropdown_options,
                        value=dropdown_value,
                        placeholder="Select Strategy",
                        searchable=False,  # Disable search to prevent "No results found" issue
                        clearable=False,   # Prevent clearing the selection
                        style={
                            'minWidth': '200px',
                            'backgroundColor': 'white',
                            'borderRadius': '4px'
                        },
                        className='custom-dropdown'
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '30px'}),
                
                # Version Selector
                html.Div([
                    html.Label("Version:", style={
                        'marginRight': '10px',
                        'fontSize': '14px',
                        'fontWeight': '500'
                    }),
                    dcc.Dropdown(
                        id='version-dropdown',
                        placeholder="Select Version",
                        style={
                            'minWidth': '120px',
                            'backgroundColor': 'white',
                            'borderRadius': '4px'
                        },
                        className='custom-dropdown'
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '30px'}),
                
                # Date Range Picker
                html.Div([
                    html.Label("Date Range:", style={
                        'marginRight': '10px',
                        'fontSize': '14px',
                        'fontWeight': '500'
                    }),
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        start_date_placeholder_text="Start Date",
                        end_date_placeholder_text="End Date",
                        style={
                            'backgroundColor': 'white',
                            'borderRadius': '4px',
                            'minWidth': '240px'  # Ensure consistent sizing
                        },
                        className='custom-date-picker'
                    )
                ], style={'display': 'flex', 'alignItems': 'center'})
                
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'flexWrap': 'wrap',
                'gap': '20px'
            })
            
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'width': '100%',
            'maxWidth': '1400px',
            'margin': '0 auto'
        })
    ], style=HEADER_STYLE)


def create_navigation_tabs() -> html.Div:
    """
    Create horizontal button-style navigation tabs for Portfolio Overview, Trade Analytics, and Risk Analysis.
    Implements modern button design positioned horizontally at the top of the content area.
    
    Returns:
        Dash HTML Div containing the horizontal tab navigation
    """
    # Tab button configurations
    tabs_config = [
        {
            'id': 'portfolio-overview',
            'label': 'Portfolio Overview',
            'icon': 'fas fa-chart-pie',
            'description': 'Performance metrics and portfolio analysis'
        },
        {
            'id': 'trade-analytics', 
            'label': 'Trade Analytics',
            'icon': 'fas fa-chart-bar',
            'description': 'Trade-level analysis and execution metrics'
        },
        {
            'id': 'individual-trades',
            'label': 'Individual Trades',
            'icon': 'fas fa-search-plus',
            'description': 'Detailed individual trade visualization and analysis'
        },
        {
            'id': 'risk-analysis',
            'label': 'Risk Analysis', 
            'icon': 'fas fa-shield-alt',
            'description': 'Risk assessment and downside protection'
        }
    ]
    
    return html.Div([
        # Horizontal navigation container
        html.Div([
            # Tab buttons container - horizontal layout
            html.Div(
                # Create individual horizontal tab buttons arranged side-by-side
                [
                    dbc.Button(
                        [
                            html.I(className=tab['icon'], style={
                                'marginRight': '8px',
                                'fontSize': '16px'
                            }),
                            html.Span(tab['label'], style={
                                'fontSize': '15px',
                                'fontWeight': '600'
                            })
                        ],
                        id=f"tab-button-{tab['id']}",
                        className='horizontal-tab-button',
                        n_clicks=0,
                        color='light',
                        outline=True,
                        style={
                            'backgroundColor': COLORS['background'] if tab['id'] != 'portfolio-overview' else COLORS['info'],
                            'color': COLORS['primary'] if tab['id'] != 'portfolio-overview' else 'white',
                            'border': f'2px solid {COLORS["border"]}' if tab['id'] != 'portfolio-overview' else f'2px solid {COLORS["info"]}',
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
                        },
                        title=tab['description']
                    ) for tab in tabs_config
                ],
                style={
                    'display': 'flex',
                    'flexDirection': 'row',  # Explicitly set to row for side-by-side layout
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'padding': '20px 30px',
                    'backgroundColor': COLORS['page_bg'],
                    'borderBottom': f'1px solid {COLORS["border"]}',
                    'gap': '15px',
                    'flexWrap': 'nowrap'  # Prevent wrapping to keep tabs on same line
                }
            ),
            
            # Hidden store to track active tab
            dcc.Store(id='active-tab-store', data='portfolio-overview')
        ], style={
            'marginTop': '90px',  # Account for fixed header
            'position': 'sticky',
            'top': '90px',
            'zIndex': '999',
            'backgroundColor': COLORS['page_bg'],
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
        })
    ])


def create_base_card(content: html.Div, size: str = 'medium', card_type: str = 'default', 
                    loading: bool = False, empty: bool = False) -> html.Div:
    """
    Create a base card component with consistent styling and responsive behavior.
    
    Args:
        content: HTML content to display in the card
        size: Card size category ('small', 'medium', 'large', 'full-width', 'chart')
        card_type: Type of card for specific styling ('kpi', 'chart', 'text', 'default')
        loading: Whether to show loading state
        empty: Whether to show empty state
        
    Returns:
        Dash HTML Div containing the styled card
    """
    # Get size configuration
    size_config = CARD_SIZES.get(size, CARD_SIZES['medium'])
    
    # Loading state
    if loading:
        content = html.Div([
            html.Div(className='loading-placeholder', style={
                'height': '20px',
                'borderRadius': '4px',
                'marginBottom': '10px'
            }),
            html.Div(className='loading-placeholder', style={
                'height': '40px',
                'borderRadius': '4px',
                'marginBottom': '10px'
            }),
            html.Div(className='loading-placeholder', style={
                'height': '15px',
                'borderRadius': '4px',
                'width': '60%'
            })
        ])
    
    # Empty state
    if empty:
        content = html.Div([
            html.I(className="fas fa-chart-line", style={
                'fontSize': '48px',
                'color': COLORS['border'],
                'marginBottom': '10px'
            }),
            html.P("No data available", style={
                'color': COLORS['tertiary'],
                'fontSize': '14px',
                'margin': '0'
            })
        ], style={'textAlign': 'center'})
    
    # Card type specific styling
    card_classes = ['dashboard-card']
    if card_type == 'kpi':
        card_classes.append('kpi-card')
    elif card_type == 'chart':
        card_classes.append('chart-card')
    
    return html.Div(
        content,
        className=' '.join(card_classes),
        style={
            **CARD_STYLE,
            **size_config,
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center' if card_type == 'kpi' else 'flex-start'
        }
    )


def create_kpi_card(title: str, value: str, subtitle: str = "", 
                   color: str = COLORS['primary'], icon: str = "", 
                   size: str = 'small', trend: Optional[str] = None,
                   loading: bool = False) -> html.Div:
    """
    Create an enhanced KPI card component for displaying key performance indicators.
    
    Args:
        title: Card title/metric name
        value: Primary metric value to display
        subtitle: Additional context or comparison
        color: Color for the primary value
        icon: Optional icon class
        size: Card size ('small', 'medium', 'large')
        trend: Trend direction ('up', 'down', 'neutral')
        loading: Whether to show loading state
        
    Returns:
        Dash HTML Div containing the enhanced KPI card
    """
    # Trend icon
    trend_icon = None
    if trend == 'up':
        trend_icon = html.I(className="fas fa-arrow-up", style={
            'color': COLORS['profit'], 'fontSize': '14px', 'marginLeft': '8px'
        })
    elif trend == 'down':
        trend_icon = html.I(className="fas fa-arrow-down", style={
            'color': COLORS['loss'], 'fontSize': '14px', 'marginLeft': '8px'
        })
    elif trend == 'neutral':
        trend_icon = html.I(className="fas fa-minus", style={
            'color': COLORS['caution'], 'fontSize': '14px', 'marginLeft': '8px'
        })
    
    content = html.Div([
        # Header with icon and title
        html.Div([
            html.I(className=icon, style={
                'fontSize': '16px',
                'color': COLORS['tertiary'],
                'marginRight': '8px'
            }) if icon else None,
            html.H4(title, style={
                'margin': '0',
                'fontSize': clamp('12px', '2vw', '14px') if size == 'small' else '14px',
                'fontWeight': '500',
                'color': COLORS['tertiary'],
                'textTransform': 'uppercase',
                'letterSpacing': '0.5px',
                'lineHeight': '1.2'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'marginBottom': '12px'
        }),
        
        # Main value with trend
        html.Div([
            html.H2(value, style={
                'margin': '0',
                'fontSize': clamp('24px', '4vw', '32px') if size == 'small' else '32px',
                'fontWeight': '700',
                'color': color,
                'fontFamily': 'JetBrains Mono, monospace',
                'lineHeight': '1'
            }),
            trend_icon
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'marginBottom': '8px'
        }),
        
        # Subtitle
        html.P(subtitle, style={
            'margin': '0',
            'fontSize': clamp('10px', '1.5vw', '12px') if size == 'small' else '12px',
            'color': COLORS['tertiary'],
            'textAlign': 'center',
            'lineHeight': '1.3'
        }) if subtitle else None
    ], style={'textAlign': 'center'})
    
    return create_base_card(content, size, 'kpi', loading)


def clamp(min_val: str, preferred: str, max_val: str) -> str:
    """Helper function to create CSS clamp() values for responsive typography."""
    return f"clamp({min_val}, {preferred}, {max_val})"


def create_chart_container(chart_id: str, title: str, height: str = "400px",
                          size: str = 'chart', loading: bool = False) -> html.Div:
    """
    Create an enhanced container for Plotly charts with consistent styling.
    
    Args:
        chart_id: Unique ID for the chart component
        title: Chart title
        height: Chart height (CSS value)
        size: Card size category
        loading: Whether to show loading state
        
    Returns:
        Dash HTML Div containing the enhanced chart container
    """
    content = html.Div([
        # Chart header with title and controls
        html.Div([
            html.H3(title, style={
                'margin': '0',
                'fontSize': '18px',
                'fontWeight': '600',
                'color': COLORS['primary'],
                'flex': '1'
            }),
            # Chart controls (export, fullscreen, etc.)
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
        
        # Chart area
        dcc.Graph(
            id=chart_id,
            style={'height': height}
        )
    ])
    
    return create_base_card(content, size, 'chart', loading)


def create_card_grid(cards: List[html.Div], grid_areas: Optional[Dict] = None) -> html.Div:
    """
    Create a responsive card grid container with organized layout.
    
    Args:
        cards: List of card components to display
        grid_areas: Optional dictionary defining named grid areas
        
    Returns:
        Dash HTML Div containing the card grid
    """
    grid_style = CARD_GRID_STYLE.copy()
    
    # Add named grid areas if provided
    if grid_areas:
        grid_style['gridTemplateAreas'] = grid_areas.get('template', '')
    
    return html.Div(
        cards,
        style=grid_style,
        className='card-grid-container'
    )


def create_card_group(title: str, cards: List[html.Div], 
                     collapsible: bool = False) -> html.Div:
    """
    Create a grouped section of cards with optional title and collapsible behavior.
    
    Args:
        title: Group title
        cards: List of cards in this group
        collapsible: Whether the group can be collapsed
        
    Returns:
        Dash HTML Div containing the card group
    """
    return html.Div([
        # Group header
        html.Div([
            html.H3(title, style={
                'margin': '0',
                'fontSize': '20px',
                'fontWeight': '600',
                'color': COLORS['primary'],
                'flex': '1'
            }),
            html.Button([
                html.I(className="fas fa-chevron-up")
            ], style={
                'background': 'none',
                'border': 'none',
                'color': COLORS['tertiary'],
                'cursor': 'pointer',
                'padding': '8px',
                'fontSize': '12px'
            }) if collapsible else None
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'marginBottom': '20px',
            'paddingBottom': '10px',
            'borderBottom': f'2px solid {COLORS["border"]}'
        }),
        
        # Card grid
        create_card_grid(cards)
    ], style={
        'marginBottom': '40px'
    }, className='card-group')


def create_main_layout() -> html.Div:
    """
    Create the enhanced main dashboard layout with responsive card grid system.
    
    Returns:
        Dash HTML Div containing the main layout structure with card grid
    """
    return html.Div([
        # Main content container with card grid
        html.Div(id='main-content-grid', className='main-dashboard-grid')
        
    ], style=CONTENT_STYLE)


def create_real_kpi_cards_from_data(analytics_data: CachedAnalyticsData, 
                                   card_categories: List[str] = None) -> html.Div:
    """
    Create real KPI cards from analytics data using the KPICardGenerator.
    
    Args:
        analytics_data: CachedAnalyticsData containing portfolio and trade metrics
        card_categories: List of card categories to include ['performance', 'risk', 'trade_efficiency', 'statistical']
        
    Returns:
        Dash HTML Div containing KPI card groups with real data
    """
    if card_categories is None:
        card_categories = ['performance', 'risk', 'trade_efficiency', 'statistical']
    
    try:
        # TODO: Temporarily disabled until KPICards.py is restored
        # Initialize KPI card generator with real data
        # kpi_generator = KPICardGenerator(analytics_data)
        
        # Generate all KPI cards
        # all_cards = kpi_generator.generate_all_kpi_cards()
        all_cards = {}
        
        # Create card groups
        card_groups = []
        
        group_configs = {
            'performance': {
                'title': 'Performance Metrics',
                'description': 'Core performance indicators showing strategy profitability and efficiency'
            },
            'risk': {
                'title': 'Risk Analysis',
                'description': 'Comprehensive risk assessment and downside protection metrics'
            },
            'trade_efficiency': {
                'title': 'Trade Efficiency',
                'description': 'Trade-level analysis showing execution quality and alpha generation'
            },
            'statistical': {
                'title': 'Statistical Analysis',
                'description': 'Distribution characteristics and statistical significance testing'
            }
        }
        
        for category in card_categories:
            if category in all_cards and all_cards[category]:
                config = group_configs.get(category, {'title': category.title(), 'description': ''})
                
                # Add description header
                if config['description']:
                    description_header = html.P(config['description'], style={
                        'color': COLORS['tertiary'],
                        'fontSize': '14px',
                        'marginBottom': '20px',
                        'fontStyle': 'italic'
                    })
                else:
                    description_header = None
                
                # Create card group
                card_group = html.Div([
                    description_header,
                    # TODO: Temporarily disabled until KPICards.py is restored
                    # KPICardRenderer.render_kpi_card_group(
                    #     title=config['title'],
                    #     cards=all_cards[category],
                    #     card_size='small'
                    # )
                    html.Div(f"TODO: {config['title']} KPI cards need to be implemented", 
                            style={'padding': '20px', 'text-align': 'center'})
                ])
                card_groups.append(card_group)
        
        # Create main container
        return html.Div([
            # Header section
            html.Div([
                html.H2("Key Performance Indicators", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px',
                    'fontSize': '28px',
                    'fontWeight': '600'
                }),
                html.P(f"Real-time metrics calculated from {analytics_data.strategy}/{analytics_data.version} backtest data", style={
                    'color': COLORS['tertiary'],
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '20px'}),
            
            # KPI card groups
            html.Div(card_groups, id='kpi-cards-container')
            
        ], style={
            'padding': '20px',
            'backgroundColor': COLORS['page_bg']
        })
        
    except Exception as e:
        logger.error(f"Error creating KPI cards from data: {e}")
        
        # Return error state
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={
                    'fontSize': '48px',
                    'color': COLORS['caution'],
                    'marginBottom': '20px'
                }),
                html.H3("Error Loading KPI Cards", style={
                    'color': COLORS['primary'],
                    'marginBottom': '10px'
                }),
                html.P(f"Unable to generate KPI cards from analytics data: {str(e)}", style={
                    'color': COLORS['tertiary'],
                    'textAlign': 'center'
                }),
                html.P("Please check data availability and try again.", style={
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


def create_loading_kpi_cards() -> html.Div:
    """
    Create loading state KPI cards with skeleton placeholders.
    
    Returns:
        Dash HTML Div containing loading KPI cards
    """
    loading_cards = []
    
    # Create skeleton cards for each category
    categories = [
        ('Performance Metrics', 4),
        ('Risk Analysis', 4), 
        ('Trade Efficiency', 6),
        ('Statistical Analysis', 4)
    ]
    
    for category_name, card_count in categories:
        skeleton_cards = []
        
        for i in range(card_count):
            skeleton_card = html.Div([
                # Skeleton header
                html.Div(style={
                    'height': '16px',
                    'backgroundColor': '#E0E0E0',
                    'borderRadius': '4px',
                    'marginBottom': '12px',
                    'width': '70%',
                    'animation': 'loading 1.5s infinite'
                }),
                
                # Skeleton main value
                html.Div(style={
                    'height': '32px',
                    'backgroundColor': '#E0E0E0',
                    'borderRadius': '4px',
                    'marginBottom': '8px',
                    'width': '90%',
                    'animation': 'loading 1.5s infinite'
                }),
                
                # Skeleton subtitle
                html.Div(style={
                    'height': '12px',
                    'backgroundColor': '#E0E0E0',
                    'borderRadius': '4px',
                    'width': '60%',
                    'animation': 'loading 1.5s infinite'
                })
            ], style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px',
                'padding': '20px',
                'gridColumn': 'span 3',
                'gridRow': 'span 1',
                'minHeight': '140px',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center',
                'alignItems': 'center'
            })
            skeleton_cards.append(skeleton_card)
        
        # Create category group
        category_group = html.Div([
            html.H3(category_name, style={
                'margin': '0',
                'fontSize': '20px',
                'fontWeight': '600',
                'color': COLORS['primary'],
                'marginBottom': '20px',
                'paddingBottom': '10px',
                'borderBottom': f'2px solid {COLORS["border"]}'
            }),
            html.Div(
                skeleton_cards,
                style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(12, 1fr)',
                    'gap': '20px',
                    'gridAutoRows': 'minmax(140px, auto)'
                }
            )
        ], style={'marginBottom': '40px'})
        
        loading_cards.append(category_group)
    
    return html.Div([
        html.Div([
            html.H2("Loading Performance Metrics...", style={
                'color': COLORS['primary'],
                'marginBottom': '10px',
                'fontSize': '28px',
                'fontWeight': '600'
            }),
            html.P("Calculating real-time analytics from backtest data", style={
                'color': COLORS['tertiary'],
                'fontSize': '16px',
                'marginBottom': '30px',
                'lineHeight': '1.6'
            })
        ], style={'marginBottom': '20px'}),
        
        html.Div(loading_cards)
        
    ], style={
        'padding': '20px',
        'backgroundColor': COLORS['page_bg']
    })


def create_tab_specific_content(tab_value: str) -> html.Div:
    """
    Create enhanced tab-specific content using the responsive card grid system.
    
    Args:
        tab_value: Selected tab ID
        
    Returns:
        Dash HTML Div with enhanced tab-specific content and responsive card layouts
    """
    # Enhanced tab-specific content configurations
    content_configs = {
        'portfolio-overview': {
            'title': 'Portfolio Overview',
            'description': 'Comprehensive portfolio performance metrics and key indicators',
            'kpi_cards': [
                {'title': 'Total Return', 'value': '+24.7%', 'subtitle': 'vs Benchmark: +8.3%', 
                 'color': COLORS['profit'], 'icon': 'fas fa-chart-line', 'trend': 'up', 'size': 'small'},
                {'title': 'Sharpe Ratio', 'value': '1.85', 'subtitle': 'Risk-adjusted returns', 
                 'color': COLORS['info'], 'icon': 'fas fa-balance-scale', 'trend': 'up', 'size': 'small'},
                {'title': 'Max Drawdown', 'value': '-8.3%', 'subtitle': 'Recovery: 45 days', 
                 'color': COLORS['loss'], 'icon': 'fas fa-arrow-down', 'trend': 'neutral', 'size': 'small'},
                {'title': 'Volatility', 'value': '12.4%', 'subtitle': 'Annualized', 
                 'color': COLORS['caution'], 'icon': 'fas fa-wave-square', 'trend': 'neutral', 'size': 'small'}
            ],
            'chart_title': 'Portfolio Equity Curve',
            'secondary_cards': [
                {'title': 'CAGR', 'value': '18.3%', 'subtitle': 'Compound Annual Growth', 
                 'color': COLORS['profit'], 'size': 'small'},
                {'title': 'Alpha', 'value': '7.2%', 'subtitle': 'vs Market', 
                 'color': COLORS['info'], 'size': 'small'}
            ]
        },
        'trade-analytics': {
            'title': 'Trade Analytics',
            'description': 'Detailed trade-level performance and efficiency analysis',
            'kpi_cards': [
                {'title': 'Win Rate', 'value': '67.2%', 'subtitle': '142 of 211 trades', 
                 'color': COLORS['profit'], 'icon': 'fas fa-trophy', 'trend': 'up', 'size': 'small'},
                {'title': 'Profit Factor', 'value': '1.94', 'subtitle': 'Gross profit/loss', 
                 'color': COLORS['info'], 'icon': 'fas fa-calculator', 'trend': 'up', 'size': 'small'},
                {'title': 'Avg Win', 'value': '+2.3%', 'subtitle': 'Per winning trade', 
                 'color': COLORS['profit'], 'icon': 'fas fa-arrow-up', 'trend': 'up', 'size': 'small'},
                {'title': 'Avg Loss', 'value': '-1.2%', 'subtitle': 'Per losing trade', 
                 'color': COLORS['loss'], 'icon': 'fas fa-arrow-down', 'trend': 'down', 'size': 'small'}
            ],
            'chart_title': 'MFE/MAE Analysis (Maximum Favorable/Adverse Excursion)',
            'secondary_cards': [
                {'title': 'Total Trades', 'value': '211', 'subtitle': 'Executed', 
                 'color': COLORS['primary'], 'size': 'small'},
                {'title': 'Expectancy', 'value': '+₹142', 'subtitle': 'Per trade', 
                 'color': COLORS['profit'], 'size': 'small'}
            ]
        },
        'risk-analysis': {
            'title': 'Risk Analysis',
            'description': 'Comprehensive risk assessment and exposure analysis',
            'kpi_cards': [
                {'title': 'VaR (95%)', 'value': '-3.2%', 'subtitle': 'Daily Value at Risk', 
                 'color': COLORS['loss'], 'icon': 'fas fa-exclamation-triangle', 'trend': 'neutral', 'size': 'small'},
                {'title': 'CVaR (95%)', 'value': '-4.8%', 'subtitle': 'Conditional VaR', 
                 'color': COLORS['loss'], 'icon': 'fas fa-shield-alt', 'trend': 'neutral', 'size': 'small'},
                {'title': 'Beta', 'value': '0.87', 'subtitle': 'Market correlation', 
                 'color': COLORS['info'], 'icon': 'fas fa-link', 'trend': 'neutral', 'size': 'small'},
                {'title': 'Tail Ratio', 'value': '1.31', 'subtitle': '95th/5th percentile', 
                 'color': COLORS['caution'], 'icon': 'fas fa-chart-area', 'trend': 'up', 'size': 'small'}
            ],
            'chart_title': 'Drawdown Analysis and Risk Metrics',
            'secondary_cards': [
                {'title': 'Skewness', 'value': '0.42', 'subtitle': 'Return distribution', 
                 'color': COLORS['info'], 'size': 'small'},
                {'title': 'Kurtosis', 'value': '2.8', 'subtitle': 'Tail risk', 
                 'color': COLORS['caution'], 'size': 'small'}
            ]
        },
        'individual-trades': {
            'title': 'Individual Trade Visualization',
            'description': 'Detailed individual trade analysis with candlestick charts and technical context',
            'kpi_cards': [
                {'title': 'Current Trade', 'value': 'TRADE_001', 'subtitle': 'Selected for analysis', 
                 'color': COLORS['info'], 'icon': 'fas fa-search-plus', 'trend': 'neutral', 'size': 'small'},
                {'title': 'Entry Price', 'value': '₹1,245.50', 'subtitle': 'Buy signal executed', 
                 'color': COLORS['profit'], 'icon': 'fas fa-arrow-up', 'trend': 'up', 'size': 'small'},
                {'title': 'Exit Price', 'value': '₹1,289.25', 'subtitle': 'Target achieved', 
                 'color': COLORS['profit'], 'icon': 'fas fa-bullseye', 'trend': 'up', 'size': 'small'},
                {'title': 'Trade P&L', 'value': '+3.5%', 'subtitle': '₹4,375 profit', 
                 'color': COLORS['profit'], 'icon': 'fas fa-chart-line', 'trend': 'up', 'size': 'small'}
            ],
            'chart_title': 'Trade Analysis Chart',
            'secondary_cards': [
                {'title': 'MFE', 'value': '+5.2%', 'subtitle': 'Max Favorable Excursion', 
                 'color': COLORS['profit'], 'size': 'small'},
                {'title': 'MAE', 'value': '-1.8%', 'subtitle': 'Max Adverse Excursion', 
                 'color': COLORS['loss'], 'size': 'small'}
            ]
        }
    }
    
    config = content_configs.get(tab_value, content_configs['portfolio-overview'])
    
    # Create KPI cards with enhanced styling
    kpi_cards = [
        create_kpi_card(
            title=kpi['title'],
            value=kpi['value'],
            subtitle=kpi['subtitle'],
            color=kpi['color'],
            icon=kpi.get('icon', ''),
            size=kpi.get('size', 'small'),
            trend=kpi.get('trend', None)
        ) for kpi in config['kpi_cards']
    ]
    
    # Create secondary KPI cards if available
    secondary_cards = []
    if 'secondary_cards' in config:
        secondary_cards = [
            create_kpi_card(
                title=card['title'],
                value=card['value'],
                subtitle=card['subtitle'],
                color=card['color'],
                size=card.get('size', 'small')
            ) for card in config['secondary_cards']
        ]
    
    # Create main chart
    main_chart = create_chart_container(
        "placeholder-chart", 
        config['chart_title'], 
        "400px",
        'chart'
    )
    
    return html.Div([
        # Tab Header Section
        html.Div([
            html.H2(config['title'], style={
                'color': COLORS['primary'],
                'marginBottom': '10px',
                'fontSize': '28px',
                'fontWeight': '600'
            }),
            html.P(config['description'], style={
                'color': COLORS['tertiary'],
                'fontSize': '16px',
                'marginBottom': '30px',
                'lineHeight': '1.6'
            })
        ], style={'marginBottom': '20px'}),
        
        # Primary KPI Cards Group
        create_card_group("Key Performance Indicators", kpi_cards),
        
        # Main Chart Section
        html.Div([main_chart], style={'marginBottom': '30px'}),
        
        # Secondary KPI Cards Group (if available)
        create_card_group("Additional Metrics", secondary_cards) if secondary_cards else None,
        
        # Coming Soon Section
        create_base_card(
            html.Div([
                html.H4(f"{config['title']} - Advanced Features", style={
                    'color': COLORS['primary'],
                    'marginBottom': '15px',
                    'fontSize': '18px',
                    'fontWeight': '600'
                }),
                html.P(f"Advanced {config['title'].lower()} features will be implemented in Phase 2.", 
                       style={'color': COLORS['tertiary'], 'marginBottom': '20px'}),
                html.Div([
                    html.Span("Coming Soon", style={
                        'backgroundColor': COLORS['info'],
                        'color': 'white',
                        'padding': '4px 12px',
                        'borderRadius': '12px',
                        'fontSize': '12px',
                        'fontWeight': '500',
                        'marginRight': '8px'
                    }),
                    html.Span("Interactive Charts", style={
                        'backgroundColor': COLORS['secondary'],
                        'color': 'white',
                        'padding': '4px 12px',
                        'borderRadius': '12px',
                        'fontSize': '12px',
                        'fontWeight': '500',
                        'marginRight': '8px'
                    }),
                    html.Span("Real-time Data", style={
                        'backgroundColor': COLORS['primary'],
                        'color': 'white',
                        'padding': '4px 12px',
                        'borderRadius': '12px',
                        'fontSize': '12px',
                        'fontWeight': '500'
                    })
                ])
            ]),
            'full-width'
        )
        
    ], style=CONTENT_STYLE)


# Enhanced CSS with Responsive Card Grid System
DASHBOARD_CSS = """
/* CSS Custom Properties for Consistent Theming */
:root {
    --primary-color: #0D1B2A;
    --secondary-color: #1B2951;
    --tertiary-color: #415A77;
    --profit-color: #00C851;
    --loss-color: #FF4444;
    --caution-color: #FF8800;
    --info-color: #007BFF;
    --background-color: #FFFFFF;
    --page-bg-color: #F8F9FA;
    --border-color: #E9ECEF;
    --card-shadow: 0 2px 4px rgba(0,0,0,0.05);
    --card-shadow-hover: 0 8px 25px rgba(0,0,0,0.1);
    --border-radius: 8px;
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 12px;
    --spacing-lg: 20px;
    --spacing-xl: 30px;
}

/* Enhanced Card Grid System */
.card-grid-container {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: var(--spacing-lg);
    padding: var(--spacing-lg);
    grid-auto-rows: minmax(120px, auto);
}

/* Responsive Grid Breakpoints */
@media (max-width: 1366px) {
    .card-grid-container {
        grid-template-columns: repeat(8, 1fr);
        gap: var(--spacing-md);
        padding: var(--spacing-md);
    }
    
    .dashboard-card[style*="span 3"] {
        grid-column: span 2 !important;
    }
    
    .dashboard-card[style*="span 6"] {
        grid-column: span 4 !important;
    }
}

@media (max-width: 1024px) {
    .card-grid-container {
        grid-template-columns: repeat(6, 1fr);
        gap: var(--spacing-md);
    }
    
    .dashboard-card[style*="span 3"] {
        grid-column: span 3 !important;
    }
    
    .dashboard-card[style*="span 6"] {
        grid-column: span 6 !important;
    }
}

@media (max-width: 768px) {
    .card-grid-container {
        grid-template-columns: repeat(2, 1fr);
        gap: var(--spacing-sm);
        padding: var(--spacing-sm);
    }
    
    .dashboard-card {
        grid-column: span 2 !important;
        min-height: 100px !important;
    }
    
    .dashboard-card[style*="span 12"] {
        grid-column: span 2 !important;
    }
}

@media (max-width: 480px) {
    .card-grid-container {
        grid-template-columns: 1fr;
        gap: var(--spacing-sm);
    }
    
    .dashboard-card {
        grid-column: span 1 !important;
    }
}

/* Enhanced Card Styling */
.dashboard-card {
    background: var(--background-color);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    box-shadow: var(--card-shadow);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
    position: relative;
}

.dashboard-card:hover {
    box-shadow: var(--card-shadow-hover);
    transform: translateY(-2px);
    border-color: var(--info-color);
}

/* KPI Card Specific Styling */
.kpi-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    padding: var(--spacing-lg);
}

.kpi-card:hover {
    background: linear-gradient(135deg, var(--background-color) 0%, #fbfcfd 100%);
}

/* Chart Card Specific Styling */
.chart-card {
    padding: var(--spacing-lg);
}

.chart-card .plotly {
    height: 100% !important;
}

/* Card Groups */
.card-group {
    margin-bottom: var(--spacing-xl);
}

.card-group h3 {
    font-size: clamp(18px, 2.5vw, 20px);
    color: var(--primary-color);
    margin-bottom: var(--spacing-lg);
    font-weight: 600;
}

/* Responsive Typography */
.dashboard-card h2 {
    font-size: clamp(24px, 4vw, 32px);
    font-weight: 700;
    line-height: 1;
    margin: 0;
}

.dashboard-card h4 {
    font-size: clamp(12px, 2vw, 14px);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    line-height: 1.2;
    margin: 0;
}

.dashboard-card p {
    font-size: clamp(10px, 1.5vw, 12px);
    line-height: 1.3;
    margin: 0;
}

/* Loading States */
.loading-placeholder {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
    border-radius: 4px;
}

@keyframes loading {
    0% {
        background-position: 200% 0;
    }
    100% {
        background-position: -200% 0;
    }
}

/* Custom dropdown styling */
.custom-dropdown .Select-control {
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    background-color: var(--background-color) !important;
}

.custom-dropdown .Select-value-label {
    color: var(--primary-color) !important;
    font-weight: 500 !important;
}

/* Dropdown menu options styling - Fix for white-on-white text issue */
.custom-dropdown .Select-menu-outer {
    background-color: var(--background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    z-index: 1000 !important;
}

.custom-dropdown .Select-menu {
    background-color: var(--background-color) !important;
    border-radius: 4px !important;
}

.custom-dropdown .Select-option {
    background-color: var(--background-color) !important;
    color: #333333 !important; /* Dark text for readability */
    font-weight: 500 !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
    border-radius: 0 !important;
}

.custom-dropdown .Select-option:hover,
.custom-dropdown .Select-option.is-focused {
    background-color: #f8f9fa !important; /* Light gray hover */
    color: #0D1B2A !important; /* Primary dark color */
}

.custom-dropdown .Select-option.is-selected {
    background-color: var(--info-color) !important;
    color: white !important;
}

.custom-dropdown .Select-option.is-selected:hover {
    background-color: #0056b3 !important; /* Darker blue on hover when selected */
    color: white !important;
}

.custom-dropdown .Select-placeholder {
    color: #6c757d !important; /* Gray placeholder text */
    font-weight: 400 !important;
}

.custom-dropdown .Select-input > input {
    color: #333333 !important; /* Dark search text */
}

/* Modern React Select styling for newer Dash versions */
.custom-dropdown .css-1okebmr-indicatorSeparator {
    background-color: var(--border-color) !important;
}

.custom-dropdown .css-1hwfws3 {
    background-color: var(--background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
}

.custom-dropdown .css-1pahdxg-control {
    background-color: var(--background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    min-height: 38px !important;
}

.custom-dropdown .css-1pahdxg-control:hover {
    border-color: var(--info-color) !important;
}

.custom-dropdown .css-26l3qy-menu {
    background-color: var(--background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

.custom-dropdown .css-1n7v3ny-option {
    background-color: var(--background-color) !important;
    color: #333333 !important;
    font-weight: 500 !important;
}

.custom-dropdown .css-1n7v3ny-option:hover {
    background-color: #f8f9fa !important;
    color: #0D1B2A !important;
}

.custom-dropdown .css-1gl4k7y {
    color: #333333 !important;
}

/* Date Range Picker styling to match dropdowns */
.DateInput {
    background-color: var(--background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    min-width: 120px !important;
    height: 38px !important;
}

.DateInput_input {
    background-color: var(--background-color) !important;
    color: #333333 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    border: none !important;
    border-radius: 4px !important;
    height: 36px !important;
    line-height: 20px !important;
}

.DateInput_input::placeholder {
    color: #6c757d !important;
    font-weight: 400 !important;
}

.DateRangePickerInput {
    background-color: var(--background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    display: flex !important;
    align-items: center !important;
    height: 38px !important;
}

.DateRangePickerInput:hover {
    border-color: var(--info-color) !important;
}

.DateRangePickerInput_arrow {
    color: var(--border-color) !important;
}

.DayPicker {
    background-color: var(--background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    color: #333333 !important; /* Fix white text on white background */
}

.DayPicker_weekHeader {
    color: #333333 !important;
    font-weight: 600 !important;
}

.DayPicker_weekHeader_ul {
    color: #333333 !important;
}

.DayPicker_weekHeader_li {
    color: #333333 !important;
}

.CalendarDay__default {
    background-color: var(--background-color) !important;
    color: #333333 !important;
    border: 1px solid transparent !important;
}

.CalendarDay__default:hover {
    background-color: #f8f9fa !important;
    color: #0D1B2A !important;
    border-color: var(--info-color) !important;
}

.CalendarDay__selected,
.CalendarDay__selected:hover {
    background-color: var(--info-color) !important;
    color: white !important;
    border-color: var(--info-color) !important;
}

.CalendarDay__selected_span {
    background-color: rgba(0, 123, 255, 0.1) !important;
    color: #0D1B2A !important;
}

/* Additional calendar text elements */
.DayPicker_caption {
    color: #333333 !important;
    font-weight: 600 !important;
}

.DayPicker_caption_div {
    color: #333333 !important;
}

.DayPickerNavigation_button {
    color: #333333 !important;
    background-color: transparent !important;
    border: none !important;
}

.DayPickerNavigation_button:hover {
    background-color: #f8f9fa !important;
    color: var(--info-color) !important;
}

.DayPicker_month {
    color: #333333 !important;
}

.DayPicker_weekdays {
    color: #333333 !important;
}

.DayPicker_weekday {
    color: #333333 !important;
    font-weight: 600 !important;
}

/* Custom date picker specific styling */
.custom-date-picker .DateRangePickerInput {
    min-width: 240px !important;
    height: 38px !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    background-color: var(--background-color) !important;
}

.custom-date-picker .DateInput_input {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #333333 !important;
    padding: 8px 12px !important;
}

/* MORE AGGRESSIVE CALENDAR STYLING - Override any conflicting styles */
div.DayPicker,
.DayPicker.DayPicker,
div[class*="DayPicker"] {
    background-color: #FFFFFF !important;
    color: #333333 !important;
    font-family: 'Inter', sans-serif !important;
}

div.DayPicker_caption,
.DayPicker_caption.DayPicker_caption,
div[class*="DayPicker_caption"] {
    color: #333333 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

div.DayPicker_weekHeader,
.DayPicker_weekHeader.DayPicker_weekHeader,
div[class*="DayPicker_weekHeader"] {
    color: #333333 !important;
    font-weight: 600 !important;
}

div.DayPicker_weekHeader_ul,
.DayPicker_weekHeader_ul.DayPicker_weekHeader_ul,
ul[class*="DayPicker_weekHeader"] {
    color: #333333 !important;
}

div.DayPicker_weekHeader_li,
.DayPicker_weekHeader_li.DayPicker_weekHeader_li,
li[class*="DayPicker_weekHeader"] {
    color: #333333 !important;
    font-weight: 600 !important;
}

div.DayPicker_weekday,
.DayPicker_weekday.DayPicker_weekday,
div[class*="DayPicker_weekday"] {
    color: #333333 !important;
    font-weight: 600 !important;
}

/* Force all text elements in calendar to be dark */
.DayPicker * {
    color: #333333 !important;
}

.DayPicker span,
.DayPicker div,
.DayPicker td,
.DayPicker th,
.DayPicker li,
.DayPicker ul {
    color: #333333 !important;
}

/* Additional calendar overrides for React components */
[class*="DateRangePicker"] .DayPicker,
[class*="DateRangePicker"] .DayPicker *,
[class*="DateInput"] .DayPicker,
[class*="DateInput"] .DayPicker * {
    color: #333333 !important;
    background-color: transparent !important;
}

/* Target any remaining invisible text */
.DayPicker [style*="color: white"],
.DayPicker [style*="color: #fff"],
.DayPicker [style*="color: #ffffff"],
.DayPicker [style*="color:white"],
.DayPicker [style*="color:#fff"],
.DayPicker [style*="color:#ffffff"] {
    color: #333333 !important;
}

/* Ensure calendar popup has correct z-index and visibility */
.DateRangePicker__picker {
    background-color: #FFFFFF !important;
    border: 1px solid #E9ECEF !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    z-index: 1000 !important;
}

.DateRangePicker__picker .DayPicker {
    background-color: #FFFFFF !important;
    color: #333333 !important;
}

/* Header Responsive Design */
@media (max-width: 1366px) {
    .header-controls {
        flex-wrap: wrap !important;
        gap: 15px !important;
    }
}

@media (max-width: 768px) {
    .header-bar {
        padding: 10px 15px !important;
        flex-direction: column !important;
        height: auto !important;
    }
    
    .content-area {
        margin-top: 120px !important;
    }
}

/* Chart container styling */
.js-plotly-plot .plotly .modebar {
    background-color: transparent !important;
}

/* Enhanced Horizontal Tab Button Styling */
.horizontal-tab-button {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    position: relative !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    text-transform: none !important;
    flex-shrink: 0 !important;  /* Prevent buttons from shrinking */
    white-space: nowrap !important;  /* Keep text on one line */
}

.horizontal-tab-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15) !important;
    border-color: var(--info-color) !important;
    background-color: var(--info-color) !important;
    color: white !important;
}

.horizontal-tab-button:focus {
    outline: 2px solid var(--info-color) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(0, 123, 255, 0.1) !important;
}

.horizontal-tab-button:active {
    transform: translateY(0) !important;
}

/* Active tab button styling */
.horizontal-tab-button.active {
    background-color: var(--info-color) !important;
    color: white !important;
    border-color: var(--info-color) !important;
    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.25) !important;
}

.horizontal-tab-button.active:hover {
    background-color: var(--info-color) !important;
    opacity: 0.9 !important;
}

/* Tab navigation container styling */
.horizontal-tab-navigation {
    background: linear-gradient(135deg, var(--page-bg-color) 0%, #f0f2f5 100%) !important;
    border-bottom: 1px solid var(--border-color) !important;
    position: sticky !important;
    top: 90px !important;
    z-index: 999 !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}

/* Tab content area styling */
.tab-content {
    background-color: var(--page-bg-color) !important;
    min-height: calc(100vh - 180px) !important;
    padding: var(--spacing-lg) !important;
    border-radius: 0 0 var(--border-radius) var(--border-radius) !important;
}

/* Responsive tab button styling */
@media (max-width: 1366px) {
    .horizontal-tab-button {
        min-width: 160px !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        height: 45px !important;
    }
}

@media (max-width: 768px) {
    .horizontal-tab-button {
        min-width: 140px !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        height: 40px !important;
        margin: 2px !important;
    }
    
    .horizontal-tab-navigation .tab-buttons-container {
        padding: 15px 20px !important;
        gap: 5px !important;
    }
    
    .tab-content {
        min-height: calc(100vh - 160px) !important;
        padding: var(--spacing-md) !important;
    }
}

@media (max-width: 480px) {
    .horizontal-tab-button {
        min-width: 120px !important;
        font-size: 12px !important;
        padding: 6px 12px !important;
        height: 36px !important;
        flex-direction: column !important;
        gap: 2px !important;
    }
    
    .horizontal-tab-button i {
        margin-right: 0 !important;
        font-size: 14px !important;
    }
    
    .horizontal-tab-button span {
        font-size: 11px !important;
        font-weight: 500 !important;
    }
}

/* Accessibility Enhancements */
.dashboard-card:focus-within {
    outline: 2px solid var(--info-color);
    outline-offset: 2px;
}

/* Print Styles */
@media print {
    .dashboard-card {
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #000;
    }
    
    .card-grid-container {
        display: block;
    }
    
    .dashboard-card {
        margin-bottom: var(--spacing-md);
        page-break-inside: avoid;
    }
}

/* High Contrast Mode Support */
@media (prefers-contrast: high) {
    .dashboard-card {
        border-width: 2px;
    }
    
    .dashboard-card:hover {
        border-width: 3px;
    }
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
    .dashboard-card,
    .loading-placeholder {
        transition: none;
        animation: none;
    }
}

/* Professional Duration Analysis Layout Styles */
.duration-analysis-container {
    background-color: #FFFFFF !important;
    border-radius: var(--border-radius);
    padding: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid var(--border-color);
}

.duration-analysis-header {
    text-align: center;
    margin-bottom: var(--spacing-xl);
    padding-bottom: var(--spacing-lg);
    border-bottom: 2px solid var(--border-color);
}

.duration-analysis-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 25px;
    padding: 0 10px;
    align-items: start;
}

/* Responsive Design for Duration Analysis */
@media (max-width: 1366px) {
    .duration-analysis-grid {
        gap: 20px;
        padding: 0 5px;
    }
    
    .duration-analysis-container {
        padding: var(--spacing-lg);
    }
}

@media (max-width: 1024px) {
    .duration-analysis-grid {
        gap: 15px;
        padding: 0;
    }
}

@media (max-width: 768px) {
    .duration-analysis-grid {
        grid-template-columns: 1fr;
        gap: var(--spacing-lg);
    }
    
    .duration-analysis-container {
        padding: var(--spacing-md);
    }
    
    .duration-analysis-header h3 {
        font-size: 18px !important;
    }
    
    .duration-analysis-header p {
        font-size: 13px !important;
    }
}

@media (max-width: 480px) {
    .duration-analysis-container {
        padding: var(--spacing-sm);
        margin-bottom: var(--spacing-lg);
    }
    
    .duration-analysis-grid {
        gap: var(--spacing-md);
    }
}

/* Duration Chart Card Styling */
.duration-analysis-grid .dashboard-card {
    background-color: #FFFFFF !important;
    border: 1px solid var(--border-color);
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease;
}

.duration-analysis-grid .dashboard-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Ensure chart backgrounds are transparent */
.duration-analysis-grid .js-plotly-plot,
.duration-analysis-grid .plotly {
    background-color: transparent !important;
}

.duration-analysis-grid .js-plotly-plot .plot-container,
.duration-analysis-grid .plotly .plot-container {
    background-color: transparent !important;
}

/* Dark Mode Support (Future Enhancement) */
@media (prefers-color-scheme: dark) {
    :root {
        --background-color: #1a1a1a;
        --page-bg-color: #121212;
        --border-color: #333333;
        --primary-color: #ffffff;
        --tertiary-color: #b0b0b0;
    }
}
"""