# meluna/analysis/trade_visualization/visualizations/TradeChart.py

"""
Core Candlestick Chart Rendering with Trade Markers for Individual Trade Visualization.

This module implements the primary visualization component of the Individual Trade 
Visualization system. The main chart (80% height) displays OHLCV candlesticks with 
trade-specific annotations including entry/exit points, stop loss, target levels, 
and trade duration highlighting.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from ..base_interfaces import (
    BaseChartComponent, 
    TradeVisualizationConfig, 
    ChartAnnotation,
    TradeDataProcessor
)
from ..calculations.TradeMetrics import TradeVisualizationMetrics
from ..calculations.PriceActionMetrics import PriceActionAnalysis


class TradeChart(BaseChartComponent):
    """
    Core candlestick chart with trade annotations and risk management levels.
    
    Features:
    - OHLCV candlestick rendering with 80% height allocation
    - Trade entry/exit markers with price labels
    - Stop loss and target lines
    - Trade duration highlighting
    - Zoom and pan functionality
    - Context window integration
    - Professional styling following dashboard theme
    """
    
    # Professional color scheme following dashboard standards
    COLORS = {
        'primary': '#0D1B2A',      # Deep Navy Blue
        'secondary': '#1B2951',    # Slate Blue
        'tertiary': '#415A77',     # Steel Gray
        'profit': '#00C851',       # Profit Green
        'loss': '#FF4444',         # Loss Red
        'caution': '#FF8800',      # Caution Orange
        'info': '#007BFF',         # Insight Blue
        'background': '#FFFFFF',   # Pure White
        'page_bg': '#F8F9FA',      # Light Gray
        'border': '#E9ECEF',       # Warm Gray
        'grid': '#F1F3F4',         # Grid lines
        'text': '#2C3E50',         # Text color
    }
    
    def __init__(self, config: TradeVisualizationConfig):
        """
        Initialize the TradeChart component.
        
        Args:
            config: Visualization configuration settings
        """
        super().__init__(config)
        self.chart = None
        self.price_data = None
        self.trade_data = None
        self.context_window_data = None
        
    def prepare_data(self, trade_metrics: TradeVisualizationMetrics, 
                     price_action: PriceActionAnalysis) -> None:
        """
        Prepare OHLCV data and trade information for visualization.
        
        Args:
            trade_metrics: Calculated trade metrics including entry/exit data
            price_action: Price action analysis with OHLCV data
        """
        # Store the complete trade metrics for P&L evolution
        self.trade_metrics = trade_metrics
        
        # Extract trade information
        self.trade_data = {
            'entry_time': trade_metrics.entry_timestamp,
            'exit_time': trade_metrics.exit_timestamp,
            'entry_price': trade_metrics.entry_price,
            'exit_price': trade_metrics.exit_price,
            'stop_loss': getattr(trade_metrics, 'planned_risk', None),
            'target': getattr(trade_metrics, 'planned_reward', None),
            'pnl': trade_metrics.total_pnl,
            'quantity': trade_metrics.quantity,
            'trade_id': trade_metrics.trade_id
        }
        
        # Extract context window data
        self.context_window_data = TradeDataProcessor.extract_context_window(
            price_data=price_action.price_data,
            entry_time=self.trade_data['entry_time'],
            exit_time=self.trade_data['exit_time'],
            bars_before=self.config.bars_before_entry,
            bars_after=self.config.bars_after_exit
        )
        
        # Validate required OHLCV columns
        required_columns = ['open', 'high', 'low', 'close']
        if 'volume' in self.context_window_data.columns:
            required_columns.append('volume')
            
        for col in required_columns:
            if col not in self.context_window_data.columns:
                raise ValueError(f"Missing required column: {col}")
                
        self.price_data = self.context_window_data.copy()
        
        # Create trade markers
        self.annotations = TradeDataProcessor.create_trade_markers(
            entry_time=self.trade_data['entry_time'],
            exit_time=self.trade_data['exit_time'],
            entry_price=self.trade_data['entry_price'],
            exit_price=self.trade_data['exit_price'],
            trade_pnl=self.trade_data['pnl']
        )
        
    def create_plot(self) -> go.Figure:
        """
        Create the main candlestick chart with trade annotations.
        
        Returns:
            Plotly Figure object with candlestick chart and trade markers
        """
        if self.price_data is None:
            raise ValueError("Price data not prepared. Call prepare_data() first.")
            
        # Create subplot structure: 60% main chart, 20% P&L evolution, 20% volume
        height_ratios = [0.6, 0.2, 0.2]
        
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=height_ratios,
            subplot_titles=('Price Chart', 'P&L Evolution', 'Volume'),
            vertical_spacing=0.03,
            shared_xaxes=True
        )
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=self.price_data.index,
            open=self.price_data['open'],
            high=self.price_data['high'],
            low=self.price_data['low'],
            close=self.price_data['close'],
            name="OHLC",
            increasing=dict(line=dict(color=self.COLORS['profit'], width=1)),
            decreasing=dict(line=dict(color=self.COLORS['loss'], width=1)),
            line=dict(width=1),
            whiskerwidth=0.8
        )
        
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add P&L evolution subplot
        self._add_pnl_evolution_subplot(fig)
        
        # Add volume bar chart if available
        if 'volume' in self.price_data.columns and self.config.show_volume:
            # Color volume bars based on price movement
            volume_colors = []
            for i in range(len(self.price_data)):
                if self.price_data['close'].iloc[i] >= self.price_data['open'].iloc[i]:
                    volume_colors.append(self.COLORS['profit'])
                else:
                    volume_colors.append(self.COLORS['loss'])
            
            volume_trace = go.Bar(
                x=self.price_data.index,
                y=self.price_data['volume'],
                name="Volume",
                marker_color=volume_colors,
                opacity=0.7,
                showlegend=False
            )
            
            fig.add_trace(volume_trace, row=3, col=1)
        
        # Add trade annotations and risk levels
        self._add_trade_annotations(fig)
        self._add_risk_levels(fig)
        self._add_trade_duration_highlighting(fig)
        
        # Apply professional styling
        self._apply_chart_styling(fig)
        
        self.chart = fig
        return fig
        
    def _add_trade_annotations(self, fig: go.Figure) -> None:
        """
        Add trade entry and exit markers with price labels.
        
        Args:
            fig: Plotly figure to add annotations to
        """
        for annotation in self.annotations:
            if annotation.annotation_type == "entry":
                # Entry marker - Green triangle up
                fig.add_trace(go.Scatter(
                    x=[annotation.timestamp],
                    y=[annotation.value],
                    mode='markers+text',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color=self.COLORS['profit'],
                        line=dict(width=2, color='white')
                    ),
                    text=[annotation.text],
                    textposition="top center",
                    textfont=dict(size=10, color=self.COLORS['text']),
                    name="Entry",
                    showlegend=True,
                    hovertemplate=f"<b>Trade Entry</b><br>" +
                                f"Time: {annotation.timestamp}<br>" +
                                f"Price: ₹{annotation.value:.2f}<br>" +
                                f"Quantity: {self.trade_data['quantity']}<extra></extra>"
                ), row=1, col=1)
                
            elif annotation.annotation_type == "exit":
                # Exit marker - Triangle down (color based on P&L)
                marker_color = self.COLORS['profit'] if self.trade_data['pnl'] > 0 else self.COLORS['loss']
                
                fig.add_trace(go.Scatter(
                    x=[annotation.timestamp],
                    y=[annotation.value],
                    mode='markers+text',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color=marker_color,
                        line=dict(width=2, color='white')
                    ),
                    text=[annotation.text],
                    textposition="bottom center",
                    textfont=dict(size=10, color=self.COLORS['text']),
                    name="Exit",
                    showlegend=True,
                    hovertemplate=f"<b>Trade Exit</b><br>" +
                                f"Time: {annotation.timestamp}<br>" +
                                f"Price: ₹{annotation.value:.2f}<br>" +
                                f"P&L: ₹{self.trade_data['pnl']:.2f}<extra></extra>"
                ), row=1, col=1)
                
    def _add_risk_levels(self, fig: go.Figure) -> None:
        """
        Add horizontal lines for stop loss and target levels.
        
        Args:
            fig: Plotly figure to add risk level lines to
        """
        x_range = [self.price_data.index[0], self.price_data.index[-1]]
        
        # Stop loss line
        if self.trade_data['stop_loss'] is not None:
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[self.trade_data['stop_loss'], self.trade_data['stop_loss']],
                mode='lines',
                line=dict(
                    color=self.COLORS['loss'],
                    width=2,
                    dash='dash'
                ),
                name="Stop Loss",
                showlegend=True,
                hovertemplate=f"<b>Stop Loss</b><br>Price: ₹{self.trade_data['stop_loss']:.2f}<extra></extra>"
            ), row=1, col=1)
            
        # Target line
        if self.trade_data['target'] is not None:
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[self.trade_data['target'], self.trade_data['target']],
                mode='lines',
                line=dict(
                    color=self.COLORS['profit'],
                    width=2,
                    dash='dash'
                ),
                name="Target",
                showlegend=True,
                hovertemplate=f"<b>Target</b><br>Price: ₹{self.trade_data['target']:.2f}<extra></extra>"
            ), row=1, col=1)
            
    def _add_trade_duration_highlighting(self, fig: go.Figure) -> None:
        """
        Add shaded background area to highlight trade duration.
        
        Args:
            fig: Plotly figure to add trade duration highlighting to
        """
        # Calculate y-axis range for the rectangle
        y_min = self.price_data['low'].min() * 0.99
        y_max = self.price_data['high'].max() * 1.01
        
        # Add semi-transparent rectangle for trade duration
        fig.add_shape(
            type="rect",
            x0=self.trade_data['entry_time'],
            x1=self.trade_data['exit_time'],
            y0=y_min,
            y1=y_max,
            fillcolor=self.COLORS['info'],
            opacity=0.1,
            line=dict(width=0),
            layer="below",
            row=1, col=1
        )
        
        # Add annotation for trade duration
        trade_duration = self.trade_data['exit_time'] - self.trade_data['entry_time']
        duration_text = f"Trade Duration: {trade_duration.days}d {trade_duration.seconds//3600}h"
        
        # Calculate midpoint timestamp properly
        mid_timestamp = self.trade_data['entry_time'] + (trade_duration / 2)
        
        fig.add_annotation(
            x=mid_timestamp,
            y=y_max * 0.98,
            text=duration_text,
            showarrow=False,
            font=dict(size=10, color=self.COLORS['text']),
            bgcolor=self.COLORS['background'],
            bordercolor=self.COLORS['border'],
            borderwidth=1,
            row=1, col=1
        )
        
    def _add_pnl_evolution_subplot(self, fig: go.Figure) -> None:
        """
        Add P&L evolution subplot with MFE/MAE markers.
        
        Args:
            fig: Plotly figure to add P&L evolution subplot to
        """
        if not hasattr(self, 'trade_metrics') or self.trade_metrics is None:
            return
            
        # Get P&L curve data from trade metrics
        pnl_curve = getattr(self.trade_metrics, 'pnl_curve', pd.Series())
        
        if pnl_curve.empty:
            return
            
        # Create P&L evolution line chart
        pnl_line = go.Scatter(
            x=pnl_curve.index,
            y=pnl_curve.values,
            mode='lines',
            line=dict(
                color=self.COLORS['info'],
                width=2
            ),
            name="P&L Evolution",
            showlegend=True,
            hovertemplate="<b>P&L Evolution</b><br>" +
                         "Time: %{x}<br>" +
                         "P&L: ₹%{y:.2f}<br>" +
                         "Percentage: %{customdata:.2f}%<extra></extra>",
            customdata=[((val / (self.trade_data['entry_price'] * abs(self.trade_data['quantity']))) * 100) 
                       for val in pnl_curve.values]
        )
        
        fig.add_trace(pnl_line, row=2, col=1)
        
        # Add zero line (break-even reference)
        x_range = [pnl_curve.index[0], pnl_curve.index[-1]]
        zero_line = go.Scatter(
            x=x_range,
            y=[0, 0],
            mode='lines',
            line=dict(
                color=self.COLORS['text'],
                width=1,
                dash='dash'
            ),
            name="Break-even",
            showlegend=False,
            hoverinfo='skip'
        )
        
        fig.add_trace(zero_line, row=2, col=1)
        
        # Add MFE marker (Maximum Favorable Excursion)
        if hasattr(self.trade_metrics, 'mfe_timestamp') and self.trade_metrics.mfe_timestamp:
            mfe_marker = go.Scatter(
                x=[self.trade_metrics.mfe_timestamp],
                y=[self.trade_metrics.mfe_value],
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color=self.COLORS['profit'],
                    line=dict(width=2, color='white')
                ),
                text=[f"MFE: ₹{self.trade_metrics.mfe_value:.2f}"],
                textposition="top center",
                textfont=dict(size=9, color=self.COLORS['text']),
                name="MFE",
                showlegend=True,
                hovertemplate="<b>Maximum Favorable Excursion</b><br>" +
                             f"Time: {self.trade_metrics.mfe_timestamp}<br>" +
                             f"P&L: ₹{self.trade_metrics.mfe_value:.2f}<br>" +
                             f"Percentage: {self.trade_metrics.mfe_percentage:.2f}%<extra></extra>"
            )
            
            fig.add_trace(mfe_marker, row=2, col=1)
        
        # Add MAE marker (Maximum Adverse Excursion)
        if hasattr(self.trade_metrics, 'mae_timestamp') and self.trade_metrics.mae_timestamp:
            mae_marker = go.Scatter(
                x=[self.trade_metrics.mae_timestamp],
                y=[self.trade_metrics.mae_value],
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color=self.COLORS['loss'],
                    line=dict(width=2, color='white')
                ),
                text=[f"MAE: ₹{self.trade_metrics.mae_value:.2f}"],
                textposition="bottom center",
                textfont=dict(size=9, color=self.COLORS['text']),
                name="MAE",
                showlegend=True,
                hovertemplate="<b>Maximum Adverse Excursion</b><br>" +
                             f"Time: {self.trade_metrics.mae_timestamp}<br>" +
                             f"P&L: ₹{self.trade_metrics.mae_value:.2f}<br>" +
                             f"Percentage: {self.trade_metrics.mae_percentage:.2f}%<extra></extra>"
            )
            
            fig.add_trace(mae_marker, row=2, col=1)
        
        # Add final P&L endpoint marker
        final_pnl = pnl_curve.iloc[-1] if not pnl_curve.empty else 0
        final_time = pnl_curve.index[-1] if not pnl_curve.empty else self.trade_data['exit_time']
        final_color = self.COLORS['profit'] if final_pnl > 0 else self.COLORS['loss']
        
        final_marker = go.Scatter(
            x=[final_time],
            y=[final_pnl],
            mode='markers+text',
            marker=dict(
                symbol='diamond',
                size=12,
                color=final_color,
                line=dict(width=2, color='white')
            ),
            text=[f"Final: ₹{final_pnl:.2f}"],
            textposition="top center",
            textfont=dict(size=9, color=self.COLORS['text']),
            name="Final P&L",
            showlegend=True,
            hovertemplate="<b>Final Trade Result</b><br>" +
                         f"Time: {final_time}<br>" +
                         f"P&L: ₹{final_pnl:.2f}<br>" +
                         f"Percentage: {self.trade_metrics.pnl_percentage:.2f}%<extra></extra>"
        )
        
        fig.add_trace(final_marker, row=2, col=1)
        
    def _apply_chart_styling(self, fig: go.Figure) -> None:
        """
        Apply professional styling consistent with dashboard theme.
        
        Args:
            fig: Plotly figure to style
        """
        fig.update_layout(
            # Overall layout
            height=self.config.chart_height,
            width=self.config.chart_width,
            title=dict(
                text=f"Trade Analysis - {self.trade_data['trade_id']}",
                font=dict(size=16, color=self.COLORS['primary']),
                x=0.5,
                xanchor='center'
            ),
            
            # Background colors
            plot_bgcolor=self.COLORS['background'],
            paper_bgcolor=self.COLORS['page_bg'],
            
            # Font styling
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color=self.COLORS['text']
            ),
            
            # Legend styling
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor=self.COLORS['background'],
                bordercolor=self.COLORS['border'],
                borderwidth=1
            ),
            
            # Remove range selector
            xaxis=dict(rangeslider=dict(visible=False)),
            
            # Margin optimization
            margin=dict(l=60, r=60, t=80, b=60),
            
            # Hover mode
            hovermode='x unified'
        )
        
        # Style x-axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.COLORS['grid'],
            showline=True,
            linewidth=1,
            linecolor=self.COLORS['border'],
            mirror=True
        )
        
        # Style y-axes
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.COLORS['grid'],
            showline=True,
            linewidth=1,
            linecolor=self.COLORS['border'],
            mirror=True,
            fixedrange=False  # Allow zoom/pan
        )
        
        # Specific styling for main price chart
        fig.update_yaxes(
            title_text="Price (₹)",
            title_font=dict(size=12, color=self.COLORS['text']),
            tickformat='.2f',
            row=1, col=1
        )
        
        # Specific styling for P&L evolution chart
        fig.update_yaxes(
            title_text="P&L (₹)",
            title_font=dict(size=12, color=self.COLORS['text']),
            tickformat='.2f',
            row=2, col=1
        )
        
        # Specific styling for volume chart
        if self.config.show_volume:
            fig.update_yaxes(
                title_text="Volume",
                title_font=dict(size=12, color=self.COLORS['text']),
                tickformat='.0f',
                row=3, col=1
            )
            
    def update_context_window(self, bars_before: int, bars_after: int) -> None:
        """
        Update the context window around the trade.
        
        Args:
            bars_before: Number of bars to show before trade entry
            bars_after: Number of bars to show after trade exit
        """
        if not (self.config.min_bars_before <= bars_before <= self.config.max_bars_before):
            raise ValueError(f"bars_before must be between {self.config.min_bars_before} and {self.config.max_bars_before}")
            
        self.config.bars_before_entry = bars_before
        self.config.bars_after_exit = bars_after
        
        # Re-extract context window data
        if hasattr(self, 'trade_data') and self.trade_data is not None:
            # Re-prepare data with new context window
            # Note: This requires the original price_action data to be stored
            pass
            
    def add_technical_indicator(self, indicator_data: pd.Series, 
                              indicator_name: str, 
                              color: str = None) -> None:
        """
        Add a technical indicator overlay to the main chart.
        
        Args:
            indicator_data: Time series data for the indicator
            indicator_name: Name of the indicator for legend
            color: Color for the indicator line
        """
        if self.chart is None:
            raise ValueError("Chart not created. Call create_plot() first.")
            
        color = color or self.COLORS['info']
        
        indicator_trace = go.Scatter(
            x=indicator_data.index,
            y=indicator_data.values,
            mode='lines',
            line=dict(color=color, width=2),
            name=indicator_name,
            showlegend=True,
            hovertemplate=f"<b>{indicator_name}</b><br>Value: %{{y:.2f}}<extra></extra>"
        )
        
        self.chart.add_trace(indicator_trace, row=1, col=1)
        
    def export_chart(self, filename: str, format: str = "html") -> bool:
        """
        Export the chart to file.
        
        Args:
            filename: Output filename
            format: Export format ('html', 'png', 'svg', 'pdf')
            
        Returns:
            True if export successful, False otherwise
        """
        if self.chart is None:
            return False
            
        try:
            if format.lower() == "html":
                self.chart.write_html(filename)
            elif format.lower() == "png":
                self.chart.write_image(filename, format="png")
            elif format.lower() == "svg":
                self.chart.write_image(filename, format="svg")
            elif format.lower() == "pdf":
                self.chart.write_image(filename, format="pdf")
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            return True
            
        except Exception as e:
            print(f"Error exporting chart: {e}")
            return False
            
    def get_chart_data(self) -> Dict[str, Any]:
        """
        Get the underlying chart data for external analysis.
        
        Returns:
            Dictionary containing price data, trade data, and annotations
        """
        return {
            'price_data': self.price_data,
            'trade_data': self.trade_data,
            'annotations': self.annotations,
            'config': self.config
        }
        
    def update_data(self, new_data: pd.DataFrame) -> None:
        """
        Update the chart with new price data.
        
        Args:
            new_data: New OHLCV data to display
        """
        self.price_data = new_data
        # Re-create plot with new data
        if self.chart is not None:
            self.create_plot()