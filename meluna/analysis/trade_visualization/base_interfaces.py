# meluna/analysis/trade_visualization/base_interfaces.py

"""
Base interfaces and data structures for trade visualization components.

This module defines abstract base classes and interfaces that will be used
by visualization components in future phases of development.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Protocol
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

from .calculations.TradeMetrics import TradeVisualizationMetrics
from .calculations.PriceActionMetrics import PriceActionAnalysis


@dataclass
class TradeVisualizationConfig:
    """Configuration for trade visualization components."""
    # Chart dimensions and layout
    chart_height: int = 800
    chart_width: int = 1200
    subplot_height_ratios: List[float] = None  # [0.7, 0.3] for main chart and P&L subplot
    
    # Context window settings
    bars_before_entry: int = 50
    bars_after_exit: int = 20
    min_bars_before: int = 20
    max_bars_before: int = 200
    
    # Display preferences
    show_volume: bool = True
    show_indicators: bool = True
    show_swing_points: bool = True
    show_mfe_mae: bool = True
    
    # Color scheme
    color_scheme: str = "default"  # "default", "dark", "light"
    
    def __post_init__(self):
        if self.subplot_height_ratios is None:
            self.subplot_height_ratios = [0.7, 0.3]


@dataclass
class ChartAnnotation:
    """Represents an annotation on the chart."""
    timestamp: datetime
    value: float
    annotation_type: str  # "entry", "exit", "mfe", "mae", "custom"
    text: str
    color: str = "blue"
    marker_style: str = "triangle"


@dataclass
class PlotData:
    """Container for data to be plotted."""
    x_data: List[datetime]
    y_data: List[float]
    plot_type: str  # "line", "bar", "candlestick", "scatter"
    name: str
    color: str = "blue"
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class TradeVisualizationInterface(Protocol):
    """Protocol defining the interface for trade visualization components."""
    
    def render(self, trade_metrics: TradeVisualizationMetrics, 
               price_action: PriceActionAnalysis,
               config: TradeVisualizationConfig) -> Any:
        """
        Render the visualization component.
        
        Args:
            trade_metrics: Calculated trade metrics
            price_action: Price action analysis
            config: Visualization configuration
            
        Returns:
            Rendered visualization object
        """
        ...
    
    def update_data(self, new_data: pd.DataFrame) -> None:
        """Update the visualization with new data."""
        ...
    
    def add_annotation(self, annotation: ChartAnnotation) -> None:
        """Add an annotation to the visualization."""
        ...
    
    def export(self, filename: str, format: str = "html") -> bool:
        """Export the visualization to file."""
        ...


class BaseChartComponent(ABC):
    """Abstract base class for chart components."""
    
    def __init__(self, config: TradeVisualizationConfig):
        self.config = config
        self.annotations: List[ChartAnnotation] = []
        self.plot_data: List[PlotData] = []
    
    @abstractmethod
    def prepare_data(self, trade_metrics: TradeVisualizationMetrics, 
                     price_action: PriceActionAnalysis) -> None:
        """Prepare data for visualization."""
        pass
    
    @abstractmethod
    def create_plot(self) -> Any:
        """Create the plot/chart."""
        pass
    
    def add_annotation(self, annotation: ChartAnnotation) -> None:
        """Add an annotation to the chart."""
        self.annotations.append(annotation)
    
    def clear_annotations(self) -> None:
        """Clear all annotations."""
        self.annotations.clear()
    
    def update_config(self, config: TradeVisualizationConfig) -> None:
        """Update the configuration."""
        self.config = config


class BaseMetricsPanel(ABC):
    """Abstract base class for metrics display panels."""
    
    def __init__(self, panel_type: str):
        self.panel_type = panel_type
        self.metrics_data: Dict[str, Any] = {}
    
    @abstractmethod
    def format_metrics(self, trade_metrics: TradeVisualizationMetrics, 
                      price_action: PriceActionAnalysis) -> Dict[str, str]:
        """Format metrics for display."""
        pass
    
    @abstractmethod
    def render_panel(self) -> Any:
        """Render the metrics panel."""
        pass
    
    def update_metrics(self, trade_metrics: TradeVisualizationMetrics, 
                      price_action: PriceActionAnalysis) -> None:
        """Update the metrics data."""
        self.metrics_data = self.format_metrics(trade_metrics, price_action)


class TradeDataProcessor:
    """Utility class for processing trade data for visualization."""
    
    @staticmethod
    def extract_context_window(price_data: pd.DataFrame, 
                             entry_time: datetime, 
                             exit_time: datetime,
                             bars_before: int = 50, 
                             bars_after: int = 20) -> pd.DataFrame:
        """
        Extract price data with context window around the trade.
        
        Args:
            price_data: Full OHLCV data
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            bars_before: Number of bars to include before entry
            bars_after: Number of bars to include after exit
            
        Returns:
            Filtered DataFrame with context window
        """
        # Ensure datetime index
        if not isinstance(price_data.index, pd.DatetimeIndex):
            if 'timestamp' in price_data.columns:
                price_data = price_data.set_index('timestamp')
            else:
                raise ValueError("Price data must have datetime index or timestamp column")
        
        # Find entry and exit positions
        entry_idx = price_data.index.get_indexer([entry_time], method='nearest')[0]
        exit_idx = price_data.index.get_indexer([exit_time], method='nearest')[0]
        
        # Calculate window bounds
        start_idx = max(0, entry_idx - bars_before)
        end_idx = min(len(price_data) - 1, exit_idx + bars_after)
        
        # Extract window
        context_data = price_data.iloc[start_idx:end_idx + 1].copy()
        
        return context_data
    
    @staticmethod
    def validate_trade_data(trade_data: Dict[str, Any]) -> bool:
        """
        Validate trade data structure.
        
        Args:
            trade_data: Trade information dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'trade_id', 'symbol', 'entry_timestamp', 'exit_timestamp',
            'entry_price', 'exit_price', 'quantity'
        ]
        
        for field in required_fields:
            if field not in trade_data:
                return False
            if trade_data[field] is None:
                return False
        
        # Validate timestamp order
        entry_time = pd.to_datetime(trade_data['entry_timestamp'])
        exit_time = pd.to_datetime(trade_data['exit_timestamp'])
        
        if entry_time >= exit_time:
            return False
        
        return True
    
    @staticmethod
    def create_trade_markers(entry_time: datetime, 
                           exit_time: datetime,
                           entry_price: float, 
                           exit_price: float,
                           trade_pnl: float) -> List[ChartAnnotation]:
        """
        Create standard trade entry/exit markers.
        
        Args:
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            trade_pnl: Trade P&L
            
        Returns:
            List of chart annotations for entry/exit points
        """
        markers = []
        
        # Entry marker
        entry_marker = ChartAnnotation(
            timestamp=entry_time,
            value=entry_price,
            annotation_type="entry",
            text=f"Entry: ₹{entry_price:.2f}",
            color="green",
            marker_style="triangle-up"
        )
        markers.append(entry_marker)
        
        # Exit marker
        exit_color = "green" if trade_pnl > 0 else "red"
        exit_marker = ChartAnnotation(
            timestamp=exit_time,
            value=exit_price,
            annotation_type="exit",
            text=f"Exit: ₹{exit_price:.2f}",
            color=exit_color,
            marker_style="triangle-down"
        )
        markers.append(exit_marker)
        
        return markers


# Type aliases for commonly used types
TradeData = Dict[str, Any]
PriceData = pd.DataFrame
VisualizationComponent = Union[BaseChartComponent, BaseMetricsPanel]