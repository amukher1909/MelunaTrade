# meluna/analysis/trade_visualization/calculations/TradeMetrics.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class TradeVisualizationMetrics:
    """Container for single trade visualization metrics."""
    # Core trade information
    trade_id: str
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    quantity: int
    duration_hours: float
    
    # P&L metrics
    total_pnl: float
    pnl_percentage: float
    pnl_curve: pd.Series = field(default_factory=pd.Series)
    
    # MFE/MAE metrics
    mfe_value: float = 0.0
    mae_value: float = 0.0
    mfe_timestamp: Optional[datetime] = None
    mae_timestamp: Optional[datetime] = None
    mfe_percentage: float = 0.0
    mae_percentage: float = 0.0
    
    # Time analysis
    time_in_profit_hours: float = 0.0
    time_in_loss_hours: float = 0.0
    time_in_profit_percentage: float = 0.0
    time_in_loss_percentage: float = 0.0
    
    # Risk/Reward metrics
    planned_risk: Optional[float] = None
    planned_reward: Optional[float] = None
    achieved_risk_reward_ratio: Optional[float] = None
    planned_risk_reward_ratio: Optional[float] = None
    
    # Additional metrics
    trade_efficiency: float = 0.0  # How much of potential profit was captured
    risk_efficiency: float = 0.0   # How well risk was managed


class TradeMetricsCalculator:
    """
    Calculates comprehensive metrics for individual trade visualization.
    
    This class provides detailed analysis of a single trade including:
    - Trade entry/exit points with exact timestamps
    - P&L curve calculation (point by point)
    - Maximum Favorable/Adverse Excursion (MFE/MAE)
    - Time in profit vs time in loss analysis
    - Risk/Reward achieved vs planned calculations
    """
    
    def __init__(self, price_data: pd.DataFrame, trade_data: Dict[str, Any]):
        """
        Initialize the TradeMetricsCalculator.
        
        Args:
            price_data (pd.DataFrame): OHLCV data with columns:
                - timestamp: datetime index or column
                - open, high, low, close: price data
                - volume: trading volume
            trade_data (Dict): Trade information containing:
                - trade_id, symbol, entry_timestamp, exit_timestamp
                - entry_price, exit_price, quantity
                - stop_loss (optional), target (optional)
        """
        self.price_data = price_data.copy()
        self.trade_data = trade_data.copy()
        
        # Validate inputs
        self._validate_inputs()
        
        # Ensure timestamp is datetime and set as index if needed
        self._process_timestamps()
        
        # Filter price data to trade period
        self.trade_period_data = self._extract_trade_period_data()
        
        logger.info(f"TradeMetricsCalculator initialized for trade {trade_data.get('trade_id', 'unknown')}")
    
    def _validate_inputs(self) -> None:
        """Validate input data structure and required fields."""
        # Validate price data
        required_price_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_price_cols if col not in self.price_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required price data columns: {missing_cols}")
        
        # Validate trade data
        required_trade_fields = [
            'trade_id', 'symbol', 'entry_timestamp', 'exit_timestamp',
            'entry_price', 'exit_price', 'quantity'
        ]
        missing_fields = [field for field in required_trade_fields if field not in self.trade_data]
        if missing_fields:
            raise ValueError(f"Missing required trade data fields: {missing_fields}")
        
        # Validate price data is not empty
        if self.price_data.empty:
            raise ValueError("Price data cannot be empty")
    
    def _process_timestamps(self) -> None:
        """Process timestamps and set appropriate index."""
        # Handle timestamp column vs index
        if 'timestamp' in self.price_data.columns:
            self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
            self.price_data.set_index('timestamp', inplace=True)
        elif not isinstance(self.price_data.index, pd.DatetimeIndex):
            raise ValueError("Price data must have datetime index or timestamp column")
        
        # Convert trade timestamps
        self.trade_data['entry_timestamp'] = pd.to_datetime(self.trade_data['entry_timestamp'])
        self.trade_data['exit_timestamp'] = pd.to_datetime(self.trade_data['exit_timestamp'])
        
        # Validate timestamp order
        if self.trade_data['entry_timestamp'] >= self.trade_data['exit_timestamp']:
            raise ValueError("Entry timestamp must be before exit timestamp")
    
    def _extract_trade_period_data(self) -> pd.DataFrame:
        """Extract price data for the trade period."""
        entry_time = self.trade_data['entry_timestamp']
        exit_time = self.trade_data['exit_timestamp']
        
        # Filter data to trade period
        trade_data = self.price_data[
            (self.price_data.index >= entry_time) & 
            (self.price_data.index <= exit_time)
        ].copy()
        
        if trade_data.empty:
            raise ValueError(f"No price data available for trade period {entry_time} to {exit_time}")
        
        return trade_data
    
    def calculate_pnl_curve(self) -> pd.Series:
        """
        Calculate point-by-point P&L curve during the trade.
        
        Returns:
            pd.Series: P&L values indexed by timestamp
        """
        entry_price = self.trade_data['entry_price']
        quantity = self.trade_data['quantity']
        
        # Calculate P&L using close prices
        pnl_curve = (self.trade_period_data['close'] - entry_price) * quantity
        
        logger.debug(f"P&L curve calculated with {len(pnl_curve)} data points")
        return pnl_curve
    
    def calculate_mfe_mae(self, pnl_curve: pd.Series) -> Tuple[float, float, datetime, datetime]:
        """
        Calculate Maximum Favorable and Adverse Excursion with timestamps.
        
        Args:
            pnl_curve (pd.Series): P&L curve values
            
        Returns:
            Tuple: (mfe_value, mae_value, mfe_timestamp, mae_timestamp)
        """
        if pnl_curve.empty:
            return 0.0, 0.0, None, None
        
        # Find MFE (maximum profit point)
        mfe_idx = pnl_curve.idxmax()
        mfe_value = pnl_curve.loc[mfe_idx]
        mfe_timestamp = mfe_idx
        
        # Find MAE (maximum loss point)
        mae_idx = pnl_curve.idxmin()
        mae_value = pnl_curve.loc[mae_idx]
        mae_timestamp = mae_idx
        
        logger.debug(f"MFE: {mfe_value:.2f} at {mfe_timestamp}, MAE: {mae_value:.2f} at {mae_timestamp}")
        return mfe_value, mae_value, mfe_timestamp, mae_timestamp
    
    def calculate_time_analysis(self, pnl_curve: pd.Series) -> Tuple[float, float]:
        """
        Calculate time in profit vs time in loss.
        
        Args:
            pnl_curve (pd.Series): P&L curve values
            
        Returns:
            Tuple: (time_in_profit_hours, time_in_loss_hours)
        """
        if pnl_curve.empty or len(pnl_curve) < 2:
            return 0.0, 0.0
        
        # Determine profit/loss periods
        profit_mask = pnl_curve > 0
        loss_mask = pnl_curve < 0
        
        # Calculate time differences between consecutive timestamps
        time_diffs = pd.Series(pnl_curve.index).diff().dt.total_seconds() / 3600  # Convert to hours
        time_diffs.index = pnl_curve.index
        time_diffs = time_diffs.fillna(0)  # First diff is NaN
        
        # Calculate time in profit and loss
        time_in_profit = time_diffs[profit_mask].sum()
        time_in_loss = time_diffs[loss_mask].sum()
        
        logger.debug(f"Time in profit: {time_in_profit:.2f}h, Time in loss: {time_in_loss:.2f}h")
        return time_in_profit, time_in_loss
    
    def calculate_risk_reward_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate risk/reward ratios both planned and achieved.
        
        Returns:
            Tuple: (achieved_rr_ratio, planned_rr_ratio)
        """
        entry_price = self.trade_data['entry_price']
        exit_price = self.trade_data['exit_price']
        quantity = self.trade_data['quantity']
        
        # Calculate achieved risk/reward
        actual_pnl = (exit_price - entry_price) * quantity
        
        # Get planned levels if available
        stop_loss = self.trade_data.get('stop_loss')
        target = self.trade_data.get('target')
        
        achieved_rr_ratio = None
        planned_rr_ratio = None
        
        if stop_loss is not None:
            # Calculate maximum risk (distance to stop loss)
            max_risk = abs((entry_price - stop_loss) * quantity)
            
            if max_risk > 0:
                achieved_rr_ratio = actual_pnl / max_risk if actual_pnl < 0 else actual_pnl / max_risk
        
        if stop_loss is not None and target is not None:
            # Calculate planned risk/reward ratio
            planned_risk = abs((entry_price - stop_loss) * quantity)
            planned_reward = abs((target - entry_price) * quantity)
            
            if planned_risk > 0:
                planned_rr_ratio = planned_reward / planned_risk
        
        logger.debug(f"Achieved R/R: {achieved_rr_ratio}, Planned R/R: {planned_rr_ratio}")
        return achieved_rr_ratio, planned_rr_ratio
    
    def calculate_trade_efficiency(self, mfe_value: float, total_pnl: float) -> float:
        """
        Calculate trade efficiency (how much of potential profit was captured).
        
        Args:
            mfe_value (float): Maximum favorable excursion
            total_pnl (float): Final trade P&L
            
        Returns:
            float: Efficiency ratio (0-1)
        """
        if mfe_value <= 0:
            return 0.0
        
        efficiency = total_pnl / mfe_value if total_pnl > 0 else 0.0
        efficiency = max(0.0, min(1.0, efficiency))  # Clamp between 0 and 1
        
        logger.debug(f"Trade efficiency: {efficiency:.2%}")
        return efficiency
    
    def calculate_risk_efficiency(self, mae_value: float, total_pnl: float) -> float:
        """
        Calculate risk efficiency (how well adverse excursion was managed).
        
        Args:
            mae_value (float): Maximum adverse excursion
            total_pnl (float): Final trade P&L
            
        Returns:
            float: Risk efficiency ratio (0-1)
        """
        if mae_value >= 0:
            return 1.0  # No adverse excursion
        
        if total_pnl >= mae_value:
            # Managed to recover from maximum loss
            efficiency = 1.0 - (abs(mae_value - total_pnl) / abs(mae_value))
        else:
            # Ended worse than maximum adverse point
            efficiency = total_pnl / mae_value
        
        efficiency = max(0.0, min(1.0, efficiency))  # Clamp between 0 and 1
        
        logger.debug(f"Risk efficiency: {efficiency:.2%}")
        return efficiency
    
    def calculate_all_metrics(self) -> TradeVisualizationMetrics:
        """
        Calculate all trade visualization metrics.
        
        Returns:
            TradeVisualizationMetrics: Complete metrics for the trade
        """
        logger.info(f"Calculating metrics for trade {self.trade_data['trade_id']}")
        
        # Basic trade information
        entry_time = self.trade_data['entry_timestamp']
        exit_time = self.trade_data['exit_timestamp']
        entry_price = self.trade_data['entry_price']
        exit_price = self.trade_data['exit_price']
        quantity = self.trade_data['quantity']
        
        # Calculate duration
        duration_hours = (exit_time - entry_time).total_seconds() / 3600
        
        # Calculate total P&L
        total_pnl = (exit_price - entry_price) * quantity
        pnl_percentage = (total_pnl / (entry_price * abs(quantity))) * 100
        
        # Calculate P&L curve
        pnl_curve = self.calculate_pnl_curve()
        
        # Calculate MFE/MAE
        mfe_value, mae_value, mfe_timestamp, mae_timestamp = self.calculate_mfe_mae(pnl_curve)
        mfe_percentage = (mfe_value / (entry_price * abs(quantity))) * 100
        mae_percentage = (mae_value / (entry_price * abs(quantity))) * 100
        
        # Calculate time analysis
        time_in_profit, time_in_loss = self.calculate_time_analysis(pnl_curve)
        total_time = time_in_profit + time_in_loss
        
        time_in_profit_pct = (time_in_profit / total_time * 100) if total_time > 0 else 0
        time_in_loss_pct = (time_in_loss / total_time * 100) if total_time > 0 else 0
        
        # Calculate risk/reward metrics
        achieved_rr, planned_rr = self.calculate_risk_reward_metrics()
        
        # Calculate efficiency metrics
        trade_efficiency = self.calculate_trade_efficiency(mfe_value, total_pnl)
        risk_efficiency = self.calculate_risk_efficiency(mae_value, total_pnl)
        
        # Create metrics object
        metrics = TradeVisualizationMetrics(
            trade_id=self.trade_data['trade_id'],
            symbol=self.trade_data['symbol'],
            entry_timestamp=entry_time,
            exit_timestamp=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            duration_hours=duration_hours,
            total_pnl=total_pnl,
            pnl_percentage=pnl_percentage,
            pnl_curve=pnl_curve,
            mfe_value=mfe_value,
            mae_value=mae_value,
            mfe_timestamp=mfe_timestamp,
            mae_timestamp=mae_timestamp,
            mfe_percentage=mfe_percentage,
            mae_percentage=mae_percentage,
            time_in_profit_hours=time_in_profit,
            time_in_loss_hours=time_in_loss,
            time_in_profit_percentage=time_in_profit_pct,
            time_in_loss_percentage=time_in_loss_pct,
            planned_risk=self.trade_data.get('stop_loss'),
            planned_reward=self.trade_data.get('target'),
            achieved_risk_reward_ratio=achieved_rr,
            planned_risk_reward_ratio=planned_rr,
            trade_efficiency=trade_efficiency,
            risk_efficiency=risk_efficiency
        )
        
        logger.info(f"Metrics calculation completed for trade {self.trade_data['trade_id']}")
        return metrics
    
    def get_formatted_summary(self, metrics: TradeVisualizationMetrics) -> Dict[str, Any]:
        """
        Get a formatted summary of the trade metrics.
        
        Args:
            metrics (TradeVisualizationMetrics): Calculated metrics
            
        Returns:
            Dict: Formatted summary for display
        """
        summary = {
            'Trade ID': metrics.trade_id,
            'Symbol': metrics.symbol,
            'Entry Time': metrics.entry_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Exit Time': metrics.exit_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Duration (hours)': f"{metrics.duration_hours:.1f}",
            'Entry Price': f"₹{metrics.entry_price:.2f}",
            'Exit Price': f"₹{metrics.exit_price:.2f}",
            'Quantity': metrics.quantity,
            'Total P&L': f"₹{metrics.total_pnl:.2f}",
            'P&L %': f"{metrics.pnl_percentage:.2f}%",
            'MFE': f"₹{metrics.mfe_value:.2f} ({metrics.mfe_percentage:.2f}%)",
            'MAE': f"₹{metrics.mae_value:.2f} ({metrics.mae_percentage:.2f}%)",
            'MFE Time': metrics.mfe_timestamp.strftime('%Y-%m-%d %H:%M:%S') if metrics.mfe_timestamp else 'N/A',
            'MAE Time': metrics.mae_timestamp.strftime('%Y-%m-%d %H:%M:%S') if metrics.mae_timestamp else 'N/A',
            'Time in Profit': f"{metrics.time_in_profit_hours:.1f}h ({metrics.time_in_profit_percentage:.1f}%)",
            'Time in Loss': f"{metrics.time_in_loss_hours:.1f}h ({metrics.time_in_loss_percentage:.1f}%)",
            'Trade Efficiency': f"{metrics.trade_efficiency:.1%}",
            'Risk Efficiency': f"{metrics.risk_efficiency:.1%}",
            'Achieved R/R': f"{metrics.achieved_risk_reward_ratio:.2f}" if metrics.achieved_risk_reward_ratio else 'N/A',
            'Planned R/R': f"{metrics.planned_risk_reward_ratio:.2f}" if metrics.planned_risk_reward_ratio else 'N/A'
        }
        
        return summary


def calculate_trade_metrics(price_data: pd.DataFrame, trade_data: Dict[str, Any]) -> TradeVisualizationMetrics:
    """
    Convenience function to calculate trade metrics.
    
    Args:
        price_data (pd.DataFrame): OHLCV price data
        trade_data (Dict): Trade information
        
    Returns:
        TradeVisualizationMetrics: Calculated metrics
    """
    calculator = TradeMetricsCalculator(price_data, trade_data)
    return calculator.calculate_all_metrics()