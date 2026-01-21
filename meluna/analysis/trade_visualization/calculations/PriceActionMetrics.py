# meluna/analysis/trade_visualization/calculations/PriceActionMetrics.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

@dataclass
class VolumeMetrics:
    """Container for volume analysis metrics."""
    volume_at_entry: float
    volume_at_exit: float
    avg_volume_period: float
    volume_ratio_entry: float  # Entry volume / Avg volume
    volume_ratio_exit: float   # Exit volume / Avg volume
    volume_climax_points: List[datetime] = field(default_factory=list)
    volume_dry_up_points: List[datetime] = field(default_factory=list)
    total_volume_traded: float = 0.0

@dataclass
class VolatilityMetrics:
    """Container for volatility analysis metrics."""
    atr_values: pd.Series = field(default_factory=pd.Series)
    avg_atr: float = 0.0
    volatility_at_entry: float = 0.0
    volatility_at_exit: float = 0.0
    high_low_ranges: pd.Series = field(default_factory=pd.Series)
    volatility_expansion_points: List[datetime] = field(default_factory=list)
    volatility_contraction_points: List[datetime] = field(default_factory=list)

@dataclass
class SwingPoints:
    """Container for price swing analysis."""
    swing_highs: List[Tuple[datetime, float]] = field(default_factory=list)
    swing_lows: List[Tuple[datetime, float]] = field(default_factory=list)
    pivot_points: List[Tuple[datetime, float, str]] = field(default_factory=list)  # (time, price, type)

@dataclass
class TrendMetrics:
    """Container for trend analysis metrics."""
    trend_direction: str  # 'uptrend', 'downtrend', 'sideways'
    trend_strength: float  # 0-1 scale
    sma_slope: float
    ema_slope: float
    trend_consistency: float  # How consistent the trend was
    trend_duration_bars: int

@dataclass
class PriceActionAnalysis:
    """Container for comprehensive price action analysis."""
    symbol: str
    analysis_period: Tuple[datetime, datetime]
    volatility_metrics: VolatilityMetrics
    volume_metrics: VolumeMetrics
    swing_points: SwingPoints
    trend_metrics: TrendMetrics
    

class PriceActionCalculator:
    """
    Calculates comprehensive price action metrics for trade visualization.
    
    This class provides detailed analysis including:
    - Bar-by-bar volatility using ATR or high-low range
    - Volume analysis (volume vs average, volume climax detection)
    - Price swing detection (local highs/lows)
    - Trend strength calculation before entry
    """
    
    def __init__(self, price_data: pd.DataFrame, atr_period: int = 14, 
                 volume_lookback: int = 20, trend_periods: List[int] = None):
        """
        Initialize the PriceActionCalculator.
        
        Args:
            price_data (pd.DataFrame): OHLCV data with datetime index
            atr_period (int): Period for ATR calculation
            volume_lookback (int): Period for volume average calculation
            trend_periods (List[int]): Periods for trend analysis [fast, slow]
        """
        self.price_data = price_data.copy()
        self.atr_period = atr_period
        self.volume_lookback = volume_lookback
        self.trend_periods = trend_periods or [10, 20]
        
        # Validate inputs
        self._validate_inputs()
        
        # Ensure datetime index
        self._process_timestamps()
        
        logger.info(f"PriceActionCalculator initialized with {len(self.price_data)} bars")
    
    def _validate_inputs(self) -> None:
        """Validate input data structure."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.price_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if self.price_data.empty:
            raise ValueError("Price data cannot be empty")
        
        if len(self.price_data) < max(self.atr_period, self.volume_lookback):
            raise ValueError(f"Insufficient data points. Need at least {max(self.atr_period, self.volume_lookback)} bars")
    
    def _process_timestamps(self) -> None:
        """Process timestamps and ensure datetime index."""
        if 'timestamp' in self.price_data.columns:
            self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
            self.price_data.set_index('timestamp', inplace=True)
        elif not isinstance(self.price_data.index, pd.DatetimeIndex):
            raise ValueError("Price data must have datetime index or timestamp column")
        
        # Sort by timestamp
        self.price_data.sort_index(inplace=True)
    
    def calculate_atr(self) -> pd.Series:
        """
        Calculate Average True Range for volatility analysis.
        
        Returns:
            pd.Series: ATR values indexed by timestamp
        """
        high = self.price_data['high']
        low = self.price_data['low']
        close = self.price_data['close']
        prev_close = close.shift(1)
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as simple moving average of True Range
        atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()
        
        logger.debug(f"ATR calculated with period {self.atr_period}")
        return atr
    
    def calculate_high_low_ranges(self) -> pd.Series:
        """
        Calculate high-low ranges as alternative volatility measure.
        
        Returns:
            pd.Series: High-low range percentages
        """
        high = self.price_data['high']
        low = self.price_data['low']
        close = self.price_data['close']
        
        # Calculate range as percentage of close
        hl_range = ((high - low) / close) * 100
        
        logger.debug("High-low ranges calculated")
        return hl_range
    
    def detect_volatility_patterns(self, atr_values: pd.Series) -> Tuple[List[datetime], List[datetime]]:
        """
        Detect volatility expansion and contraction points.
        
        Args:
            atr_values (pd.Series): ATR values
            
        Returns:
            Tuple: (expansion_points, contraction_points)
        """
        if len(atr_values) < self.atr_period:
            return [], []
        
        # Calculate ATR rolling statistics
        atr_ma = atr_values.rolling(window=self.atr_period).mean()
        atr_std = atr_values.rolling(window=self.atr_period).std()
        
        # Define expansion and contraction thresholds
        expansion_threshold = atr_ma + (atr_std * 1.5)
        contraction_threshold = atr_ma - (atr_std * 1.0)
        
        # Find expansion points (ATR significantly above average)
        expansion_mask = atr_values > expansion_threshold
        expansion_points = atr_values[expansion_mask].index.tolist()
        
        # Find contraction points (ATR significantly below average)
        contraction_mask = atr_values < contraction_threshold
        contraction_points = atr_values[contraction_mask].index.tolist()
        
        logger.debug(f"Found {len(expansion_points)} expansion and {len(contraction_points)} contraction points")
        return expansion_points, contraction_points
    
    def calculate_volume_metrics(self) -> VolumeMetrics:
        """
        Calculate comprehensive volume analysis metrics.
        
        Returns:
            VolumeMetrics: Complete volume analysis
        """
        volume = self.price_data['volume']
        
        # Calculate rolling average volume
        avg_volume = volume.rolling(window=self.volume_lookback, min_periods=1).mean()
        
        # Volume ratios
        volume_ratios = volume / avg_volume
        
        # Detect volume climax (unusually high volume)
        climax_threshold = volume_ratios.quantile(0.95)
        climax_mask = volume_ratios > climax_threshold
        climax_points = volume[climax_mask].index.tolist()
        
        # Detect volume dry-up (unusually low volume)
        dryup_threshold = volume_ratios.quantile(0.05)
        dryup_mask = volume_ratios < dryup_threshold
        dryup_points = volume[dryup_mask].index.tolist()
        
        # Get first and last values for entry/exit analysis
        volume_at_entry = volume.iloc[0] if len(volume) > 0 else 0
        volume_at_exit = volume.iloc[-1] if len(volume) > 0 else 0
        avg_volume_period = avg_volume.mean()
        
        volume_ratio_entry = volume_at_entry / avg_volume_period if avg_volume_period > 0 else 1.0
        volume_ratio_exit = volume_at_exit / avg_volume_period if avg_volume_period > 0 else 1.0
        
        total_volume = volume.sum()
        
        metrics = VolumeMetrics(
            volume_at_entry=volume_at_entry,
            volume_at_exit=volume_at_exit,
            avg_volume_period=avg_volume_period,
            volume_ratio_entry=volume_ratio_entry,
            volume_ratio_exit=volume_ratio_exit,
            volume_climax_points=climax_points,
            volume_dry_up_points=dryup_points,
            total_volume_traded=total_volume
        )
        
        logger.debug(f"Volume metrics calculated: {len(climax_points)} climax, {len(dryup_points)} dry-up points")
        return metrics
    
    def detect_swing_points(self, prominence_threshold: float = 0.5) -> SwingPoints:
        """
        Detect price swing highs and lows using peak detection.
        
        Args:
            prominence_threshold (float): Minimum prominence for peak detection
            
        Returns:
            SwingPoints: Detected swing highs and lows
        """
        high_prices = self.price_data['high'].values
        low_prices = self.price_data['low'].values
        timestamps = self.price_data.index
        
        # Detect swing highs
        high_peaks, _ = find_peaks(high_prices, prominence=prominence_threshold)
        swing_highs = [(timestamps[i], high_prices[i]) for i in high_peaks]
        
        # Detect swing lows (invert the data)
        low_peaks, _ = find_peaks(-low_prices, prominence=prominence_threshold)
        swing_lows = [(timestamps[i], low_prices[i]) for i in low_peaks]
        
        # Combine into pivot points
        pivot_points = []
        for time, price in swing_highs:
            pivot_points.append((time, price, 'high'))
        for time, price in swing_lows:
            pivot_points.append((time, price, 'low'))
        
        # Sort by timestamp
        pivot_points.sort(key=lambda x: x[0])
        
        swing_data = SwingPoints(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            pivot_points=pivot_points
        )
        
        logger.debug(f"Detected {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
        return swing_data
    
    def calculate_trend_metrics(self) -> TrendMetrics:
        """
        Calculate trend strength and direction metrics.
        
        Returns:
            TrendMetrics: Comprehensive trend analysis
        """
        close_prices = self.price_data['close']
        
        if len(close_prices) < max(self.trend_periods):
            # Insufficient data for trend analysis
            return TrendMetrics(
                trend_direction='sideways',
                trend_strength=0.0,
                sma_slope=0.0,
                ema_slope=0.0,
                trend_consistency=0.0,
                trend_duration_bars=len(close_prices)
            )
        
        # Calculate moving averages
        fast_sma = close_prices.rolling(window=self.trend_periods[0]).mean()
        slow_sma = close_prices.rolling(window=self.trend_periods[1]).mean()
        
        fast_ema = close_prices.ewm(span=self.trend_periods[0]).mean()
        slow_ema = close_prices.ewm(span=self.trend_periods[1]).mean()
        
        # Calculate slopes (rate of change)
        sma_slope = self._calculate_slope(slow_sma.dropna())
        ema_slope = self._calculate_slope(slow_ema.dropna())
        
        # Determine trend direction
        latest_fast_sma = fast_sma.iloc[-1] if not fast_sma.empty else 0
        latest_slow_sma = slow_sma.iloc[-1] if not slow_sma.empty else 0
        latest_fast_ema = fast_ema.iloc[-1] if not fast_ema.empty else 0
        latest_slow_ema = slow_ema.iloc[-1] if not slow_ema.empty else 0
        
        # Combined trend signal
        sma_signal = 1 if latest_fast_sma > latest_slow_sma else -1
        ema_signal = 1 if latest_fast_ema > latest_slow_ema else -1
        slope_signal = 1 if (sma_slope + ema_slope) > 0 else -1
        
        trend_votes = sma_signal + ema_signal + slope_signal
        
        if trend_votes >= 2:
            trend_direction = 'uptrend'
        elif trend_votes <= -2:
            trend_direction = 'downtrend'
        else:
            trend_direction = 'sideways'
        
        # Calculate trend strength (0-1)
        trend_strength = abs(trend_votes) / 3.0
        
        # Calculate trend consistency
        trend_consistency = self._calculate_trend_consistency(fast_sma, slow_sma)
        
        metrics = TrendMetrics(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            sma_slope=sma_slope,
            ema_slope=ema_slope,
            trend_consistency=trend_consistency,
            trend_duration_bars=len(close_prices)
        )
        
        logger.debug(f"Trend analysis: {trend_direction} with strength {trend_strength:.2f}")
        return metrics
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate the slope of a time series using linear regression."""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        try:
            slope, _, _, _, _ = stats.linregress(x_clean, y_clean)
            return slope
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_trend_consistency(self, fast_ma: pd.Series, slow_ma: pd.Series) -> float:
        """Calculate how consistent the trend direction has been."""
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            return 0.0
        
        # Calculate the difference between fast and slow MA
        ma_diff = fast_ma - slow_ma
        ma_diff_clean = ma_diff.dropna()
        
        if len(ma_diff_clean) == 0:
            return 0.0
        
        # Count how many times the trend direction changed
        trend_changes = (ma_diff_clean * ma_diff_clean.shift(1) < 0).sum()
        total_periods = len(ma_diff_clean) - 1
        
        if total_periods == 0:
            return 1.0
        
        # Consistency is 1 - (change_rate)
        consistency = 1.0 - (trend_changes / total_periods)
        return max(0.0, min(1.0, consistency))
    
    def calculate_all_metrics(self, symbol: str = "UNKNOWN") -> PriceActionAnalysis:
        """
        Calculate all price action metrics.
        
        Args:
            symbol (str): Symbol identifier
            
        Returns:
            PriceActionAnalysis: Complete price action analysis
        """
        logger.info(f"Calculating price action metrics for {symbol}")
        
        # Analysis period
        start_time = self.price_data.index[0]
        end_time = self.price_data.index[-1]
        analysis_period = (start_time, end_time)
        
        # Calculate ATR and volatility metrics
        atr_values = self.calculate_atr()
        hl_ranges = self.calculate_high_low_ranges()
        expansion_points, contraction_points = self.detect_volatility_patterns(atr_values)
        
        volatility_metrics = VolatilityMetrics(
            atr_values=atr_values,
            avg_atr=atr_values.mean(),
            volatility_at_entry=atr_values.iloc[0] if len(atr_values) > 0 else 0,
            volatility_at_exit=atr_values.iloc[-1] if len(atr_values) > 0 else 0,
            high_low_ranges=hl_ranges,
            volatility_expansion_points=expansion_points,
            volatility_contraction_points=contraction_points
        )
        
        # Calculate volume metrics
        volume_metrics = self.calculate_volume_metrics()
        
        # Detect swing points
        swing_points = self.detect_swing_points()
        
        # Calculate trend metrics
        trend_metrics = self.calculate_trend_metrics()
        
        # Create comprehensive analysis
        analysis = PriceActionAnalysis(
            symbol=symbol,
            analysis_period=analysis_period,
            volatility_metrics=volatility_metrics,
            volume_metrics=volume_metrics,
            swing_points=swing_points,
            trend_metrics=trend_metrics
        )
        
        logger.info(f"Price action analysis completed for {symbol}")
        return analysis
    
    def get_formatted_summary(self, analysis: PriceActionAnalysis) -> Dict[str, Any]:
        """
        Get a formatted summary of the price action analysis.
        
        Args:
            analysis (PriceActionAnalysis): Complete analysis
            
        Returns:
            Dict: Formatted summary for display
        """
        summary = {
            'Symbol': analysis.symbol,
            'Analysis Period': f"{analysis.analysis_period[0].strftime('%Y-%m-%d')} to {analysis.analysis_period[1].strftime('%Y-%m-%d')}",
            'Total Bars': len(self.price_data),
            
            # Volatility metrics
            'Avg ATR': f"{analysis.volatility_metrics.avg_atr:.4f}",
            'Entry ATR': f"{analysis.volatility_metrics.volatility_at_entry:.4f}",
            'Exit ATR': f"{analysis.volatility_metrics.volatility_at_exit:.4f}",
            'Volatility Expansions': len(analysis.volatility_metrics.volatility_expansion_points),
            'Volatility Contractions': len(analysis.volatility_metrics.volatility_contraction_points),
            
            # Volume metrics
            'Entry Volume': f"{analysis.volume_metrics.volume_at_entry:,.0f}",
            'Exit Volume': f"{analysis.volume_metrics.volume_at_exit:,.0f}",
            'Avg Volume': f"{analysis.volume_metrics.avg_volume_period:,.0f}",
            'Entry Volume Ratio': f"{analysis.volume_metrics.volume_ratio_entry:.2f}x",
            'Exit Volume Ratio': f"{analysis.volume_metrics.volume_ratio_exit:.2f}x",
            'Volume Climax Points': len(analysis.volume_metrics.volume_climax_points),
            'Volume Dry-up Points': len(analysis.volume_metrics.volume_dry_up_points),
            
            # Swing points
            'Swing Highs': len(analysis.swing_points.swing_highs),
            'Swing Lows': len(analysis.swing_points.swing_lows),
            'Total Pivot Points': len(analysis.swing_points.pivot_points),
            
            # Trend metrics
            'Trend Direction': analysis.trend_metrics.trend_direction.title(),
            'Trend Strength': f"{analysis.trend_metrics.trend_strength:.1%}",
            'SMA Slope': f"{analysis.trend_metrics.sma_slope:.6f}",
            'EMA Slope': f"{analysis.trend_metrics.ema_slope:.6f}",
            'Trend Consistency': f"{analysis.trend_metrics.trend_consistency:.1%}"
        }
        
        return summary


def calculate_price_action_metrics(price_data: pd.DataFrame, symbol: str = "UNKNOWN", 
                                 atr_period: int = 14, volume_lookback: int = 20) -> PriceActionAnalysis:
    """
    Convenience function to calculate price action metrics.
    
    Args:
        price_data (pd.DataFrame): OHLCV price data
        symbol (str): Symbol identifier
        atr_period (int): Period for ATR calculation
        volume_lookback (int): Period for volume analysis
        
    Returns:
        PriceActionAnalysis: Complete price action analysis
    """
    calculator = PriceActionCalculator(price_data, atr_period, volume_lookback)
    return calculator.calculate_all_metrics(symbol)