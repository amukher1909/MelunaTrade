# meluna/analysis/trade_visualization/TradeDataExtractor.py

"""
Trade Data Integration and Context Window System for Individual Trade Visualization.

This module provides specialized data extraction capabilities for individual trades,
building on the existing DashboardDataService to provide trade-specific data preparation
with configurable context windows.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import lru_cache
import warnings

from ..dashboard.DashboardDataService import DashboardDataService, CachedAnalyticsData
from .base_interfaces import TradeVisualizationConfig

logger = logging.getLogger(__name__)


@dataclass
class ContextWindowConfig:
    """Configuration for trade context window extraction."""
    bars_before_entry: int = 50
    bars_after_exit: int = 20
    min_bars_before: int = 20
    max_bars_before: int = 200
    min_bars_after: int = 10
    max_bars_after: int = 100
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Clamp bars_before_entry to valid range
        if self.bars_before_entry < self.min_bars_before:
            logger.warning(f"bars_before_entry ({self.bars_before_entry}) below minimum ({self.min_bars_before}), adjusting")
            self.bars_before_entry = self.min_bars_before
        elif self.bars_before_entry > self.max_bars_before:
            logger.warning(f"bars_before_entry ({self.bars_before_entry}) above maximum ({self.max_bars_before}), adjusting")
            self.bars_before_entry = self.max_bars_before
        
        # Clamp bars_after_exit to valid range
        if self.bars_after_exit < self.min_bars_after:
            logger.warning(f"bars_after_exit ({self.bars_after_exit}) below minimum ({self.min_bars_after}), adjusting")
            self.bars_after_exit = self.min_bars_after
        elif self.bars_after_exit > self.max_bars_after:
            logger.warning(f"bars_after_exit ({self.bars_after_exit}) above maximum ({self.max_bars_after}), adjusting")
            self.bars_after_exit = self.max_bars_after


@dataclass
class TradeDataWindow:
    """Container for extracted trade data with context window."""
    trade_id: str
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    quantity: float
    trade_pnl: float
    
    # Context window data
    context_ohlcv: pd.DataFrame
    context_start_time: datetime
    context_end_time: datetime
    
    # Position indices within context
    entry_index: int
    exit_index: int
    
    # Data quality metrics
    bars_before_actual: int
    bars_after_actual: int
    missing_data_periods: List[Tuple[datetime, datetime]]
    data_quality_score: float  # 0.0 to 1.0


@dataclass
class ExtractionResult:
    """Container for extraction operation results."""
    success: bool
    trade_window: Optional[TradeDataWindow]
    error_message: str = ""
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TradeDataExtractor:
    """
    Specialized data extraction layer for Individual Trade Visualization.
    
    Extracts and prepares OHLCV data with trade context for single-trade analysis,
    building on the existing DashboardDataService infrastructure.
    
    Features:
    - Trade period extraction with configurable context windows
    - OHLCV data alignment with trade timestamps
    - Data validation and missing data handling
    - Performance optimization with caching
    - Edge case handling for trades near data boundaries
    """
    
    def __init__(self, 
                 dashboard_service: Optional[DashboardDataService] = None,
                 results_directory: str = "results",
                 data_directory: str = "data",
                 enable_caching: bool = True):
        """
        Initialize TradeDataExtractor.
        
        Args:
            dashboard_service: Existing DashboardDataService instance or None to create new
            results_directory: Path to backtest results directory
            data_directory: Path to OHLCV data directory
            enable_caching: Enable extraction result caching
        """
        # Initialize or use existing DashboardDataService
        if dashboard_service is None:
            self.dashboard_service = DashboardDataService(
                results_directory=results_directory,
                enable_threading=True
            )
        else:
            self.dashboard_service = dashboard_service
        
        self.data_directory = Path(data_directory)
        self.enable_caching = enable_caching
        
        # Initialize caches
        self._ohlcv_cache = {}
        self._extraction_cache = {}
        self._cache_max_age = timedelta(hours=2)
        
        # Initialize symbol mapping for demo/test data
        self._symbol_mapping = self._create_symbol_mapping()
        
        logger.info(f"TradeDataExtractor initialized with data_directory: {self.data_directory}")
        
        # Validate data directory
        if not self.data_directory.exists():
            logger.warning(f"Data directory does not exist: {self.data_directory}")
    
    def _create_symbol_mapping(self) -> Dict[str, str]:
        """
        Create mapping from trade symbols to actual OHLCV data files.
        
        This method auto-detects available OHLCV files and maps synthetic trade symbols
        to real data files for demo/testing purposes.
        
        Returns:
            Dictionary mapping trade symbols to OHLCV file symbols
        """
        mapping = {}
        
        try:
            # Find available OHLCV files
            available_files = []
            if self.data_directory.exists():
                for file_path in self.data_directory.iterdir():
                    if file_path.suffix in ['.parquet', '.csv']:
                        symbol = file_path.stem
                        available_files.append(symbol)
            
            if available_files:
                # Map all synthetic symbols to the first available real data file
                # This is for demo purposes - in production this would be a proper mapping
                primary_symbol = available_files[0]
                synthetic_symbols = [f"STOCK{i:02d}" for i in range(10)]
                
                for sym in synthetic_symbols:
                    mapping[sym] = primary_symbol
                
                logger.info(f"Created symbol mapping: {len(synthetic_symbols)} symbols -> {primary_symbol}")
            else:
                logger.warning("No OHLCV data files found for symbol mapping")
                
        except Exception as e:
            logger.error(f"Error creating symbol mapping: {e}")
        
        return mapping
    
    def _get_mapped_symbol(self, trade_symbol: str) -> str:
        """
        Get the mapped OHLCV symbol for a trade symbol.
        
        Args:
            trade_symbol: Symbol from trade data
            
        Returns:
            Mapped symbol for OHLCV data lookup
        """
        return self._symbol_mapping.get(trade_symbol, trade_symbol)
    
    def extract_trade_data(self, 
                          strategy: str, 
                          version: str,
                          trade_id: str,
                          symbol: str,
                          context_config: Optional[ContextWindowConfig] = None) -> ExtractionResult:
        """
        Extract OHLCV data for a specific trade with context window.
        
        Args:
            strategy: Strategy name
            version: Version identifier
            trade_id: Unique trade identifier
            symbol: Trading symbol
            context_config: Context window configuration
            
        Returns:
            ExtractionResult containing trade data window or error information
        """
        if context_config is None:
            context_config = ContextWindowConfig()
        
        try:
            # Generate cache key
            cache_key = f"{strategy}_{version}_{trade_id}_{symbol}_{hash(str(asdict(context_config)))}"
            
            # Check cache first
            if self.enable_caching and cache_key in self._extraction_cache:
                cached_result = self._extraction_cache[cache_key]
                cache_age = datetime.now() - cached_result['timestamp']
                if cache_age < self._cache_max_age:
                    logger.debug(f"Using cached extraction result for trade {trade_id}")
                    return cached_result['result']
            
            # Load analytics data to get trade information
            analytics_data = self.dashboard_service.load_analytics_data(strategy, version)
            if analytics_data is None:
                return ExtractionResult(
                    success=False,
                    trade_window=None,
                    error_message=f"Failed to load analytics data for {strategy}/{version}"
                )
            
            # Find the specific trade
            trade_info = self._find_trade(analytics_data, trade_id, symbol)
            if trade_info is None:
                return ExtractionResult(
                    success=False,
                    trade_window=None,
                    error_message=f"Trade {trade_id} not found for symbol {symbol}"
                )
            
            # Load OHLCV data for the symbol (with mapping)
            mapped_symbol = self._get_mapped_symbol(symbol)
            ohlcv_data = self._load_ohlcv_data(mapped_symbol)
            if ohlcv_data is None or ohlcv_data.empty:
                return ExtractionResult(
                    success=False,
                    trade_window=None,
                    error_message=f"Failed to load OHLCV data for symbol {symbol}"
                )
            
            # Extract context window
            trade_window = self._extract_context_window(
                trade_info, ohlcv_data, context_config
            )
            
            # Cache the result
            result = ExtractionResult(success=True, trade_window=trade_window)
            if self.enable_caching:
                self._extraction_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting trade data for {trade_id}: {e}")
            return ExtractionResult(
                success=False,
                trade_window=None,
                error_message=f"Extraction failed: {str(e)}"
            )
    
    def extract_multiple_trades(self,
                               strategy: str,
                               version: str,
                               trade_ids: List[str],
                               symbol: str,
                               context_config: Optional[ContextWindowConfig] = None) -> Dict[str, ExtractionResult]:
        """
        Extract data for multiple trades efficiently.
        
        Args:
            strategy: Strategy name
            version: Version identifier
            trade_ids: List of trade identifiers
            symbol: Trading symbol
            context_config: Context window configuration
            
        Returns:
            Dictionary mapping trade_id to ExtractionResult
        """
        results = {}
        
        try:
            # Pre-load analytics data and OHLCV data once
            analytics_data = self.dashboard_service.load_analytics_data(strategy, version)
            if analytics_data is None:
                error_result = ExtractionResult(
                    success=False,
                    trade_window=None,
                    error_message=f"Failed to load analytics data for {strategy}/{version}"
                )
                return {trade_id: error_result for trade_id in trade_ids}
            
            mapped_symbol = self._get_mapped_symbol(symbol)
            ohlcv_data = self._load_ohlcv_data(mapped_symbol)
            if ohlcv_data is None or ohlcv_data.empty:
                error_result = ExtractionResult(
                    success=False,
                    trade_window=None,
                    error_message=f"Failed to load OHLCV data for symbol {symbol} (mapped to {mapped_symbol})"
                )
                return {trade_id: error_result for trade_id in trade_ids}
            
            # Extract each trade
            for trade_id in trade_ids:
                try:
                    trade_info = self._find_trade(analytics_data, trade_id, symbol)
                    if trade_info is None:
                        results[trade_id] = ExtractionResult(
                            success=False,
                            trade_window=None,
                            error_message=f"Trade {trade_id} not found"
                        )
                        continue
                    
                    trade_window = self._extract_context_window(
                        trade_info, ohlcv_data, context_config or ContextWindowConfig()
                    )
                    
                    results[trade_id] = ExtractionResult(
                        success=True,
                        trade_window=trade_window
                    )
                    
                except Exception as e:
                    logger.error(f"Error extracting trade {trade_id}: {e}")
                    results[trade_id] = ExtractionResult(
                        success=False,
                        trade_window=None,
                        error_message=str(e)
                    )
            
        except Exception as e:
            logger.error(f"Error in multiple trade extraction: {e}")
            error_result = ExtractionResult(
                success=False,
                trade_window=None,
                error_message=str(e)
            )
            results = {trade_id: error_result for trade_id in trade_ids}
        
        return results
    
    def find_similar_trades(self,
                           strategy: str,
                           version: str,
                           reference_trade_id: str,
                           symbol: str,
                           similarity_criteria: Dict[str, Any] = None) -> List[str]:
        """
        Find trades similar to a reference trade for comparative analysis.
        
        Args:
            strategy: Strategy name
            version: Version identifier
            reference_trade_id: Reference trade ID
            symbol: Trading symbol
            similarity_criteria: Criteria for similarity (duration, pnl_range, etc.)
            
        Returns:
            List of similar trade IDs
        """
        try:
            analytics_data = self.dashboard_service.load_analytics_data(strategy, version)
            if analytics_data is None or analytics_data.trade_log is None:
                return []
            
            trade_log = analytics_data.trade_log
            
            # Find reference trade
            ref_trade = trade_log[trade_log['trade_id'] == reference_trade_id]
            if ref_trade.empty:
                logger.warning(f"Reference trade {reference_trade_id} not found")
                return []
            
            ref_trade = ref_trade.iloc[0]
            
            # Default similarity criteria
            if similarity_criteria is None:
                similarity_criteria = {
                    'duration_tolerance': timedelta(hours=24),  # ±24 hours
                    'pnl_tolerance': 0.2,  # ±20% of reference P&L
                    'max_results': 10
                }
            
            # Filter similar trades
            symbol_trades = trade_log[trade_log['symbol'] == symbol].copy()
            similar_trades = []
            
            ref_duration = ref_trade['exit_timestamp'] - ref_trade['entry_timestamp']
            ref_pnl = ref_trade['realized_pnl']
            
            for _, trade in symbol_trades.iterrows():
                if trade['trade_id'] == reference_trade_id:
                    continue
                
                # Check duration similarity
                trade_duration = trade['exit_timestamp'] - trade['entry_timestamp']
                duration_diff = abs(trade_duration - ref_duration)
                
                if duration_diff <= similarity_criteria.get('duration_tolerance', timedelta(hours=24)):
                    # Check P&L similarity
                    pnl_diff = abs(trade['realized_pnl'] - ref_pnl) / abs(ref_pnl) if ref_pnl != 0 else 0
                    
                    if pnl_diff <= similarity_criteria.get('pnl_tolerance', 0.2):
                        similar_trades.append(trade['trade_id'])
            
            # Limit results
            max_results = similarity_criteria.get('max_results', 10)
            return similar_trades[:max_results]
            
        except Exception as e:
            logger.error(f"Error finding similar trades: {e}")
            return []
    
    def validate_trade_data_availability(self,
                                       strategy: str,
                                       version: str,
                                       symbol: str) -> Dict[str, Any]:
        """
        Validate data availability for trade extraction.
        
        Args:
            strategy: Strategy name
            version: Version identifier
            symbol: Trading symbol
            
        Returns:
            Validation report dictionary
        """
        report = {
            'strategy_data_available': False,
            'ohlcv_data_available': False,
            'trade_count': 0,
            'date_range': None,
            'data_quality_issues': [],
            'recommendations': []
        }
        
        try:
            # Check strategy data
            analytics_data = self.dashboard_service.load_analytics_data(strategy, version)
            if analytics_data is not None and analytics_data.trade_log is not None:
                report['strategy_data_available'] = True
                
                symbol_trades = analytics_data.trade_log[
                    analytics_data.trade_log['symbol'] == symbol
                ]
                report['trade_count'] = len(symbol_trades)
                
                if not symbol_trades.empty:
                    report['date_range'] = (
                        symbol_trades['entry_timestamp'].min(),
                        symbol_trades['exit_timestamp'].max()
                    )
            
            # Check OHLCV data
            ohlcv_data = self._load_ohlcv_data(symbol)
            if ohlcv_data is not None and not ohlcv_data.empty:
                report['ohlcv_data_available'] = True
                
                # Check for data gaps
                if isinstance(ohlcv_data.index, pd.DatetimeIndex):
                    time_diffs = ohlcv_data.index.to_series().diff()
                    median_diff = time_diffs.median()
                    large_gaps = time_diffs[time_diffs > median_diff * 3]
                    
                    if len(large_gaps) > 0:
                        report['data_quality_issues'].append(
                            f"Found {len(large_gaps)} potential data gaps"
                        )
            
            # Generate recommendations
            if not report['strategy_data_available']:
                report['recommendations'].append(
                    "Strategy data not available - check results directory"
                )
            
            if not report['ohlcv_data_available']:
                report['recommendations'].append(
                    f"OHLCV data not available for {symbol} - check data directory"
                )
            
            if report['trade_count'] == 0:
                report['recommendations'].append(
                    f"No trades found for symbol {symbol} in this strategy"
                )
            
        except Exception as e:
            logger.error(f"Error validating trade data availability: {e}")
            report['data_quality_issues'].append(f"Validation error: {str(e)}")
        
        return report
    
    @lru_cache(maxsize=10)
    def _load_ohlcv_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a symbol with caching.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            # Try different file formats and locations
            possible_files = [
                self.data_directory / f"{symbol}.parquet",
                self.data_directory / f"{symbol}.csv",
                self.data_directory / f"{symbol.upper()}.parquet",
                self.data_directory / f"{symbol.upper()}.csv",
                self.data_directory / "processed" / f"{symbol}.parquet",
                self.data_directory / "processed" / f"{symbol}.csv"
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    try:
                        if file_path.suffix == '.parquet':
                            df = pd.read_parquet(file_path)
                        else:
                            df = pd.read_csv(file_path)
                        
                        # Standardize column names
                        df.columns = df.columns.str.lower()
                        
                        # Ensure datetime index
                        if not isinstance(df.index, pd.DatetimeIndex):
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                df.set_index('timestamp', inplace=True)
                            elif 'date' in df.columns:
                                df['date'] = pd.to_datetime(df['date'])
                                df.set_index('date', inplace=True)
                            else:
                                logger.warning(f"No timestamp column found in {file_path}")
                                continue
                        
                        # Validate required columns
                        required_cols = ['open', 'high', 'low', 'close']
                        if not all(col in df.columns for col in required_cols):
                            logger.warning(f"Missing required OHLC columns in {file_path}")
                            continue
                        
                        # Sort by timestamp
                        df.sort_index(inplace=True)
                        
                        logger.info(f"Successfully loaded OHLCV data for {symbol} from {file_path}")
                        return df
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
                        continue
            
            logger.error(f"Could not find OHLCV data for symbol {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading OHLCV data for {symbol}: {e}")
            return None
    
    def _find_trade(self, 
                   analytics_data: CachedAnalyticsData, 
                   trade_id: str, 
                   symbol: str) -> Optional[pd.Series]:
        """
        Find a specific trade in the analytics data.
        
        Args:
            analytics_data: Cached analytics data
            trade_id: Trade identifier
            symbol: Trading symbol
            
        Returns:
            Trade information as pandas Series or None if not found
        """
        if analytics_data.trade_log is None:
            return None
        
        # Find trade by ID and symbol
        trade_mask = (
            (analytics_data.trade_log['trade_id'] == trade_id) &
            (analytics_data.trade_log['symbol'] == symbol)
        )
        
        matching_trades = analytics_data.trade_log[trade_mask]
        
        if matching_trades.empty:
            return None
        
        if len(matching_trades) > 1:
            logger.warning(f"Multiple trades found for ID {trade_id}, using first match")
        
        return matching_trades.iloc[0]
    
    def _extract_context_window(self, 
                               trade_info: pd.Series, 
                               ohlcv_data: pd.DataFrame,
                               context_config: ContextWindowConfig) -> TradeDataWindow:
        """
        Extract context window around a trade.
        
        Args:
            trade_info: Trade information
            ohlcv_data: OHLCV data
            context_config: Context window configuration
            
        Returns:
            TradeDataWindow with extracted data
        """
        entry_time = pd.to_datetime(trade_info['entry_timestamp'])
        exit_time = pd.to_datetime(trade_info['exit_timestamp'])
        
        # Find nearest indices for entry and exit times
        entry_idx = ohlcv_data.index.get_indexer([entry_time], method='nearest')[0]
        exit_idx = ohlcv_data.index.get_indexer([exit_time], method='nearest')[0]
        
        # Handle case where trade times are outside OHLCV data range
        if entry_idx == -1 or exit_idx == -1:
            raise ValueError(f"Trade times outside OHLCV data range")
        
        # Calculate context window bounds
        start_idx = max(0, entry_idx - context_config.bars_before_entry)
        end_idx = min(len(ohlcv_data) - 1, exit_idx + context_config.bars_after_exit)
        
        # Extract context data
        context_ohlcv = ohlcv_data.iloc[start_idx:end_idx + 1].copy()
        
        # Calculate actual bars extracted
        bars_before_actual = entry_idx - start_idx
        bars_after_actual = end_idx - exit_idx
        
        # Detect missing data periods
        missing_periods = self._detect_missing_data_periods(context_ohlcv)
        
        # Calculate data quality score
        expected_bars = context_config.bars_before_entry + context_config.bars_after_exit + (exit_idx - entry_idx + 1)
        actual_bars = len(context_ohlcv)
        data_quality_score = min(1.0, actual_bars / expected_bars) if expected_bars > 0 else 0.0
        
        # Adjust for missing data
        if missing_periods:
            data_quality_score *= 0.8  # Penalize for data gaps
        
        return TradeDataWindow(
            trade_id=trade_info['trade_id'],
            symbol=trade_info['symbol'],
            entry_timestamp=entry_time,
            exit_timestamp=exit_time,
            entry_price=trade_info['entry_price'],
            exit_price=trade_info['exit_price'],
            quantity=trade_info['quantity'],
            trade_pnl=trade_info['realized_pnl'],
            context_ohlcv=context_ohlcv,
            context_start_time=context_ohlcv.index[0],
            context_end_time=context_ohlcv.index[-1],
            entry_index=entry_idx - start_idx,
            exit_index=exit_idx - start_idx,
            bars_before_actual=bars_before_actual,
            bars_after_actual=bars_after_actual,
            missing_data_periods=missing_periods,
            data_quality_score=data_quality_score
        )
    
    def _detect_missing_data_periods(self, ohlcv_data: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """
        Detect periods with missing data in OHLCV dataset.
        
        Args:
            ohlcv_data: OHLCV data to analyze
            
        Returns:
            List of (start_time, end_time) tuples for missing periods
        """
        if len(ohlcv_data) < 2:
            return []
        
        try:
            # Calculate time differences between consecutive bars
            time_diffs = ohlcv_data.index.to_series().diff()
            median_diff = time_diffs.median()
            
            # Find gaps larger than expected
            large_gaps = time_diffs[time_diffs > median_diff * 2]
            
            missing_periods = []
            for gap_end_time, gap_duration in large_gaps.items():
                gap_start_time = gap_end_time - gap_duration
                missing_periods.append((gap_start_time, gap_end_time))
            
            return missing_periods
            
        except Exception as e:
            logger.warning(f"Error detecting missing data periods: {e}")
            return []
    
    def clear_cache(self):
        """Clear all cached data."""
        self._ohlcv_cache.clear()
        self._extraction_cache.clear()
        # Also clear LRU cache
        self._load_ohlcv_data.cache_clear()
        logger.info("Cleared TradeDataExtractor cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'ohlcv_cache_size': len(self._ohlcv_cache),
            'extraction_cache_size': len(self._extraction_cache),
            'lru_cache_info': self._load_ohlcv_data.cache_info()._asdict(),
            'cache_max_age_hours': self._cache_max_age.total_seconds() / 3600
        }