# meluna/analysis/MetricsEngine.py

import pandas as pd
import numpy as np
import logging
import threading
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import weakref
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class DataSchema:
    """
    Defines expected schemas for data validation.
    """
    required_columns: list
    optional_columns: list = None
    column_types: Dict[str, type] = None
    
    def __post_init__(self):
        if self.optional_columns is None:
            self.optional_columns = []
        if self.column_types is None:
            self.column_types = {}


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class FileNotFoundError(Exception):
    """Raised when required data files are not found."""
    pass


class DataValidator:
    """
    Validates parquet file schemas and data integrity.
    """
    
    # Define expected schemas
    TRADE_LOG_SCHEMA = DataSchema(
        required_columns=['trade_id', 'symbol', 'entry_timestamp', 'exit_timestamp', 
                         'entry_price', 'exit_price', 'quantity', 'pnl'],
        optional_columns=['side', 'fees', 'commission'],
        column_types={
            'trade_id': (str, int),
            'symbol': str,
            'entry_price': (float, int),
            'exit_price': (float, int),
            'quantity': (float, int),
            'pnl': (float, int)
        }
    )
    
    EQUITY_CURVE_SCHEMA = DataSchema(
        required_columns=['date'],
        optional_columns=['equity', 'Equity', 'value', 'portfolio_value'],
        column_types={
            'date': (str, 'datetime64[ns]')
        }
    )
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, schema: DataSchema, data_type: str) -> None:
        """
        Validates a DataFrame against a schema.
        
        Args:
            df: DataFrame to validate
            schema: Schema to validate against
            data_type: Type of data for error messages
            
        Raises:
            DataValidationError: If validation fails
        """
        if df.empty:
            raise DataValidationError(f"{data_type} DataFrame is empty")
        
        # Check required columns
        missing_columns = set(schema.required_columns) - set(df.columns)
        if missing_columns:
            raise DataValidationError(
                f"{data_type} missing required columns: {missing_columns}"
            )
        
        # Validate column types if specified
        for col, expected_types in schema.column_types.items():
            if col in df.columns:
                if not isinstance(expected_types, tuple):
                    expected_types = (expected_types,)
                
                # Handle datetime columns
                if 'datetime64[ns]' in str(expected_types):
                    try:
                        pd.to_datetime(df[col])
                    except (ValueError, TypeError) as e:
                        raise DataValidationError(
                            f"{data_type} column '{col}' cannot be converted to datetime: {e}"
                        )
                else:
                    # Check data types more permissively
                    actual_dtype = df[col].dtype
                    
                    # For string types
                    if str in expected_types:
                        if not pd.api.types.is_string_dtype(actual_dtype) and not pd.api.types.is_object_dtype(actual_dtype):
                            # Only raise error if it's definitely not string-like
                            if not (int in expected_types and pd.api.types.is_integer_dtype(actual_dtype)):
                                if not (float in expected_types and pd.api.types.is_numeric_dtype(actual_dtype)):
                                    raise DataValidationError(
                                        f"{data_type} column '{col}' has invalid type {actual_dtype}, "
                                        f"expected one of {expected_types}"
                                    )
                    
                    # For numeric types - be more permissive
                    elif any(t in expected_types for t in (int, float)):
                        if not pd.api.types.is_numeric_dtype(actual_dtype):
                            raise DataValidationError(
                                f"{data_type} column '{col}' has invalid type {actual_dtype}, "
                                f"expected numeric type from {expected_types}"
                            )
        
        logger.debug(f"{data_type} validation passed for {len(df)} rows")


class CacheManager:
    """
    Manages caching of loaded data for performance optimization.
    """
    
    def __init__(self, max_cache_size: int = 10):
        """
        Initialize cache manager.
        
        Args:
            max_cache_size: Maximum number of datasets to cache
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._max_cache_size = max_cache_size
        self._lock = threading.RLock()
        
        logger.debug(f"CacheManager initialized with max size: {max_cache_size}")
    
    def get_cache_key(self, trade_log_path: Path, equity_curve_path: Path) -> str:
        """Generate a unique cache key for the data paths."""
        return f"{trade_log_path.stem}_{equity_curve_path.stem}_{trade_log_path.stat().st_mtime}_{equity_curve_path.stat().st_mtime}"
    
    def get(self, cache_key: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Retrieve cached data if available."""
        with self._lock:
            if cache_key in self._cache:
                self._access_times[cache_key] = datetime.now()
                logger.debug(f"Cache hit for key: {cache_key}")
                return self._cache[cache_key]
            
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
    
    def put(self, cache_key: str, data: Dict[str, pd.DataFrame]) -> None:
        """Store data in cache with LRU eviction."""
        with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_cache_size and cache_key not in self._cache:
                if self._access_times:  # Ensure we have entries to evict
                    oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
                    del self._cache[oldest_key]
                    del self._access_times[oldest_key]
                    logger.debug(f"Evicted cache entry: {oldest_key}")
            
            # Store new data
            self._cache[cache_key] = data.copy()  # Defensive copy
            self._access_times[cache_key] = datetime.now()
            logger.debug(f"Cached data for key: {cache_key}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self._max_cache_size,
                'cached_keys': list(self._cache.keys())
            }


class DataLoader:
    """
    Handles efficient loading of parquet files with comprehensive error handling.
    """
    
    @staticmethod
    def load_parquet_file(file_path: Path, data_type: str) -> pd.DataFrame:
        """
        Load a parquet file with error handling.
        
        Args:
            file_path: Path to the parquet file
            data_type: Type of data for error messages
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: For other loading errors
        """
        if not file_path.exists():
            raise FileNotFoundError(f"{data_type} file not found: {file_path}")
        
        try:
            logger.debug(f"Loading {data_type} from: {file_path}")
            df = pd.read_parquet(file_path)
            logger.info(f"Successfully loaded {data_type}: {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {data_type} from {file_path}: {e}")
            raise Exception(f"Error loading {data_type}: {e}") from e
    
    @staticmethod
    def load_data_files(trade_log_path: Path, equity_curve_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both trade log and equity curve files.
        
        Args:
            trade_log_path: Path to trade log parquet file
            equity_curve_path: Path to equity curve parquet file
            
        Returns:
            Tuple of (trade_log_df, equity_curve_df)
        """
        trade_log_df = DataLoader.load_parquet_file(trade_log_path, "Trade log")
        equity_curve_df = DataLoader.load_parquet_file(equity_curve_path, "Equity curve")
        
        return trade_log_df, equity_curve_df


class MetricsEngine:
    """
    Core analytical engine that serves as the data processing foundation for dashboard analytics.
    
    This engine processes trade and portfolio data files, providing a standardized interface
    for accessing both trade-level and portfolio-level data with built-in caching, validation,
    and thread-safe operations.
    """
    
    def __init__(self, cache_size: int = 10, enable_caching: bool = True):
        """
        Initialize the MetricsEngine.
        
        Args:
            cache_size: Maximum number of datasets to cache
            enable_caching: Whether to enable caching
        """
        self._cache_manager = CacheManager(cache_size) if enable_caching else None
        self._data_validator = DataValidator()
        self._data_loader = DataLoader()
        self._lock = threading.RLock()
        self._loaded_data: Optional[Dict[str, pd.DataFrame]] = None
        self._current_cache_key: Optional[str] = None
        
        logger.info(f"MetricsEngine initialized (caching: {enable_caching}, cache_size: {cache_size})")
    
    def load_data(self, trade_log_path: Union[str, Path], equity_curve_path: Union[str, Path], 
                  validate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load and process parquet files with validation and caching.
        
        Args:
            trade_log_path: Path to trade_log.parquet file
            equity_curve_path: Path to equity_curve.parquet file  
            validate: Whether to validate data schemas
            
        Returns:
            Dictionary containing 'trade_log' and 'equity_curve' DataFrames
            
        Raises:
            FileNotFoundError: If required files don't exist
            DataValidationError: If data validation fails
        """
        trade_log_path = Path(trade_log_path)
        equity_curve_path = Path(equity_curve_path)
        
        with self._lock:
            # Check cache first if enabled
            cache_key = None
            if self._cache_manager:
                try:
                    cache_key = self._cache_manager.get_cache_key(trade_log_path, equity_curve_path)
                    cached_data = self._cache_manager.get(cache_key)
                    if cached_data:
                        self._loaded_data = cached_data
                        self._current_cache_key = cache_key
                        logger.info("Data loaded from cache")
                        return cached_data
                except (OSError, FileNotFoundError):
                    # File doesn't exist, will be handled by data loader
                    pass
            
            # Load data from files
            try:
                trade_log_df, equity_curve_df = self._data_loader.load_data_files(
                    trade_log_path, equity_curve_path
                )
                
                # Validate data if requested
                if validate:
                    self._data_validator.validate_dataframe(
                        trade_log_df, DataValidator.TRADE_LOG_SCHEMA, "Trade log"
                    )
                    self._data_validator.validate_dataframe(
                        equity_curve_df, DataValidator.EQUITY_CURVE_SCHEMA, "Equity curve"
                    )
                
                # Prepare return data
                data = {
                    'trade_log': trade_log_df,
                    'equity_curve': equity_curve_df
                }
                
                # Cache the data if caching is enabled
                if self._cache_manager and cache_key:
                    self._cache_manager.put(cache_key, data)
                
                # Store reference to current data
                self._loaded_data = data
                self._current_cache_key = cache_key
                
                logger.info(f"Data loaded successfully: {len(trade_log_df)} trades, "
                           f"{len(equity_curve_df)} equity points")
                
                return data
                
            except Exception as e:
                logger.error(f"Failed to load data: {e}")
                raise
    
    def get_trade_data(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded trade log data.
        
        Returns:
            Trade log DataFrame or None if no data is loaded
        """
        with self._lock:
            if self._loaded_data:
                return self._loaded_data['trade_log'].copy()
            return None
    
    def get_equity_data(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded equity curve data.
        
        Returns:
            Equity curve DataFrame or None if no data is loaded
        """
        with self._lock:
            if self._loaded_data:
                return self._loaded_data['equity_curve'].copy()
            return None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of currently loaded data.
        
        Returns:
            Dictionary with data summary information
        """
        with self._lock:
            if not self._loaded_data:
                return {'status': 'no_data_loaded'}
            
            trade_log = self._loaded_data['trade_log']
            equity_curve = self._loaded_data['equity_curve']
            
            summary = {
                'status': 'data_loaded',
                'trade_count': len(trade_log),
                'equity_points': len(equity_curve),
                'symbols': trade_log['symbol'].nunique() if 'symbol' in trade_log.columns else 0,
                'date_range': {
                    'start': None,
                    'end': None
                }
            }
            
            # Calculate date range from equity curve
            if not equity_curve.empty and 'date' in equity_curve.columns:
                dates = pd.to_datetime(equity_curve['date'])
                summary['date_range'] = {
                    'start': dates.min().isoformat(),
                    'end': dates.max().isoformat()
                }
            
            return summary
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self._cache_manager:
            self._cache_manager.clear()
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics or None if caching is disabled
        """
        if self._cache_manager:
            return self._cache_manager.get_stats()
        return {'caching': 'disabled'}
    
    def validate_data_files(self, trade_log_path: Union[str, Path], 
                           equity_curve_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate data files without loading them into memory.
        
        Args:
            trade_log_path: Path to trade_log.parquet file
            equity_curve_path: Path to equity_curve.parquet file
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'trade_log': {'valid': False, 'errors': []},
            'equity_curve': {'valid': False, 'errors': []}
        }
        
        # Validate trade log
        try:
            trade_log_df = self._data_loader.load_parquet_file(Path(trade_log_path), "Trade log")
            self._data_validator.validate_dataframe(
                trade_log_df, DataValidator.TRADE_LOG_SCHEMA, "Trade log"
            )
            results['trade_log']['valid'] = True
            results['trade_log']['row_count'] = len(trade_log_df)
        except Exception as e:
            results['trade_log']['errors'].append(str(e))
        
        # Validate equity curve
        try:
            equity_curve_df = self._data_loader.load_parquet_file(Path(equity_curve_path), "Equity curve")
            self._data_validator.validate_dataframe(
                equity_curve_df, DataValidator.EQUITY_CURVE_SCHEMA, "Equity curve"
            )
            results['equity_curve']['valid'] = True
            results['equity_curve']['row_count'] = len(equity_curve_df)
        except Exception as e:
            results['equity_curve']['errors'].append(str(e))
        
        return results
    
    def __repr__(self) -> str:
        """String representation of the MetricsEngine."""
        cache_info = "enabled" if self._cache_manager else "disabled"
        data_status = "loaded" if self._loaded_data else "no data"
        return f"MetricsEngine(cache: {cache_info}, data: {data_status})"