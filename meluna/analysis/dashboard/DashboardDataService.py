# meluna/analysis/DashboardDataService.py

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import lru_cache
import yaml
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from ..metrics.PortfolioMetrics import PortfolioMetrics
from ..metrics.TradeAnalyzer import TradeAnalyzer, TradeMetrics

logger = logging.getLogger(__name__)


@dataclass
class LoadingState:
    """Container for data loading progress and status."""
    is_loading: bool = False
    progress: float = 0.0  # 0.0 to 1.0
    status_message: str = ""
    error_message: str = ""
    last_updated: Optional[datetime] = None


@dataclass
class DataValidationResult:
    """Container for data validation results."""
    is_valid: bool
    missing_files: List[str]
    corrupted_files: List[str]
    warnings: List[str]
    error_details: Dict[str, str]


@dataclass
class StrategyVersionInfo:
    """Information about a specific strategy version."""
    strategy: str
    version: str
    config_path: str
    has_trades: bool
    has_equity_curve: bool
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    total_trades: int
    data_quality: str  # 'complete', 'partial', 'missing'


@dataclass
class CachedAnalyticsData:
    """Container for cached analytics data."""
    strategy: str
    version: str
    portfolio_metrics: Dict[str, Any]
    trade_metrics: Optional[TradeMetrics]
    equity_curve: pd.Series
    trade_log: Optional[pd.DataFrame]
    raw_data: Dict[str, Any]
    cache_timestamp: datetime
    data_hash: str


class DashboardDataService:
    """
    Comprehensive data loading service for dashboard analytics.
    
    Provides robust data loading infrastructure to connect dashboard components
    with backtest results and analytics data from PortfolioMetrics and TradeAnalyzer modules.
    
    Features:
    - Strategy and version discovery from results directory
    - Data loading with caching and performance optimization
    - Error handling and loading states
    - Data validation and integrity checks
    - Date range filtering capabilities
    - Concurrent loading for improved performance
    """
    
    def __init__(self, results_directory: str = "results", 
                 cache_size: int = 10, enable_threading: bool = True):
        """
        Initialize DashboardDataService.
        
        Args:
            results_directory: Path to backtest results directory
            cache_size: Maximum number of cached analytics datasets
            enable_threading: Enable concurrent data loading
        """
        self.results_directory = Path(results_directory)
        self.cache_size = cache_size
        self.enable_threading = enable_threading
        
        # Initialize caches
        self._strategy_cache = {}
        self._analytics_cache = {}
        self._max_cache_age = timedelta(hours=1)  # Cache validity period
        
        # Loading state tracking
        self.loading_states = {}
        
        # Validation rules
        self.required_files = ['config.yml']
        self.data_files = ['equity_curve.parquet', 'equity_curve.csv', 
                          'trade_log.parquet', 'trade_log.xlsx', 'trade_log.csv']
        
        logger.info(f"DashboardDataService initialized with results_directory: {self.results_directory}")
        
        # Validate results directory
        if not self.results_directory.exists():
            logger.warning(f"Results directory does not exist: {self.results_directory}")
    
    def discover_strategies(self) -> Dict[str, List[str]]:
        """
        Discover available strategies and their versions from results directory structure.
        
        Returns:
            Dictionary mapping strategy names to list of available versions
        """
        cache_key = "strategy_discovery"
        
        # Check cache first
        if cache_key in self._strategy_cache:
            cache_data = self._strategy_cache[cache_key]
            if datetime.now() - cache_data['timestamp'] < self._max_cache_age:
                logger.debug("Using cached strategy discovery data")
                return cache_data['strategies']
        
        logger.info("Discovering strategies from results directory...")
        strategies = {}
        
        try:
            if not self.results_directory.exists():
                logger.warning(f"Results directory does not exist: {self.results_directory}")
                return strategies
            
            # Scan directory structure
            for strategy_path in self.results_directory.iterdir():
                if strategy_path.is_dir():
                    strategy_name = strategy_path.name
                    versions = []
                    
                    # Find version directories
                    for version_path in strategy_path.iterdir():
                        if version_path.is_dir() and version_path.name.startswith('v'):
                            version = version_path.name
                            
                            # Basic validation - check if config exists
                            config_path = version_path / 'config.yml'
                            if config_path.exists():
                                versions.append(version)
                    
                    if versions:
                        # Sort versions naturally
                        versions.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
                        strategies[strategy_name] = versions
            
            # Cache results
            self._strategy_cache[cache_key] = {
                'strategies': strategies,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Discovered {len(strategies)} strategies with "
                       f"{sum(len(v) for v in strategies.values())} total versions")
            
        except Exception as e:
            logger.error(f"Error discovering strategies: {e}")
            strategies = {}
        
        return strategies
    
    def get_strategy_version_info(self, strategy: str, version: str) -> Optional[StrategyVersionInfo]:
        """
        Get detailed information about a specific strategy version.
        
        Args:
            strategy: Strategy name
            version: Version identifier
            
        Returns:
            StrategyVersionInfo object or None if not found
        """
        try:
            version_path = self.results_directory / strategy / version
            if not version_path.exists():
                return None
            
            config_path = version_path / 'config.yml'
            if not config_path.exists():
                return None
            
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check for data files
            has_trades = any((version_path / f).exists() for f in 
                           ['trade_log.parquet', 'trade_log.xlsx', 'trade_log.csv'])
            has_equity_curve = any((version_path / f).exists() for f in 
                                 ['equity_curve.parquet', 'equity_curve.csv'])
            
            # Get date range from config
            start_date = None
            end_date = None
            if 'backtest_settings' in config:
                try:
                    start_date = datetime.strptime(
                        config['backtest_settings']['start_date'], '%Y-%m-%d'
                    )
                    end_date = datetime.strptime(
                        config['backtest_settings']['end_date'], '%Y-%m-%d'
                    )
                except (KeyError, ValueError) as e:
                    logger.debug(f"Could not parse dates from config: {e}")
            
            # Count trades if available
            total_trades = 0
            if has_trades:
                try:
                    trade_file = None
                    if (version_path / 'trade_log.parquet').exists():
                        trade_file = version_path / 'trade_log.parquet'
                    elif (version_path / 'trade_log.xlsx').exists():
                        trade_file = version_path / 'trade_log.xlsx'
                    elif (version_path / 'trade_log.csv').exists():
                        trade_file = version_path / 'trade_log.csv'
                    
                    if trade_file:
                        if trade_file.suffix == '.parquet':
                            df = pd.read_parquet(trade_file)
                        elif trade_file.suffix == '.xlsx':
                            df = pd.read_excel(trade_file)
                        elif trade_file.suffix == '.csv':
                            df = pd.read_csv(trade_file)
                        total_trades = len(df)
                except Exception as e:
                    logger.debug(f"Could not count trades: {e}")
            
            # Determine data quality
            if has_trades and has_equity_curve:
                data_quality = 'complete'
            elif has_equity_curve:
                data_quality = 'partial'
            else:
                data_quality = 'missing'
            
            return StrategyVersionInfo(
                strategy=strategy,
                version=version,
                config_path=str(config_path),
                has_trades=has_trades,
                has_equity_curve=has_equity_curve,
                start_date=start_date,
                end_date=end_date,
                total_trades=total_trades,
                data_quality=data_quality
            )
            
        except Exception as e:
            logger.error(f"Error getting strategy version info for {strategy}/{version}: {e}")
            return None
    
    def validate_data_integrity(self, strategy: str, version: str) -> DataValidationResult:
        """
        Validate data integrity for a specific strategy version.
        
        Args:
            strategy: Strategy name
            version: Version identifier
            
        Returns:
            DataValidationResult with validation details
        """
        version_path = self.results_directory / strategy / version
        missing_files = []
        corrupted_files = []
        warnings_list = []
        error_details = {}
        
        try:
            # Check required files
            for required_file in self.required_files:
                file_path = version_path / required_file
                if not file_path.exists():
                    missing_files.append(required_file)
            
            # Check data files (at least one should exist)
            data_file_found = False
            for data_file in self.data_files:
                file_path = version_path / data_file
                if file_path.exists():
                    data_file_found = True
                    # Try to validate file format
                    try:
                        if data_file.endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                            if len(df) > 1:  # Just check if it has data
                                df = df.head(1)  # Take first row for validation
                        elif data_file.endswith('.csv'):
                            df = pd.read_csv(file_path, nrows=1)
                        elif data_file.endswith('.xlsx'):
                            df = pd.read_excel(file_path, nrows=1)
                        
                        if df.empty:
                            warnings_list.append(f"{data_file} exists but is empty")
                            
                    except Exception as e:
                        corrupted_files.append(data_file)
                        error_details[data_file] = str(e)
            
            if not data_file_found:
                warnings_list.append("No valid data files found")
            
            # Validate config file if it exists
            config_path = version_path / 'config.yml'
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check for required config sections
                    required_sections = ['backtest_settings']
                    for section in required_sections:
                        if section not in config:
                            warnings_list.append(f"Missing config section: {section}")
                            
                except yaml.YAMLError as e:
                    corrupted_files.append('config.yml')
                    error_details['config.yml'] = f"YAML parsing error: {e}"
            
            is_valid = (len(missing_files) == 0 and 
                       len(corrupted_files) == 0 and 
                       data_file_found)
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            error_details['validation'] = str(e)
            is_valid = False
        
        return DataValidationResult(
            is_valid=is_valid,
            missing_files=missing_files,
            corrupted_files=corrupted_files,
            warnings=warnings_list,
            error_details=error_details
        )
    
    def _generate_cache_key(self, strategy: str, version: str, 
                           date_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Generate cache key for analytics data."""
        key_parts = [strategy, version]
        if date_range:
            key_parts.extend([date_range[0].isoformat(), date_range[1].isoformat()])
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def _calculate_data_hash(self, data_dict: Dict[str, Any]) -> str:
        """Calculate hash of data for cache validation."""
        # Create a deterministic string representation
        data_str = str(sorted(data_dict.items()))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _update_loading_state(self, key: str, is_loading: bool = False, 
                             progress: float = 0.0, status: str = "", 
                             error: str = ""):
        """Update loading state for a specific operation."""
        self.loading_states[key] = LoadingState(
            is_loading=is_loading,
            progress=progress,
            status_message=status,
            error_message=error,
            last_updated=datetime.now()
        )
    
    def get_loading_state(self, strategy: str, version: str) -> LoadingState:
        """Get current loading state for a strategy version."""
        key = f"{strategy}_{version}"
        return self.loading_states.get(key, LoadingState())
    
    def load_analytics_data(self, strategy: str, version: str, 
                           date_range: Optional[Tuple[datetime, datetime]] = None,
                           force_reload: bool = False) -> Optional[CachedAnalyticsData]:
        """
        Load comprehensive analytics data for a strategy version.
        
        Args:
            strategy: Strategy name
            version: Version identifier
            date_range: Optional date range filter (start_date, end_date)
            force_reload: Force reload ignoring cache
            
        Returns:
            CachedAnalyticsData object or None if loading fails
        """
        loading_key = f"{strategy}_{version}"
        cache_key = self._generate_cache_key(strategy, version, date_range)
        
        try:
            # Update loading state
            self._update_loading_state(loading_key, True, 0.1, 
                                     "Initializing data loading...")
            
            # Check cache first (unless force reload)
            if not force_reload and cache_key in self._analytics_cache:
                cached_data = self._analytics_cache[cache_key]
                cache_age = datetime.now() - cached_data.cache_timestamp
                
                if cache_age < self._max_cache_age:
                    logger.debug(f"Using cached data for {strategy}/{version}")
                    self._update_loading_state(loading_key, False, 1.0, "Data loaded from cache")
                    return cached_data
            
            # Validate data first
            self._update_loading_state(loading_key, True, 0.2, "Validating data integrity...")
            validation_result = self.validate_data_integrity(strategy, version)
            
            if not validation_result.is_valid:
                error_msg = f"Data validation failed: {validation_result.error_details}"
                self._update_loading_state(loading_key, False, 0, "", error_msg)
                logger.error(error_msg)
                return None
            
            # Load raw data
            self._update_loading_state(loading_key, True, 0.4, "Loading raw data files...")
            raw_data = self._load_raw_data(strategy, version, date_range)
            
            if not raw_data:
                error_msg = "Failed to load raw data files"
                self._update_loading_state(loading_key, False, 0, "", error_msg)
                logger.error(error_msg)
                return None
            
            # Calculate portfolio metrics
            self._update_loading_state(loading_key, True, 0.6, "Calculating portfolio metrics...")
            portfolio_metrics = None
            
            if 'equity_curve' in raw_data and raw_data['equity_curve'] is not None:
                try:
                    portfolio_calculator = PortfolioMetrics(raw_data['equity_curve'])
                    portfolio_metrics = portfolio_calculator.calculate_all_metrics()
                except Exception as e:
                    logger.warning(f"Portfolio metrics calculation failed: {e}")
                    portfolio_metrics = {}
            
            # Calculate trade metrics
            self._update_loading_state(loading_key, True, 0.8, "Calculating trade metrics...")
            trade_metrics = None
            
            if 'trade_log' in raw_data and raw_data['trade_log'] is not None:
                try:
                    trade_analyzer = TradeAnalyzer(raw_data['trade_log'])
                    trade_metrics = trade_analyzer.calculate_all_metrics()
                except Exception as e:
                    logger.warning(f"Trade metrics calculation failed: {e}")
                    trade_metrics = None
            
            # Create cached data object
            self._update_loading_state(loading_key, True, 0.95, "Finalizing data...")
            
            cached_data = CachedAnalyticsData(
                strategy=strategy,
                version=version,
                portfolio_metrics=portfolio_metrics or {},
                trade_metrics=trade_metrics,
                equity_curve=raw_data.get('equity_curve'),
                trade_log=raw_data.get('trade_log'),
                raw_data=raw_data,
                cache_timestamp=datetime.now(),
                data_hash=self._calculate_data_hash(raw_data)
            )
            
            # Cache the data (with size limit)
            if len(self._analytics_cache) >= self.cache_size:
                # Remove oldest cached item
                oldest_key = min(self._analytics_cache.keys(), 
                               key=lambda k: self._analytics_cache[k].cache_timestamp)
                del self._analytics_cache[oldest_key]
                logger.debug(f"Removed oldest cache entry: {oldest_key}")
            
            self._analytics_cache[cache_key] = cached_data
            
            self._update_loading_state(loading_key, False, 1.0, 
                                     "Data loading completed successfully")
            
            logger.info(f"Successfully loaded analytics data for {strategy}/{version}")
            return cached_data
            
        except Exception as e:
            error_msg = f"Error loading analytics data: {e}"
            self._update_loading_state(loading_key, False, 0, "", error_msg)
            logger.error(error_msg)
            return None
    
    def _load_raw_data(self, strategy: str, version: str, 
                      date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Load raw data files for a strategy version.
        
        Args:
            strategy: Strategy name
            version: Version identifier
            date_range: Optional date range filter
            
        Returns:
            Dictionary containing loaded data
        """
        version_path = self.results_directory / strategy / version
        data = {}
        
        try:
            # Load configuration
            config_path = version_path / 'config.yml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data['config'] = yaml.safe_load(f)
            
            # Load equity curve
            equity_curve = None
            for file_name in ['equity_curve.parquet', 'equity_curve.csv']:
                file_path = version_path / file_name
                if file_path.exists():
                    try:
                        if file_name.endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                        else:
                            df = pd.read_csv(file_path)
                        
                        # Standardize column names and index - check multiple date column variants
                        date_cols = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp']
                        date_col_found = None
                        for date_col in date_cols:
                            if date_col in df.columns:
                                df[date_col] = pd.to_datetime(df[date_col])
                                df.set_index(date_col, inplace=True)
                                date_col_found = date_col
                                break
                        
                        # If no date column found but index looks like datetime, use it
                        if date_col_found is None and hasattr(df.index, 'dtype'):
                            try:
                                df.index = pd.to_datetime(df.index)
                            except (ValueError, TypeError):
                                logger.debug(f"Could not convert index to datetime for {file_name}")
                        
                        # Find equity column - expanded list of possible names
                        equity_col = None
                        equity_col_options = [
                            'Equity', 'equity', 'EQUITY',
                            'portfolio_value', 'Portfolio_Value', 'PORTFOLIO_VALUE',
                            'value', 'Value', 'VALUE',
                            'balance', 'Balance', 'BALANCE',
                            'total_value', 'Total_Value', 'TOTAL_VALUE',
                            'nav', 'NAV', 'net_asset_value'
                        ]
                        for col in equity_col_options:
                            if col in df.columns:
                                equity_col = col
                                break
                        
                        if equity_col:
                            equity_curve = df[equity_col]
                            logger.debug(f"Successfully loaded equity curve from {file_name} using column '{equity_col}'")
                            
                            # Apply date range filter if specified
                            if date_range and not equity_curve.empty:
                                mask = ((equity_curve.index >= date_range[0]) & 
                                       (equity_curve.index <= date_range[1]))
                                equity_curve = equity_curve[mask]
                                logger.debug(f"Applied date range filter, resulting shape: {equity_curve.shape}")
                            
                            break
                        else:
                            logger.warning(f"No recognized equity column found in {file_name}. Available columns: {list(df.columns)}")
                    except Exception as e:
                        logger.warning(f"Failed to load {file_name}: {e}")
                        continue
            
            data['equity_curve'] = equity_curve
            
            # Load trade log (prioritize parquet > xlsx > csv)
            trade_log = None
            for file_name in ['trade_log.parquet', 'trade_log.xlsx', 'trade_log.csv']:
                file_path = version_path / file_name
                if file_path.exists():
                    try:
                        if file_name.endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                        elif file_name.endswith('.xlsx'):
                            df = pd.read_excel(file_path)
                        elif file_name.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        
                        # Standardize timestamp columns
                        for col in ['entry_timestamp', 'exit_timestamp']:
                            if col in df.columns:
                                df[col] = pd.to_datetime(df[col])
                        
                        # Apply date range filter if specified
                        if date_range and not df.empty:
                            if 'entry_timestamp' in df.columns:
                                mask = ((df['entry_timestamp'] >= date_range[0]) & 
                                       (df['entry_timestamp'] <= date_range[1]))
                                df = df[mask]
                        
                        trade_log = df
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {file_name}: {e}")
                        continue
            
            data['trade_log'] = trade_log
            
            # Load notes if available
            notes_path = version_path / 'notes.txt'
            if notes_path.exists():
                try:
                    with open(notes_path, 'r') as f:
                        data['notes'] = f.read()
                except Exception as e:
                    logger.debug(f"Could not load notes: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading raw data for {strategy}/{version}: {e}")
            return {}
    
    def filter_data_by_date_range(self, cached_data: CachedAnalyticsData, 
                                 start_date: datetime, end_date: datetime) -> CachedAnalyticsData:
        """
        Filter existing cached data by date range.
        
        Args:
            cached_data: Existing cached analytics data
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            New CachedAnalyticsData with filtered data
        """
        try:
            filtered_equity_curve = None
            if cached_data.equity_curve is not None:
                mask = ((cached_data.equity_curve.index >= start_date) & 
                       (cached_data.equity_curve.index <= end_date))
                filtered_equity_curve = cached_data.equity_curve[mask]
            
            filtered_trade_log = None
            if cached_data.trade_log is not None:
                if 'entry_timestamp' in cached_data.trade_log.columns:
                    mask = ((cached_data.trade_log['entry_timestamp'] >= start_date) & 
                           (cached_data.trade_log['entry_timestamp'] <= end_date))
                    filtered_trade_log = cached_data.trade_log[mask]
            
            # Recalculate metrics with filtered data
            portfolio_metrics = {}
            if filtered_equity_curve is not None and not filtered_equity_curve.empty:
                portfolio_calculator = PortfolioMetrics(filtered_equity_curve)
                portfolio_metrics = portfolio_calculator.calculate_all_metrics()
            
            trade_metrics = None
            if filtered_trade_log is not None and not filtered_trade_log.empty:
                trade_analyzer = TradeAnalyzer(filtered_trade_log)
                trade_metrics = trade_analyzer.calculate_all_metrics()
            
            # Create new cached data object
            return CachedAnalyticsData(
                strategy=cached_data.strategy,
                version=cached_data.version,
                portfolio_metrics=portfolio_metrics,
                trade_metrics=trade_metrics,
                equity_curve=filtered_equity_curve,
                trade_log=filtered_trade_log,
                raw_data=cached_data.raw_data,  # Keep original raw data
                cache_timestamp=datetime.now(),
                data_hash=self._calculate_data_hash({
                    'equity_curve': filtered_equity_curve,
                    'trade_log': filtered_trade_log
                })
            )
            
        except Exception as e:
            logger.error(f"Error filtering data by date range: {e}")
            return cached_data
    
    def clear_cache(self, strategy: Optional[str] = None, version: Optional[str] = None):
        """
        Clear analytics cache.
        
        Args:
            strategy: Optional strategy filter (clear only this strategy)
            version: Optional version filter (clear only this version)
        """
        if strategy is None and version is None:
            # Clear entire cache
            self._analytics_cache.clear()
            self._strategy_cache.clear()
            logger.info("Cleared entire analytics cache")
        else:
            # Clear specific entries
            keys_to_remove = []
            for key, data in self._analytics_cache.items():
                if strategy and data.strategy != strategy:
                    continue
                if version and data.version != version:
                    continue
                keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._analytics_cache[key]
            
            logger.info(f"Cleared cache for {len(keys_to_remove)} entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'analytics_cache_size': len(self._analytics_cache),
            'strategy_cache_size': len(self._strategy_cache),
            'max_cache_size': self.cache_size,
            'cache_age_hours': self._max_cache_age.total_seconds() / 3600,
            'active_loading_states': len([k for k, v in self.loading_states.items() 
                                        if v.is_loading]),
            'cached_strategies': list(set(data.strategy for data in self._analytics_cache.values())),
            'memory_usage_mb': sum(
                len(str(data.portfolio_metrics)) + 
                (len(data.equity_curve) if data.equity_curve is not None else 0) +
                (len(data.trade_log) if data.trade_log is not None else 0)
                for data in self._analytics_cache.values()
            ) / (1024 * 1024)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the data service.
        
        Returns:
            Health check results
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'results_directory_exists': self.results_directory.exists(),
            'results_directory_readable': False,
            'discovered_strategies': 0,
            'cache_operational': True,
            'issues': []
        }
        
        try:
            # Check results directory
            if self.results_directory.exists():
                # Test read access
                list(self.results_directory.iterdir())
                health_status['results_directory_readable'] = True
                
                # Test strategy discovery
                strategies = self.discover_strategies()
                health_status['discovered_strategies'] = len(strategies)
                
                if len(strategies) == 0:
                    health_status['issues'].append("No strategies found in results directory")
            else:
                health_status['issues'].append(f"Results directory does not exist: {self.results_directory}")
            
            # Test cache functionality
            try:
                test_key = "health_check_test"
                self._strategy_cache[test_key] = {'test': True, 'timestamp': datetime.now()}
                del self._strategy_cache[test_key]
            except Exception as e:
                health_status['cache_operational'] = False
                health_status['issues'].append(f"Cache not operational: {e}")
            
            # Determine overall status
            if health_status['issues']:
                health_status['status'] = 'degraded' if health_status['results_directory_exists'] else 'unhealthy'
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['issues'].append(f"Health check failed: {e}")
        
        return health_status