# meluna/analysis/TradeAnalyzer.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TradeMetrics:
    """Container for comprehensive trade-level metrics."""
    # Core profitability metrics
    profit_factor: float
    expectancy: float
    payoff_ratio: float
    
    # MFE/MAE metrics
    avg_mfe: float
    avg_mae: float
    mfe_efficiency: float
    mae_efficiency: float
    
    # Duration analysis
    avg_holding_period: float
    winner_avg_duration: float
    loser_avg_duration: float
    
    # Alpha decay metrics
    alpha_decay_coefficient: float
    optimal_holding_period: Optional[float]
    
    # Statistical significance
    profit_factor_pvalue: float
    expectancy_pvalue: float


class TradeAnalyzer:
    """
    Comprehensive trade-level metrics analysis module that dissects individual trades
    to understand alpha generation quality and trading logic effectiveness.
    
    This class provides micro-level analysis including profitability metrics,
    MFE/MAE tracking, duration analysis, and alpha decay detection.
    
    Performance Optimized for large datasets (>100k trades) with chunked processing.
    """
    
    def __init__(self, trade_log: pd.DataFrame, price_data: Optional[pd.DataFrame] = None, 
                 min_trades: int = 30, chunk_size: int = 10000):
        """
        Initialize the TradeAnalyzer.
        
        Args:
            trade_log (pd.DataFrame): DataFrame containing trade records with columns:
                - trade_id, symbol, entry_timestamp, exit_timestamp, 
                  entry_price, exit_price, quantity, pnl
            price_data (pd.DataFrame, optional): Intraday price data for MFE/MAE calculation
            min_trades (int): Minimum number of trades for statistical significance
            chunk_size (int): Size of chunks for processing large datasets (default 10000)
        """
        if trade_log.empty:
            raise ValueError("Trade log cannot be empty")
            
        self.trade_log = trade_log.copy()
        self.price_data = price_data
        self.min_trades = min_trades
        self.chunk_size = chunk_size
        
        # Validate required columns
        required_cols = ['symbol', 'entry_timestamp', 'exit_timestamp', 
                        'entry_price', 'exit_price', 'quantity', 'pnl']
        missing_cols = [col for col in required_cols if col not in trade_log.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process timestamps and calculate durations
        self._process_timestamps()
        
        # Initialize metrics storage
        self.metrics = {}
        self.detailed_analysis = {}
        
        logger.info(f"TradeAnalyzer initialized with {len(self.trade_log)} trades")
    
    def _process_timestamps(self):
        """Process timestamps and calculate trade durations."""
        self.trade_log['entry_timestamp'] = pd.to_datetime(self.trade_log['entry_timestamp'])
        self.trade_log['exit_timestamp'] = pd.to_datetime(self.trade_log['exit_timestamp'])
        
        # Calculate duration in days
        self.trade_log['duration_days'] = (
            self.trade_log['exit_timestamp'] - self.trade_log['entry_timestamp']
        ).dt.total_seconds() / (24 * 3600)
        
        # Ensure minimum duration of 0.01 days (about 15 minutes)
        self.trade_log['duration_days'] = self.trade_log['duration_days'].clip(lower=0.01)
    
    def calculate_core_profitability_metrics(self) -> Dict[str, float]:
        """
        Calculate core profitability metrics: Profit Factor, Expectancy, Payoff Ratio.
        
        Returns:
            Dict containing profit factor, expectancy, and payoff ratio
        """
        wins = self.trade_log[self.trade_log['pnl'] > 0]
        losses = self.trade_log[self.trade_log['pnl'] < 0]
        
        # Profit Factor: Gross profits / Gross losses
        gross_profits = wins['pnl'].sum()
        gross_losses = abs(losses['pnl'].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        # Expectancy: Average expected P/L per trade
        expectancy = self.trade_log['pnl'].mean()
        
        # Payoff Ratio: Average win / Average loss
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        metrics = {
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'payoff_ratio': payoff_ratio,
            'gross_profits': gross_profits,
            'gross_losses': gross_losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_count': len(wins),
            'loss_count': len(losses),
            'total_trades': len(self.trade_log)
        }
        
        self.metrics.update(metrics)
        logger.info("Core profitability metrics calculated")
        return metrics
    
    def calculate_mfe_mae_metrics(self) -> Dict[str, float]:
        """
        Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).
        
        If price_data is not available, estimates MFE/MAE based on trade outcomes.
        
        Returns:
            Dict containing MFE/MAE metrics
        """
        if self.price_data is not None:
            return self._calculate_mfe_mae_from_price_data()
        else:
            return self._estimate_mfe_mae_from_trades()
    
    def _calculate_mfe_mae_from_price_data(self) -> Dict[str, float]:
        """Calculate actual MFE/MAE from intraday price data with chunked processing."""
        if len(self.trade_log) > self.chunk_size:
            return self._calculate_mfe_mae_chunked()
        
        mfe_list = []
        mae_list = []
        
        for _, trade in self.trade_log.iterrows():
            symbol = trade['symbol']
            entry_time = trade['entry_timestamp']
            exit_time = trade['exit_timestamp']
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            
            # Get price data for this trade period
            symbol_data = self.price_data[
                (self.price_data['symbol'] == symbol) &
                (self.price_data['timestamp'] >= entry_time) &
                (self.price_data['timestamp'] <= exit_time)
            ]
            
            if symbol_data.empty:
                # Use trade prices as fallback
                mfe_list.append(max(0, trade['pnl']))
                mae_list.append(min(0, trade['pnl']))
                continue
            
            # Calculate P/L for each price point
            position_pnl = (symbol_data['close'] - entry_price) * quantity
            
            # MFE: Maximum favorable excursion
            mfe = position_pnl.max()
            mae = position_pnl.min()
            
            mfe_list.append(mfe)
            mae_list.append(mae)
        
        self.trade_log['mfe'] = mfe_list
        self.trade_log['mae'] = mae_list
        
        return self._calculate_mfe_mae_summary()
    
    def _estimate_mfe_mae_from_trades(self) -> Dict[str, float]:
        """Estimate MFE/MAE from final trade outcomes."""
        logger.warning("Price data not available. Estimating MFE/MAE from trade outcomes.")
        
        # For winners: MFE = pnl, MAE = estimated adverse excursion
        # For losers: MAE = pnl, MFE = estimated favorable excursion
        
        mfe_list = []
        mae_list = []
        
        for _, trade in self.trade_log.iterrows():
            pnl = trade['pnl']
            
            if pnl > 0:  # Winning trade
                # MFE is at least the final profit
                mfe = pnl
                # Estimate MAE based on volatility (assume 20% adverse excursion)
                mae = min(0, -pnl * 0.2)
            else:  # Losing trade
                # MAE is the final loss
                mae = pnl
                # Estimate MFE (assume trade was favorable at some point)
                mfe = max(0, abs(pnl) * 0.3)
            
            mfe_list.append(mfe)
            mae_list.append(mae)
        
        self.trade_log['mfe'] = mfe_list
        self.trade_log['mae'] = mae_list
        
        return self._calculate_mfe_mae_summary()
    
    def _calculate_mfe_mae_chunked(self) -> Dict[str, float]:
        """Process MFE/MAE calculation in chunks for large datasets."""
        logger.info(f"Processing {len(self.trade_log)} trades in chunks of {self.chunk_size}")
        
        all_mfe = []
        all_mae = []
        
        for i in range(0, len(self.trade_log), self.chunk_size):
            chunk = self.trade_log.iloc[i:i + self.chunk_size]
            logger.debug(f"Processing chunk {i//self.chunk_size + 1}/{(len(self.trade_log) + self.chunk_size - 1)//self.chunk_size}")
            
            chunk_mfe = []
            chunk_mae = []
            
            for _, trade in chunk.iterrows():
                symbol = trade['symbol']
                entry_time = trade['entry_timestamp']
                exit_time = trade['exit_timestamp']
                entry_price = trade['entry_price']
                quantity = trade['quantity']
                
                # Get price data for this trade period
                if self.price_data is not None:
                    symbol_data = self.price_data[
                        (self.price_data['symbol'] == symbol) &
                        (self.price_data['timestamp'] >= entry_time) &
                        (self.price_data['timestamp'] <= exit_time)
                    ]
                    
                    if not symbol_data.empty:
                        position_pnl = (symbol_data['close'] - entry_price) * quantity
                        mfe = position_pnl.max()
                        mae = position_pnl.min()
                    else:
                        mfe = max(0, trade['pnl'])
                        mae = min(0, trade['pnl'])
                else:
                    # Fallback estimation
                    mfe = max(0, trade['pnl']) if trade['pnl'] > 0 else abs(trade['pnl']) * 0.3
                    mae = min(0, trade['pnl']) if trade['pnl'] < 0 else -trade['pnl'] * 0.2
                
                chunk_mfe.append(mfe)
                chunk_mae.append(mae)
            
            all_mfe.extend(chunk_mfe)
            all_mae.extend(chunk_mae)
        
        # Store results efficiently
        self.trade_log['mfe'] = all_mfe
        self.trade_log['mae'] = all_mae
        
        return self._calculate_mfe_mae_summary()
    
    def _calculate_mfe_mae_summary(self) -> Dict[str, float]:
        """Calculate summary statistics for MFE/MAE."""
        avg_mfe = self.trade_log['mfe'].mean()
        avg_mae = self.trade_log['mae'].mean()
        
        # MFE Efficiency: How much of the favorable excursion was captured
        winning_trades = self.trade_log[self.trade_log['pnl'] > 0]
        if len(winning_trades) > 0:
            mfe_efficiency = (winning_trades['pnl'] / winning_trades['mfe']).mean()
        else:
            mfe_efficiency = 0.0
        
        # MAE Efficiency: How well adverse excursions were minimized
        losing_trades = self.trade_log[self.trade_log['pnl'] < 0]
        if len(losing_trades) > 0:
            mae_efficiency = (losing_trades['pnl'] / losing_trades['mae']).mean()
        else:
            mae_efficiency = 1.0
        
        metrics = {
            'avg_mfe': avg_mfe,
            'avg_mae': avg_mae,
            'mfe_efficiency': mfe_efficiency,
            'mae_efficiency': mae_efficiency,
            'mfe_std': self.trade_log['mfe'].std(),
            'mae_std': self.trade_log['mae'].std()
        }
        
        self.metrics.update(metrics)
        logger.info("MFE/MAE metrics calculated")
        return metrics
    
    def analyze_duration_patterns(self) -> Dict[str, Union[float, Dict]]:
        """
        Analyze profitability patterns across different holding periods.
        
        Returns:
            Dict containing duration analysis results
        """
        # Overall duration statistics
        avg_duration = self.trade_log['duration_days'].mean()
        median_duration = self.trade_log['duration_days'].median()
        
        # Duration comparison: winners vs losers
        winners = self.trade_log[self.trade_log['pnl'] > 0]
        losers = self.trade_log[self.trade_log['pnl'] < 0]
        
        winner_avg_duration = winners['duration_days'].mean() if len(winners) > 0 else 0
        loser_avg_duration = losers['duration_days'].mean() if len(losers) > 0 else 0
        
        # Duration binning analysis
        duration_bins = [0, 1, 3, 5, 14, 21, 30, 60, float('inf')]
        duration_labels = ['<1d', '1-3d', '3-5d', '5-14d', '2-3w', '3w-1m', '1-2m', '2m+']
        
        self.trade_log['duration_bin'] = pd.cut(
            self.trade_log['duration_days'], 
            bins=duration_bins, 
            labels=duration_labels, 
            right=False
        )
        
        # Analyze profitability by duration bin
        duration_analysis = self.trade_log.groupby('duration_bin', observed=True).agg({
            'pnl': ['count', 'sum', 'mean'],
            'duration_days': 'mean'
        }).round(4)
        
        duration_analysis.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'avg_duration']
        duration_analysis['win_rate'] = (
            self.trade_log.groupby('duration_bin', observed=True)['pnl'].apply(lambda x: (x > 0).mean())
        ).round(4)
        
        metrics = {
            'avg_holding_period': avg_duration,
            'median_holding_period': median_duration,
            'winner_avg_duration': winner_avg_duration,
            'loser_avg_duration': loser_avg_duration,
            'duration_by_bin': duration_analysis.to_dict('index')
        }
        
        self.metrics.update(metrics)
        self.detailed_analysis['duration_analysis'] = duration_analysis
        logger.info("Duration pattern analysis completed")
        return metrics
    
    def detect_alpha_decay(self) -> Dict[str, float]:
        """
        Detect alpha decay patterns across different time horizons.
        
        Returns:
            Dict containing alpha decay metrics
        """
        if len(self.trade_log) < self.min_trades:
            logger.warning(f"Insufficient trades ({len(self.trade_log)}) for alpha decay analysis")
            return {'alpha_decay_coefficient': np.nan, 'optimal_holding_period': None}
        
        # Calculate returns per day
        self.trade_log['daily_return'] = self.trade_log['pnl'] / self.trade_log['duration_days']
        
        # Fit polynomial to understand return decay over time
        try:
            duration_sorted = self.trade_log.sort_values('duration_days')
            x = duration_sorted['duration_days'].values
            y = duration_sorted['daily_return'].values
            
            # Remove outliers for better fit
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
            x_clean, y_clean = x[mask], y[mask]
            
            if len(x_clean) < 10:
                raise ValueError("Too few data points after outlier removal")
            
            # Fit second-degree polynomial
            coefficients = np.polyfit(x_clean, y_clean, deg=2)
            alpha_decay_coefficient = coefficients[0]  # x^2 coefficient
            
            # Find optimal holding period (where derivative = 0)
            # For ax^2 + bx + c, derivative is 2ax + b = 0, so x = -b/(2a)
            if abs(coefficients[0]) > 1e-10:  # Avoid division by very small numbers
                optimal_period = -coefficients[1] / (2 * coefficients[0])
                optimal_period = max(0, optimal_period)  # Ensure non-negative
            else:
                optimal_period = None
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Alpha decay analysis failed: {e}")
            alpha_decay_coefficient = np.nan
            optimal_period = None
        
        metrics = {
            'alpha_decay_coefficient': alpha_decay_coefficient,
            'optimal_holding_period': optimal_period
        }
        
        self.metrics.update(metrics)
        logger.info("Alpha decay analysis completed")
        return metrics
    
    def categorize_trades(self) -> pd.DataFrame:
        """
        Categorize trades by size, duration, and outcome.
        
        Returns:
            DataFrame with trade categories
        """
        categorized = self.trade_log.copy()
        
        # Size categorization
        pnl_abs = abs(categorized['pnl'])
        pnl_quantiles = pnl_abs.quantile([0.33, 0.67])
        
        categorized['size_category'] = pd.cut(
            pnl_abs, 
            bins=[0, pnl_quantiles.iloc[0], pnl_quantiles.iloc[1], float('inf')],
            labels=['Small', 'Medium', 'Large'],
            right=False
        )
        
        # Outcome categorization
        categorized['outcome'] = categorized['pnl'].apply(
            lambda x: 'Winner' if x > 0 else ('Breakeven' if x == 0 else 'Loser')
        )
        
        # Duration category (already done in analyze_duration_patterns)
        if 'duration_bin' not in categorized.columns:
            duration_bins = [0, 1, 3, 5, 14, 21, 30, 60, float('inf')]
            duration_labels = ['<1d', '1-3d', '3-5d', '5-14d', '2-3w', '3w-1m', '1-2m', '2m+']
            categorized['duration_bin'] = pd.cut(
                categorized['duration_days'], 
                bins=duration_bins, 
                labels=duration_labels, 
                right=False
            )
        
        self.detailed_analysis['categorized_trades'] = categorized
        logger.info("Trade categorization completed")
        return categorized
    
    def test_statistical_significance(self) -> Dict[str, float]:
        """
        Test statistical significance of key metrics.
        
        Returns:
            Dict containing p-values for various tests
        """
        if len(self.trade_log) < self.min_trades:
            logger.warning("Insufficient trades for statistical significance testing")
            return {'profit_factor_pvalue': np.nan, 'expectancy_pvalue': np.nan}
        
        # Test if expectancy is significantly different from zero
        _, expectancy_pvalue = stats.ttest_1samp(self.trade_log['pnl'], 0)
        
        # Test if winners and losers have significantly different characteristics
        winners = self.trade_log[self.trade_log['pnl'] > 0]['pnl']
        losers = self.trade_log[self.trade_log['pnl'] < 0]['pnl']
        
        if len(winners) > 5 and len(losers) > 5:
            # Bootstrap test for profit factor significance
            n_bootstrap = 1000
            bootstrap_pf = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                sample_wins = np.random.choice(winners, size=len(winners), replace=True)
                sample_losses = np.random.choice(losers, size=len(losers), replace=True)
                
                sample_pf = sample_wins.sum() / abs(sample_losses.sum())
                bootstrap_pf.append(sample_pf)
            
            # Test if profit factor is significantly > 1
            profit_factor_pvalue = (np.array(bootstrap_pf) <= 1).mean()
        else:
            profit_factor_pvalue = np.nan
        
        significance_results = {
            'profit_factor_pvalue': profit_factor_pvalue,
            'expectancy_pvalue': expectancy_pvalue,
            'sample_size': len(self.trade_log),
            'is_statistically_significant': (
                len(self.trade_log) >= self.min_trades and 
                expectancy_pvalue < 0.05
            )
        }
        
        self.metrics.update(significance_results)
        logger.info("Statistical significance testing completed")
        return significance_results
    
    def calculate_all_metrics(self) -> TradeMetrics:
        """
        Calculate all trade-level metrics and return a comprehensive summary.
        
        Returns:
            TradeMetrics dataclass with all calculated metrics
        """
        logger.info("Starting comprehensive trade analysis...")
        
        # Calculate all metric categories
        profitability = self.calculate_core_profitability_metrics()
        mfe_mae = self.calculate_mfe_mae_metrics()
        duration = self.analyze_duration_patterns()
        alpha_decay = self.detect_alpha_decay()
        significance = self.test_statistical_significance()
        
        # Categorize trades
        self.categorize_trades()
        
        # Create comprehensive metrics object
        trade_metrics = TradeMetrics(
            profit_factor=profitability['profit_factor'],
            expectancy=profitability['expectancy'],
            payoff_ratio=profitability['payoff_ratio'],
            avg_mfe=mfe_mae['avg_mfe'],
            avg_mae=mfe_mae['avg_mae'],
            mfe_efficiency=mfe_mae['mfe_efficiency'],
            mae_efficiency=mfe_mae['mae_efficiency'],
            avg_holding_period=duration['avg_holding_period'],
            winner_avg_duration=duration['winner_avg_duration'],
            loser_avg_duration=duration['loser_avg_duration'],
            alpha_decay_coefficient=alpha_decay['alpha_decay_coefficient'],
            optimal_holding_period=alpha_decay['optimal_holding_period'],
            profit_factor_pvalue=significance['profit_factor_pvalue'],
            expectancy_pvalue=significance['expectancy_pvalue']
        )
        
        logger.info("Comprehensive trade analysis completed")
        return trade_metrics
    
    def get_performance_summary(self) -> Dict[str, Union[int, float, str]]:
        """
        Get a formatted summary of performance metrics.
        
        Returns:
            Dict with formatted performance summary
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        summary = {
            'Total Trades': self.metrics.get('total_trades', 0),
            'Win Rate (%)': (self.metrics.get('win_count', 0) / 
                           max(1, self.metrics.get('total_trades', 1))) * 100,
            'Profit Factor': self.metrics.get('profit_factor', 0),
            'Expectancy (₹)': self.metrics.get('expectancy', 0),
            'Payoff Ratio': self.metrics.get('payoff_ratio', 0),
            'Avg MFE (₹)': self.metrics.get('avg_mfe', 0),
            'Avg MAE (₹)': self.metrics.get('avg_mae', 0),
            'MFE Efficiency (%)': self.metrics.get('mfe_efficiency', 0) * 100,
            'MAE Efficiency (%)': self.metrics.get('mae_efficiency', 0) * 100,
            'Avg Holding Period (days)': self.metrics.get('avg_holding_period', 0),
            'Alpha Decay Coefficient': self.metrics.get('alpha_decay_coefficient', np.nan),
            'Optimal Holding Period (days)': self.metrics.get('optimal_holding_period', None),
            'Statistical Significance': 'Yes' if self.metrics.get('is_statistically_significant', False) else 'No'
        }
        
        return summary
    
    def display_analysis(self):
        """Display a comprehensive analysis report."""
        summary = self.get_performance_summary()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TRADE-LEVEL ANALYSIS")
        print("=" * 80)
        
        print(f"\n📊 CORE METRICS")
        print(f"{'Total Trades:':<30} {summary['Total Trades']:>10}")
        print(f"{'Win Rate:':<30} {summary['Win Rate (%)']:>9.1f}%")
        print(f"{'Profit Factor:':<30} {summary['Profit Factor']:>10.2f}")
        print(f"{'Expectancy:':<30} ₹{summary['Expectancy (₹)']:>9.2f}")
        print(f"{'Payoff Ratio:':<30} {summary['Payoff Ratio']:>10.2f}")
        
        print(f"\n🎯 MFE/MAE ANALYSIS")
        print(f"{'Avg MFE:':<30} ₹{summary['Avg MFE (₹)']:>9.2f}")
        print(f"{'Avg MAE:':<30} ₹{summary['Avg MAE (₹)']:>9.2f}")
        print(f"{'MFE Efficiency:':<30} {summary['MFE Efficiency (%)']:>9.1f}%")
        print(f"{'MAE Efficiency:':<30} {summary['MAE Efficiency (%)']:>9.1f}%")
        
        print(f"\n⏱️ DURATION ANALYSIS")
        print(f"{'Avg Holding Period:':<30} {summary['Avg Holding Period (days)']:>9.1f}d")
        if not np.isnan(summary['Alpha Decay Coefficient']):
            print(f"{'Alpha Decay Coeff:':<30} {summary['Alpha Decay Coefficient']:>10.4f}")
        if summary['Optimal Holding Period (days)'] is not None:
            print(f"{'Optimal Hold Period:':<30} {summary['Optimal Holding Period (days)']:>9.1f}d")
        
        print(f"\n📈 STATISTICAL SIGNIFICANCE")
        print(f"{'Statistically Significant:':<30} {summary['Statistical Significance']:>10}")
        
        # Display duration bin analysis if available
        if 'duration_by_bin' in self.metrics:
            print(f"\n📅 PROFITABILITY BY HOLDING PERIOD")
            print("-" * 60)
            print(f"{'Period':<10} {'Trades':<8} {'Avg P/L':<12} {'Win Rate':<10}")
            print("-" * 60)
            
            for period, data in self.metrics['duration_by_bin'].items():
                if pd.notna(data['avg_pnl']):
                    print(f"{period:<10} {int(data['trade_count']):<8} "
                          f"₹{data['avg_pnl']:<11.2f} {data['win_rate']:<9.1%}")
        
        print("\n" + "=" * 80 + "\n")