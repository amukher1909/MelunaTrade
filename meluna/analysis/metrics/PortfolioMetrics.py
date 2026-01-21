# meluna/analysis/PortfolioMetrics.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class PortfolioMetrics:
    """
    Comprehensive portfolio-level metrics calculation module that provides macro-view 
    analytics on strategy performance and risk profile.
    
    This class calculates a complete suite of performance, risk, and drawdown metrics
    from equity curve data, supporting different time periods and mathematical accuracy
    standards.
    """
    
    def __init__(self, equity_curve: pd.Series, risk_free_rate: float = 0.0, 
                 trading_days_per_year: int = 252):
        """
        Initialize PortfolioMetrics calculator.
        
        Args:
            equity_curve (pd.Series): Series with datetime index representing portfolio value
            risk_free_rate (float): Annual risk-free rate for Sharpe/Sortino calculations
            trading_days_per_year (int): Number of trading days for annualization
            
        Raises:
            ValueError: If equity curve is empty or has insufficient data
        """
        if equity_curve.empty:
            raise ValueError("Equity curve cannot be empty")
        
        if len(equity_curve) < 2:
            raise ValueError("Equity curve must have at least 2 data points")
        
        self.equity_curve = equity_curve.copy()
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        
        # Calculate daily returns, handling potential data issues
        self.returns = self.equity_curve.pct_change().dropna()
        
        if self.returns.empty:
            raise ValueError("Unable to calculate returns from equity curve")
        
        # Store calculated metrics
        self.metrics = {}
        
        logger.info(f"PortfolioMetrics initialized with {len(self.equity_curve)} equity points, "
                   f"{len(self.returns)} return observations")
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio metrics suite.
        
        Returns:
            Dictionary containing all calculated metrics
        """
        logger.info("Calculating comprehensive portfolio metrics...")
        
        # Performance metrics
        self._calculate_performance_metrics()
        
        # Risk metrics
        self._calculate_risk_metrics()
        
        # Tail risk metrics
        self._calculate_tail_risk_metrics()
        
        # Drawdown metrics
        self._calculate_drawdown_metrics()
        
        # Horizon returns
        self._calculate_horizon_returns()
        
        logger.info("Portfolio metrics calculation completed successfully")
        return self.metrics
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate core performance metrics."""
        # Total return
        start_value = self.equity_curve.iloc[0]
        end_value = self.equity_curve.iloc[-1]
        total_return = (end_value / start_value) - 1
        
        self.metrics['Total Return'] = total_return
        self.metrics['Total Return (%)'] = total_return * 100
        
        # CAGR (Compound Annual Growth Rate)
        total_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = total_days / 365.25
        
        if years > 0:
            cagr = (end_value / start_value) ** (1 / years) - 1
        else:
            cagr = 0.0
        
        self.metrics['CAGR'] = cagr
        self.metrics['CAGR (%)'] = cagr * 100
        
        # Net TRI (Total Return Index)
        # TRI shows cumulative performance normalized to 100 at start
        net_tri = (self.equity_curve / start_value) * 100
        self.metrics['Net TRI (Start=100)'] = net_tri.iloc[-1]
        
        # Gain-to-Pain Ratio
        positive_returns = self.returns[self.returns > 0].sum()
        negative_returns = abs(self.returns[self.returns < 0].sum())
        
        if negative_returns > 0:
            gain_to_pain = positive_returns / negative_returns
        else:
            gain_to_pain = np.inf if positive_returns > 0 else 0
        
        self.metrics['Gain-to-Pain Ratio'] = gain_to_pain
    
    def _calculate_risk_metrics(self) -> None:
        """Calculate comprehensive risk metrics."""
        # Annualized volatility
        daily_vol = self.returns.std()
        annual_vol = daily_vol * np.sqrt(self.trading_days_per_year)
        
        self.metrics['Annualized Volatility'] = annual_vol
        self.metrics['Annualized Volatility (%)'] = annual_vol * 100
        
        # Downside deviation (volatility of negative returns only)
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(self.trading_days_per_year)
        else:
            downside_deviation = 0.0
        
        self.metrics['Downside Deviation'] = downside_deviation
        self.metrics['Downside Deviation (%)'] = downside_deviation * 100
        
        # Sharpe Ratio
        # Calculate CAGR if not already done
        if 'CAGR' not in self.metrics:
            start_value = self.equity_curve.iloc[0]
            end_value = self.equity_curve.iloc[-1]
            total_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
            years = total_days / 365.25
            
            if years > 0:
                cagr = (end_value / start_value) ** (1 / years) - 1
            else:
                cagr = 0.0
            self.metrics['CAGR'] = cagr
        
        excess_return = self.metrics['CAGR'] - self.risk_free_rate
        if annual_vol > 0:
            sharpe_ratio = excess_return / annual_vol
        else:
            sharpe_ratio = 0.0
        
        self.metrics['Sharpe Ratio'] = sharpe_ratio
        
        # Sortino Ratio
        if downside_deviation > 0:
            sortino_ratio = excess_return / downside_deviation
        else:
            sortino_ratio = np.inf if excess_return > 0 else 0.0
        
        self.metrics['Sortino Ratio'] = sortino_ratio
    
    def _calculate_tail_risk_metrics(self) -> None:
        """Calculate tail risk measures including VaR, CVaR, skewness, and kurtosis."""
        if len(self.returns) < 30:
            warning_msg = (f"Only {len(self.returns)} return observations available. "
                          "Tail risk metrics may be unreliable with < 30 observations.")
            logger.warning(warning_msg)
            warnings.warn(warning_msg, UserWarning)
        
        # VaR (Value at Risk) at 95% and 99% confidence levels
        var_95 = np.percentile(self.returns, 5) * 100  # Convert to percentage
        var_99 = np.percentile(self.returns, 1) * 100
        
        self.metrics['VaR 95% (%)'] = var_95
        self.metrics['VaR 99% (%)'] = var_99
        
        # CVaR (Conditional Value at Risk / Expected Shortfall)
        # Average of losses beyond VaR threshold
        returns_pct = self.returns * 100
        cvar_95_threshold = np.percentile(returns_pct, 5)
        cvar_99_threshold = np.percentile(returns_pct, 1)
        
        cvar_95_losses = returns_pct[returns_pct <= cvar_95_threshold]
        cvar_99_losses = returns_pct[returns_pct <= cvar_99_threshold]
        
        cvar_95 = cvar_95_losses.mean() if len(cvar_95_losses) > 0 else cvar_95_threshold
        cvar_99 = cvar_99_losses.mean() if len(cvar_99_losses) > 0 else cvar_99_threshold
        
        self.metrics['CVaR 95% (%)'] = cvar_95
        self.metrics['CVaR 99% (%)'] = cvar_99
        
        # Skewness and Kurtosis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skewness = stats.skew(self.returns)
            kurtosis_val = stats.kurtosis(self.returns, fisher=True)  # Excess kurtosis
        
        self.metrics['Skewness'] = skewness
        self.metrics['Kurtosis'] = kurtosis_val
    
    def _calculate_drawdown_metrics(self) -> None:
        """Calculate comprehensive drawdown analysis."""
        # Calculate running maximum (high water mark)
        running_max = self.equity_curve.cummax()
        
        # Calculate drawdown as percentage from peak
        drawdown = (self.equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        self.metrics['Max Drawdown'] = max_drawdown
        self.metrics['Max Drawdown (%)'] = max_drawdown * 100
        
        # Drawdown duration analysis
        # Identify drawdown periods (consecutive periods below high water mark)
        in_drawdown = drawdown < 0
        
        if in_drawdown.any():
            # Find drawdown periods
            drawdown_periods = []
            current_start = None
            
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and current_start is None:
                    current_start = i
                elif not is_dd and current_start is not None:
                    drawdown_periods.append((current_start, i - 1))
                    current_start = None
            
            # Handle case where we end in drawdown
            if current_start is not None:
                drawdown_periods.append((current_start, len(in_drawdown) - 1))
            
            if drawdown_periods:
                # Calculate durations in days
                durations = []
                for start_idx, end_idx in drawdown_periods:
                    start_date = self.equity_curve.index[start_idx]
                    end_date = self.equity_curve.index[end_idx]
                    duration_days = (end_date - start_date).days + 1  # +1 to include both days
                    durations.append(duration_days)
                
                self.metrics['Longest Drawdown Duration (days)'] = max(durations)
                self.metrics['Average Drawdown Duration (days)'] = np.mean(durations)
                self.metrics['Number of Drawdown Periods'] = len(drawdown_periods)
            else:
                self.metrics['Longest Drawdown Duration (days)'] = 0
                self.metrics['Average Drawdown Duration (days)'] = 0
                self.metrics['Number of Drawdown Periods'] = 0
        else:
            # No drawdowns occurred
            self.metrics['Longest Drawdown Duration (days)'] = 0
            self.metrics['Average Drawdown Duration (days)'] = 0
            self.metrics['Number of Drawdown Periods'] = 0
    
    def get_returns_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the returns distribution.
        
        Returns:
            Dictionary with return distribution summary
        """
        return {
            'Mean Daily Return (%)': self.returns.mean() * 100,
            'Median Daily Return (%)': self.returns.median() * 100,
            'Std Daily Return (%)': self.returns.std() * 100,
            'Min Daily Return (%)': self.returns.min() * 100,
            'Max Daily Return (%)': self.returns.max() * 100,
            'Positive Return Days': (self.returns > 0).sum(),
            'Negative Return Days': (self.returns < 0).sum(),
            'Zero Return Days': (self.returns == 0).sum(),
            'Win Rate (%)': (self.returns > 0).mean() * 100,
            'Total Observations': len(self.returns)
        }
    
    def get_rolling_metrics(self, window_days: int = 252) -> Dict[str, pd.Series]:
        """
        Calculate rolling window metrics for time-series analysis.
        
        Args:
            window_days: Rolling window size in days
            
        Returns:
            Dictionary containing rolling metrics as pandas Series
        """
        if len(self.returns) < window_days:
            logger.warning(f"Insufficient data for {window_days}-day rolling window. "
                          f"Available: {len(self.returns)} observations")
            return {}
        
        rolling_metrics = {}
        
        # Rolling volatility (annualized)
        rolling_vol = self.returns.rolling(window=window_days).std() * np.sqrt(self.trading_days_per_year)
        rolling_metrics['Rolling Volatility'] = rolling_vol
        
        # Rolling Sharpe ratio
        rolling_return = self.returns.rolling(window=window_days).mean() * self.trading_days_per_year
        rolling_sharpe = (rolling_return - self.risk_free_rate) / rolling_vol
        rolling_metrics['Rolling Sharpe Ratio'] = rolling_sharpe
        
        # Rolling maximum drawdown
        rolling_equity = self.equity_curve.rolling(window=window_days)
        rolling_max = rolling_equity.max()
        rolling_current = self.equity_curve
        rolling_dd = (rolling_current - rolling_max) / rolling_max
        rolling_metrics['Rolling Max Drawdown'] = rolling_dd
        
        return rolling_metrics
    
    def validate_minimum_data_requirements(self) -> Dict[str, bool]:
        """
        Validate data sufficiency for reliable metric calculations.
        
        Returns:
            Dictionary indicating which metrics have sufficient data
        """
        observations = len(self.returns)
        
        return {
            'basic_metrics': observations >= 1,
            'volatility_metrics': observations >= 30,
            'tail_risk_metrics': observations >= 100,
            'rolling_analysis': observations >= 252,
            'statistical_significance': observations >= 500
        }
    
    def display_metrics(self, include_rolling: bool = False, rolling_window: int = 252) -> None:
        """
        Display formatted metrics summary.
        
        Args:
            include_rolling: Whether to display rolling metrics
            rolling_window: Window size for rolling metrics
        """
        print("\n" + "=" * 60)
        print("PORTFOLIO PERFORMANCE METRICS")
        print("=" * 60)
        
        # Performance section
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 30)
        perf_metrics = ['Total Return (%)', 'CAGR (%)', 'Net TRI (Start=100)', 'Gain-to-Pain Ratio']
        for metric in perf_metrics:
            if metric in self.metrics:
                value = self.metrics[metric]
                if isinstance(value, (int, float)) and not np.isinf(value):
                    print(f"{metric:<25} {value:>12.2f}")
                else:
                    print(f"{metric:<25} {str(value):>12}")
        
        # Risk section
        print("\n⚠️  RISK METRICS")
        print("-" * 30)
        risk_metrics = ['Annualized Volatility (%)', 'Downside Deviation (%)', 'Sharpe Ratio', 'Sortino Ratio']
        for metric in risk_metrics:
            if metric in self.metrics:
                value = self.metrics[metric]
                if isinstance(value, (int, float)) and not np.isinf(value):
                    print(f"{metric:<25} {value:>12.2f}")
                else:
                    print(f"{metric:<25} {str(value):>12}")
        
        # Tail risk section
        print("\n📉 TAIL RISK METRICS")
        print("-" * 30)
        tail_metrics = ['VaR 95% (%)', 'VaR 99% (%)', 'CVaR 95% (%)', 'CVaR 99% (%)', 'Skewness', 'Kurtosis']
        for metric in tail_metrics:
            if metric in self.metrics:
                value = self.metrics[metric]
                if isinstance(value, (int, float)) and not np.isinf(value):
                    print(f"{metric:<25} {value:>12.2f}")
                else:
                    print(f"{metric:<25} {str(value):>12}")
        
        # Drawdown section
        print("\n📊 DRAWDOWN ANALYSIS")
        print("-" * 30)
        dd_metrics = ['Max Drawdown (%)', 'Longest Drawdown Duration (days)', 
                     'Average Drawdown Duration (days)', 'Number of Drawdown Periods']
        for metric in dd_metrics:
            if metric in self.metrics:
                value = self.metrics[metric]
                if isinstance(value, (int, float)) and not np.isinf(value):
                    if 'Duration' in metric or 'Number' in metric:
                        print(f"{metric:<35} {value:>7.0f}")
                    else:
                        print(f"{metric:<35} {value:>7.2f}")
                else:
                    print(f"{metric:<35} {str(value):>7}")
        
        print("\n" + "=" * 60)
        
        # Data validation summary
        validation = self.validate_minimum_data_requirements()
        print("\n📋 DATA SUFFICIENCY")
        print("-" * 30)
        for requirement, sufficient in validation.items():
            status = "✓" if sufficient else "✗"
            print(f"{requirement.replace('_', ' ').title():<25} {status:>5}")
        
        if include_rolling and validation['rolling_analysis']:
            print(f"\n📈 ROLLING METRICS ({rolling_window} days)")
            print("-" * 40)
            rolling_data = self.get_rolling_metrics(rolling_window)
            for metric_name, series in rolling_data.items():
                if not series.dropna().empty:
                    latest_value = series.dropna().iloc[-1]
                    print(f"{metric_name:<25} {latest_value:>12.2f} (latest)")
    
    def _calculate_horizon_returns(self) -> None:
        """
        Calculate returns over various time horizons (periods).
        
        Calculates period returns for common investment horizons:
        1 month, 3 months, 6 months, 1 year, 2 years, 3 years
        """
        # Define horizon periods in trading days (assuming 252 trading days per year)
        horizon_periods = {
            '1M': 21,    # ~1 month (21 trading days)
            '3M': 63,    # ~3 months (63 trading days) 
            '6M': 126,   # ~6 months (126 trading days)
            '1Y': 252,   # 1 year (252 trading days)
            '2Y': 504,   # 2 years (504 trading days)
            '3Y': 756    # 3 years (756 trading days)
        }
        
        total_observations = len(self.equity_curve)
        
        for period_name, days_back in horizon_periods.items():
            # Check if we have enough data for this horizon
            if total_observations > days_back:
                # Get values for horizon calculation
                current_value = self.equity_curve.iloc[-1]
                horizon_value = self.equity_curve.iloc[-days_back-1]  # -1 to include the day
                
                # Calculate horizon return
                horizon_return = (current_value / horizon_value - 1) * 100
                
                # Store the metric
                self.metrics[f'{period_name} Return (%)'] = horizon_return
                
                # Calculate annualized return for this horizon
                years = days_back / 252.0
                if years > 0 and horizon_value > 0:
                    annualized_return = ((current_value / horizon_value) ** (1/years) - 1) * 100
                    self.metrics[f'{period_name} Annualized Return (%)'] = annualized_return
            else:
                # Not enough data for this horizon
                self.metrics[f'{period_name} Return (%)'] = None
                self.metrics[f'{period_name} Annualized Return (%)'] = None
        
        # Calculate horizon volatilities if we have enough data
        for period_name, days_back in horizon_periods.items():
            if total_observations > days_back:
                # Get returns for the horizon period
                horizon_returns = self.returns.iloc[-days_back:]
                
                if len(horizon_returns) > 1:
                    # Calculate annualized volatility for this horizon
                    horizon_vol = horizon_returns.std() * np.sqrt(252) * 100
                    self.metrics[f'{period_name} Volatility (%)'] = horizon_vol
                    
                    # Calculate Sharpe ratio for this horizon
                    horizon_mean_return = horizon_returns.mean() * 252 * 100  # Annualized
                    if horizon_vol > 0:
                        horizon_sharpe = (horizon_mean_return - self.risk_free_rate * 100) / horizon_vol
                        self.metrics[f'{period_name} Sharpe Ratio'] = horizon_sharpe
                    else:
                        self.metrics[f'{period_name} Sharpe Ratio'] = None
                else:
                    self.metrics[f'{period_name} Volatility (%)'] = None
                    self.metrics[f'{period_name} Sharpe Ratio'] = None
            else:
                self.metrics[f'{period_name} Volatility (%)'] = None
                self.metrics[f'{period_name} Sharpe Ratio'] = None