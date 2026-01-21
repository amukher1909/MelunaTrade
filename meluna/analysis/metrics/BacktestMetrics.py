# meluna/analysis/BacktestMetrics.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BacktestMetrics:
    """Backtest performance metrics calculator."""
    
    def __init__(self, equity_series: pd.Series, trade_log: Optional[pd.DataFrame] = None):
        """Initialize with equity curve and trade log."""
        self.equity_series = equity_series.copy()
        self.trade_log = trade_log.copy() if trade_log is not None else None
        self.returns = self.equity_series.pct_change().dropna()
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Validate input data."""
        if self.equity_series.empty:
            raise ValueError("Equity series cannot be empty")
        
        if self.equity_series.isna().any():
            logger.warning("Equity series contains NaN values, forward filling")
            self.equity_series = self.equity_series.fillna(method='ffill')
        
        if self.trade_log is not None and self.trade_log.empty:
            logger.warning("Trade log is empty")
            self.trade_log = None
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all backtest metrics."""
        metrics = {}
        
        # Basic performance metrics
        metrics.update(self._calculate_return_metrics())
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics())
        
        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics())
        
        # Trade-based metrics (if trade log available)
        if self.trade_log is not None:
            metrics.update(self._calculate_trade_metrics())
        
        return metrics
    
    def _calculate_return_metrics(self) -> Dict[str, float]:
        """Calculate return-based performance metrics."""
        if len(self.returns) == 0:
            return {}
        
        # Total return
        total_return = (self.equity_series.iloc[-1] / self.equity_series.iloc[0] - 1) * 100
        
        # Annualized return (CAGR)
        days = (self.equity_series.index[-1] - self.equity_series.index[0]).days
        years = days / 365.25
        cagr = ((self.equity_series.iloc[-1] / self.equity_series.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = self.returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (self.returns.mean() * 252) / (self.returns.std() * np.sqrt(252)) if self.returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (self.returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        return {
            'Total Return (%)': total_return,
            'CAGR': cagr,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        if len(self.returns) == 0:
            return {}
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(self.returns, 5) * 100
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = self.returns[self.returns <= np.percentile(self.returns, 5)].mean() * 100
        
        # Skewness and Kurtosis
        skewness = self.returns.skew()
        kurtosis = self.returns.kurtosis()
        
        # Beta (if benchmark available - placeholder for now)
        beta = 1.0  # Would calculate vs benchmark if available
        
        return {
            'VaR 95% (%)': var_95,
            'CVaR 95% (%)': cvar_95,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Beta': beta
        }
    
    def _calculate_drawdown_metrics(self) -> Dict[str, float]:
        """Calculate drawdown-related metrics."""
        # High water mark
        high_water_mark = self.equity_series.cummax()
        
        # Drawdown series
        drawdown = (self.equity_series - high_water_mark) / high_water_mark
        
        # Maximum drawdown
        max_drawdown = drawdown.min() * 100
        
        # Calmar ratio (CAGR / |Max Drawdown|)
        cagr = self._calculate_return_metrics().get('CAGR', 0)
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Average drawdown
        drawdown_periods = drawdown[drawdown < 0]
        avg_drawdown = drawdown_periods.mean() * 100 if len(drawdown_periods) > 0 else 0
        
        # Drawdown duration analysis
        in_drawdown = drawdown < 0
        drawdown_duration = self._calculate_drawdown_duration(in_drawdown)
        
        return {
            'Max Drawdown (%)': max_drawdown,
            'Average Drawdown (%)': avg_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown Duration (days)': drawdown_duration
        }
    
    def _calculate_drawdown_duration(self, in_drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        if not in_drawdown.any():
            return 0
        
        # Find drawdown periods
        drawdown_changes = in_drawdown.diff()
        starts = in_drawdown.index[drawdown_changes & in_drawdown]
        ends = in_drawdown.index[drawdown_changes & ~in_drawdown]
        
        # Handle edge cases
        if in_drawdown.iloc[0]:
            starts = [in_drawdown.index[0]] + starts.tolist()
        if in_drawdown.iloc[-1]:
            ends = ends.tolist() + [in_drawdown.index[-1]]
        
        if len(starts) == 0 or len(ends) == 0:
            return 0
        
        # Calculate durations
        durations = [(end - start).days for start, end in zip(starts, ends)]
        return max(durations) if durations else 0
    
    def _calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculate trade-based metrics from trade log."""
        if self.trade_log is None or self.trade_log.empty:
            return {}
        
        # Ensure required columns exist
        required_cols = ['pnl']
        if not all(col in self.trade_log.columns for col in required_cols):
            logger.warning("Trade log missing required columns for trade metrics")
            return {}
        
        # Basic trade statistics
        total_trades = len(self.trade_log)
        winning_trades = len(self.trade_log[self.trade_log['pnl'] > 0])
        losing_trades = len(self.trade_log[self.trade_log['pnl'] < 0])
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = self.trade_log[self.trade_log['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trade_log[self.trade_log['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_trade = self.trade_log['pnl'].mean()
        avg_winner = self.trade_log[self.trade_log['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loser = self.trade_log[self.trade_log['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Largest winner/loser
        largest_winner = self.trade_log['pnl'].max()
        largest_loser = self.trade_log['pnl'].min()
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_wins()
        consecutive_losses = self._calculate_consecutive_losses()
        
        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Average Trade (Rs)': avg_trade,
            'Average Winner (Rs)': avg_winner,
            'Average Loser (Rs)': avg_loser,
            'Largest Winner (Rs)': largest_winner,
            'Largest Loser (Rs)': largest_loser,
            'Max Consecutive Wins': consecutive_wins,
            'Max Consecutive Losses': consecutive_losses
        }
    
    def _calculate_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades."""
        if self.trade_log is None or 'pnl' not in self.trade_log.columns:
            return 0
        
        wins = (self.trade_log['pnl'] > 0).astype(int)
        return self._max_consecutive(wins)
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades."""
        if self.trade_log is None or 'pnl' not in self.trade_log.columns:
            return 0
        
        losses = (self.trade_log['pnl'] < 0).astype(int)
        return self._max_consecutive(losses)
    
    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive occurrences in a binary series."""
        if series.empty:
            return 0
        
        # Find consecutive groups
        groups = (series != series.shift()).cumsum()
        consecutive_counts = series.groupby(groups).sum()
        
        return consecutive_counts.max() if not consecutive_counts.empty else 0
    
    def get_drawdown_series(self) -> pd.Series:
        """Get the drawdown time series for plotting."""
        high_water_mark = self.equity_series.cummax()
        drawdown = (self.equity_series - high_water_mark) / high_water_mark
        return drawdown
    
    def get_returns_series(self) -> pd.Series:
        """Get the returns time series."""
        return self.returns
    
    def display_metrics(self):
        """Display all metrics in a formatted way."""
        metrics = self.calculate_all_metrics()
        
        print("\n" + "="*60)
        print("           BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        # Group metrics by category
        return_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['return', 'cagr', 'volatility', 'sharpe', 'sortino'])}
        risk_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['var', 'cvar', 'skewness', 'kurtosis', 'beta'])}
        drawdown_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['drawdown', 'calmar', 'duration'])}
        trade_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['trades', 'win', 'profit', 'average', 'largest', 'consecutive'])}
        
        # Display Return Metrics
        if return_metrics:
            print("\n[RETURN METRICS]")
            print("-" * 40)
            for metric, value in return_metrics.items():
                if isinstance(value, (int, float)):
                    if 'ratio' in metric.lower():
                        print(f"{metric:<25}: {value:>8.3f}")
                    else:
                        print(f"{metric:<25}: {value:>8.2f}")
                else:
                    print(f"{metric:<25}: {value}")
        
        # Display Risk Metrics
        if risk_metrics:
            print("\n[RISK METRICS]")
            print("-" * 40)
            for metric, value in risk_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric:<25}: {value:>8.3f}")
                else:
                    print(f"{metric:<25}: {value}")
        
        # Display Drawdown Metrics
        if drawdown_metrics:
            print("\n[DRAWDOWN METRICS]")
            print("-" * 40)
            for metric, value in drawdown_metrics.items():
                if isinstance(value, (int, float)):
                    if 'days' in metric.lower():
                        print(f"{metric:<25}: {value:>8.0f}")
                    elif 'ratio' in metric.lower():
                        print(f"{metric:<25}: {value:>8.3f}")
                    else:
                        print(f"{metric:<25}: {value:>8.2f}")
                else:
                    print(f"{metric:<25}: {value}")
        
        # Display Trade Metrics
        if trade_metrics:
            print("\n[TRADE METRICS]")
            print("-" * 40)
            for metric, value in trade_metrics.items():
                if isinstance(value, (int, float)):
                    if 'Rs' in metric or 'rupee' in metric.lower():
                        print(f"{metric:<25}: {value:>10,.2f}")
                    elif '%' in metric:
                        print(f"{metric:<25}: {value:>8.2f}")
                    elif 'factor' in metric.lower():
                        if value == float('inf'):
                            print(f"{metric:<25}: {'INF':>8}")
                        else:
                            print(f"{metric:<25}: {value:>8.3f}")
                    else:
                        print(f"{metric:<25}: {value:>8.0f}")
                else:
                    print(f"{metric:<25}: {value}")
        
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        print("="*60)
    
    def export_metrics_to_csv(self, filepath: str):
        """Export all metrics to CSV file."""
        metrics = self.calculate_all_metrics()
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_df.to_csv(filepath, index=False)
        logger.info(f"Metrics exported to {filepath}")
    
    def export_metrics_to_excel(self, filepath: str):
        """Export all metrics to Excel file with multiple sheets."""
        metrics = self.calculate_all_metrics()

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main metrics sheet
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

            # Equity curve sheet - strip timezone from datetime index
            equity_df = pd.DataFrame({'Date': self.equity_series.index, 'Equity': self.equity_series.values})
            if pd.api.types.is_datetime64_any_dtype(equity_df['Date']):
                equity_df['Date'] = equity_df['Date'].dt.tz_localize(None)
            equity_df.to_excel(writer, sheet_name='Equity_Curve', index=False)

            # Returns sheet - strip timezone from datetime index
            returns_df = pd.DataFrame({'Date': self.returns.index, 'Returns': self.returns.values})
            if pd.api.types.is_datetime64_any_dtype(returns_df['Date']):
                returns_df['Date'] = returns_df['Date'].dt.tz_localize(None)
            returns_df.to_excel(writer, sheet_name='Returns', index=False)

            # Trade log sheet (if available) - strip timezone from datetime columns
            if self.trade_log is not None:
                trade_log_copy = self.trade_log.copy()
                for col in trade_log_copy.columns:
                    if pd.api.types.is_datetime64_any_dtype(trade_log_copy[col]):
                        trade_log_copy[col] = trade_log_copy[col].dt.tz_localize(None)
                trade_log_copy.to_excel(writer, sheet_name='Trade_Log', index=False)

        logger.info(f"Comprehensive metrics exported to {filepath}")
    
    def save_all_reports(self, output_dir):
        """Save all comprehensive reports and data to the specified directory."""
        import os
        from pathlib import Path
        
        # Convert to Path object if it's a string
        output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
        
        # Ensure directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving comprehensive backtest reports to {output_path}")
        
        # 1. Save metrics to CSV
        metrics_csv_path = output_path / 'backtest_metrics.csv'
        self.export_metrics_to_csv(str(metrics_csv_path))
        
        # 2. Save comprehensive Excel report
        excel_path = output_path / 'backtest_report.xlsx'
        self.export_metrics_to_excel(str(excel_path))
        
        # 3. Save individual data files
        # Equity curve
        if not self.equity_series.empty:
            equity_df = pd.DataFrame({
                'Date': self.equity_series.index, 
                'Equity': self.equity_series.values
            })
            equity_csv_path = output_path / 'equity_curve.csv'
            equity_df.to_csv(equity_csv_path, index=False)
            logger.info(f"Equity curve saved to {equity_csv_path}")
        
        # Returns series
        if not self.returns.empty:
            returns_df = pd.DataFrame({
                'Date': self.returns.index,
                'Returns': self.returns.values
            })
            returns_csv_path = output_path / 'returns.csv'
            returns_df.to_csv(returns_csv_path, index=False)
            logger.info(f"Returns series saved to {returns_csv_path}")
        
        # Trade log
        if self.trade_log is not None and not self.trade_log.empty:
            trade_log_path = output_path / 'trade_log.csv'
            self.trade_log.to_csv(trade_log_path, index=False)
            logger.info(f"Trade log saved to {trade_log_path}")
        
        # Drawdown series
        drawdown_series = self.get_drawdown_series()
        if not drawdown_series.empty:
            drawdown_df = pd.DataFrame({
                'Date': drawdown_series.index,
                'Drawdown': drawdown_series.values
            })
            drawdown_csv_path = output_path / 'drawdown.csv'
            drawdown_df.to_csv(drawdown_csv_path, index=False)
            logger.info(f"Drawdown series saved to {drawdown_csv_path}")
        
        # 4. Save a summary text report
        summary_path = output_path / 'backtest_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("BACKTEST PERFORMANCE SUMMARY\n")
            f.write("=" * 60 + "\n")
            
            metrics = self.calculate_all_metrics()
            
            # Categorize metrics
            return_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['return', 'cagr', 'volatility', 'sharpe', 'sortino'])}
            risk_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['var', 'cvar', 'skewness', 'kurtosis', 'beta'])}
            drawdown_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['drawdown', 'calmar', 'duration'])}
            trade_metrics = {k: v for k, v in metrics.items() if any(keyword in k.lower() for keyword in ['trades', 'win', 'profit', 'average', 'largest', 'consecutive'])}
            
            # Write categorized metrics
            for category_name, category_metrics in [
                ("RETURN METRICS", return_metrics),
                ("RISK METRICS", risk_metrics), 
                ("DRAWDOWN METRICS", drawdown_metrics),
                ("TRADE METRICS", trade_metrics)
            ]:
                if category_metrics:
                    f.write(f"\n[{category_name}]\n")
                    f.write("-" * 40 + "\n")
                    for metric, value in category_metrics.items():
                        if isinstance(value, (int, float)):
                            if 'ratio' in metric.lower():
                                f.write(f"{metric:<25}: {value:>8.3f}\n")
                            elif '%' in metric or 'days' in metric.lower():
                                f.write(f"{metric:<25}: {value:>8.2f}\n")
                            elif 'factor' in metric.lower() and value == float('inf'):
                                f.write(f"{metric:<25}: {'INF':>8}\n")
                            else:
                                f.write(f"{metric:<25}: {value:>8.2f}\n")
                        else:
                            f.write(f"{metric:<25}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Report generated successfully!\n")
        
        logger.info(f"Summary report saved to {summary_path}")
        logger.info("All backtest reports saved successfully!")