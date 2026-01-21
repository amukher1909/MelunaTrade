# meluna/analysis/portfolio_metrics_examples.py

"""
Portfolio Metrics Usage Examples

This module demonstrates how to use the PortfolioMetrics class for comprehensive
portfolio performance analysis. These examples show real-world usage patterns
for strategy evaluation and dashboard integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from .PortfolioMetrics import PortfolioMetrics


def create_sample_equity_curve(start_value: float = 100000, 
                              periods: int = 252, 
                              annual_return: float = 0.12,
                              annual_volatility: float = 0.16,
                              seed: int = 42) -> pd.Series:
    """
    Create a realistic equity curve for demonstration purposes.
    
    Args:
        start_value: Starting portfolio value
        periods: Number of trading days
        annual_return: Expected annual return
        annual_volatility: Annual volatility
        seed: Random seed for reproducibility
        
    Returns:
        Pandas Series with datetime index representing equity curve
    """
    np.random.seed(seed)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=periods, freq='B')  # Business days
    
    # Generate daily returns with realistic characteristics
    daily_return = annual_return / 252
    daily_vol = annual_volatility / np.sqrt(252)
    
    # Add some autocorrelation and fat tails
    returns = np.random.normal(daily_return, daily_vol, periods)
    
    # Add autocorrelation
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    # Add occasional extreme movements (fat tails)
    extreme_events = np.random.choice(periods, size=int(periods * 0.02), replace=False)
    returns[extreme_events] *= np.random.choice([-2, 2], size=len(extreme_events))
    
    # Calculate cumulative equity curve
    equity_values = start_value * (1 + returns).cumprod()
    
    return pd.Series(equity_values, index=dates)


def basic_usage_example():
    """Demonstrate basic PortfolioMetrics usage."""
    print("=" * 80)
    print("BASIC PORTFOLIO METRICS USAGE EXAMPLE")
    print("=" * 80)
    
    # Create sample equity curve
    equity_curve = create_sample_equity_curve(
        start_value=100000,
        periods=252,  # 1 year of trading days
        annual_return=0.15,
        annual_volatility=0.20
    )
    
    print(f"Equity curve: {len(equity_curve)} observations")
    print(f"Date range: {equity_curve.index[0].date()} to {equity_curve.index[-1].date()}")
    print(f"Start value: ₹{equity_curve.iloc[0]:,.2f}")
    print(f"End value: ₹{equity_curve.iloc[-1]:,.2f}")
    
    # Initialize PortfolioMetrics
    portfolio_metrics = PortfolioMetrics(
        equity_curve=equity_curve,
        risk_free_rate=0.03,  # 3% risk-free rate
        trading_days_per_year=252
    )
    
    # Calculate all metrics
    metrics = portfolio_metrics.calculate_all_metrics()
    
    # Display comprehensive metrics
    portfolio_metrics.display_metrics()
    
    # Show specific metrics programmatically
    print("\n" + "=" * 50)
    print("KEY METRICS SUMMARY")
    print("=" * 50)
    print(f"Total Return:     {metrics['Total Return (%)']:>8.2f}%")
    print(f"CAGR:             {metrics['CAGR (%)']:>8.2f}%")
    print(f"Sharpe Ratio:     {metrics['Sharpe Ratio']:>8.2f}")
    print(f"Max Drawdown:     {metrics['Max Drawdown (%)']:>8.2f}%")
    print(f"VaR 95%:          {metrics['VaR 95% (%)']:>8.2f}%")
    
    return portfolio_metrics


def multi_strategy_comparison_example():
    """Demonstrate comparing multiple strategies."""
    print("\n" + "=" * 80)
    print("MULTI-STRATEGY COMPARISON EXAMPLE")
    print("=" * 80)
    
    strategies = {
        'Conservative': {'return': 0.08, 'volatility': 0.12},
        'Balanced': {'return': 0.12, 'volatility': 0.16},
        'Aggressive': {'return': 0.18, 'volatility': 0.25}
    }
    
    comparison_results = {}
    
    for strategy_name, params in strategies.items():
        print(f"\nAnalyzing {strategy_name} Strategy...")
        
        # Generate equity curve for strategy
        equity_curve = create_sample_equity_curve(
            start_value=100000,
            periods=252 * 3,  # 3 years
            annual_return=params['return'],
            annual_volatility=params['volatility'],
            seed=hash(strategy_name) % 1000  # Different seed per strategy
        )
        
        # Calculate metrics
        portfolio_metrics = PortfolioMetrics(equity_curve=equity_curve, risk_free_rate=0.03)
        metrics = portfolio_metrics.calculate_all_metrics()
        
        # Store key metrics for comparison
        comparison_results[strategy_name] = {
            'CAGR (%)': metrics['CAGR (%)'],
            'Sharpe Ratio': metrics['Sharpe Ratio'],
            'Max Drawdown (%)': metrics['Max Drawdown (%)'],
            'VaR 95% (%)': metrics['VaR 95% (%)'],
            'Sortino Ratio': metrics['Sortino Ratio']
        }
    
    # Display comparison table
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    
    metrics_names = list(comparison_results['Conservative'].keys())
    print(f"{'Strategy':<12} " + " ".join(f"{metric:>12}" for metric in metrics_names))
    print("-" * 80)
    
    for strategy_name, results in comparison_results.items():
        values_str = " ".join(f"{results[metric]:>12.2f}" for metric in metrics_names)
        print(f"{strategy_name:<12} {values_str}")
    
    return comparison_results


def rolling_analysis_example():
    """Demonstrate rolling metrics analysis."""
    print("\n" + "=" * 80)
    print("ROLLING METRICS ANALYSIS EXAMPLE")
    print("=" * 80)
    
    # Create longer equity curve for rolling analysis
    equity_curve = create_sample_equity_curve(
        start_value=100000,
        periods=252 * 2,  # 2 years
        annual_return=0.12,
        annual_volatility=0.18
    )
    
    portfolio_metrics = PortfolioMetrics(equity_curve=equity_curve, risk_free_rate=0.025)
    
    # Calculate rolling metrics
    rolling_metrics = portfolio_metrics.get_rolling_metrics(window_days=63)  # Quarter
    
    print("Rolling metrics calculated successfully!")
    print(f"Available rolling metrics: {list(rolling_metrics.keys())}")
    
    # Show latest rolling values
    print("\nLatest Rolling Metrics (63-day window):")
    print("-" * 50)
    for metric_name, series in rolling_metrics.items():
        if not series.dropna().empty:
            latest_value = series.dropna().iloc[-1]
            print(f"{metric_name:<25} {latest_value:>12.4f}")
    
    # Display comprehensive metrics with rolling
    portfolio_metrics.display_metrics(include_rolling=True, rolling_window=63)
    
    return rolling_metrics


def edge_cases_example():
    """Demonstrate handling of edge cases."""
    print("\n" + "=" * 80)
    print("EDGE CASES HANDLING EXAMPLE")
    print("=" * 80)
    
    # Test case 1: Constant equity curve (no volatility)
    print("\n1. Constant Equity Curve (No Volatility):")
    print("-" * 50)
    
    constant_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    constant_curve = pd.Series([100000] * 30, index=constant_dates)
    
    try:
        constant_metrics = PortfolioMetrics(equity_curve=constant_curve)
        constant_results = constant_metrics.calculate_all_metrics()
        print(f"Total Return: {constant_results['Total Return (%)']}%")
        print(f"Volatility: {constant_results['Annualized Volatility (%)']}%")
        print(f"Sharpe Ratio: {constant_results['Sharpe Ratio']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 2: Very small dataset
    print("\n2. Small Dataset (Warning Test):")
    print("-" * 50)
    
    small_dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    small_values = np.random.uniform(99000, 101000, 10)
    small_curve = pd.Series(small_values, index=small_dates)
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            small_metrics = PortfolioMetrics(equity_curve=small_curve)
            small_results = small_metrics.calculate_all_metrics()
            
            if w:
                print(f"Warning captured: {w[0].message}")
            
            print("Metrics calculated despite small dataset:")
            print(f"VaR 95%: {small_results['VaR 95% (%)']}%")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 3: Data validation
    print("\n3. Data Validation:")
    print("-" * 50)
    
    portfolio_metrics = PortfolioMetrics(equity_curve=create_sample_equity_curve(periods=100))
    validation_results = portfolio_metrics.validate_minimum_data_requirements()
    
    print("Data sufficiency validation:")
    for requirement, sufficient in validation_results.items():
        status = "✓" if sufficient else "✗"
        print(f"  {requirement:<25} {status}")


def performance_benchmark_example():
    """Demonstrate performance with large datasets."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK EXAMPLE")
    print("=" * 80)
    
    import time
    
    # Test with different dataset sizes
    test_sizes = [252, 252*2, 252*5, 252*10]  # 1, 2, 5, 10 years
    
    print("Performance benchmarks:")
    print(f"{'Dataset Size':<15} {'Time (seconds)':<15} {'Observations':<15}")
    print("-" * 50)
    
    for size in test_sizes:
        # Create large equity curve
        equity_curve = create_sample_equity_curve(periods=size)
        
        # Time the calculation
        start_time = time.time()
        portfolio_metrics = PortfolioMetrics(equity_curve=equity_curve)
        metrics = portfolio_metrics.calculate_all_metrics()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"{size:<15} {elapsed_time:<15.4f} {len(equity_curve):<15}")
    
    print(f"\nPerformance is excellent even with large datasets!")
    print(f"Latest calculation: {len(metrics)} metrics computed")


def main():
    """Run all examples."""
    print("PORTFOLIO METRICS COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    print("This demonstrates the PortfolioMetrics module capabilities")
    print("for comprehensive portfolio performance analysis.\n")
    
    # Run all examples
    basic_usage_example()
    multi_strategy_comparison_example()
    rolling_analysis_example()
    edge_cases_example()
    performance_benchmark_example()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nThe PortfolioMetrics module is ready for production use.")
    print("Integration points:")
    print("- Dashboard analytics engine")
    print("- Strategy comparison tools")
    print("- Risk management systems")
    print("- Performance reporting")


if __name__ == "__main__":
    main()