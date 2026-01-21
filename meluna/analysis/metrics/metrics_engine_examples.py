# meluna/analysis/metrics_engine_examples.py

"""
MetricsEngine Usage Examples

This module provides comprehensive examples of how to use the MetricsEngine
for dashboard analytics and data processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from meluna.analysis.metrics.MetricsEngine import MetricsEngine
from meluna.analysis.metrics.BacktestMetrics import BacktestMetrics


def basic_usage_example():
    """
    Demonstrates basic MetricsEngine usage for loading and processing data.
    """
    print("=== Basic MetricsEngine Usage ===")
    
    # Initialize MetricsEngine with caching enabled
    engine = MetricsEngine(cache_size=10, enable_caching=True)
    
    # Paths to your parquet files
    trade_log_path = Path("results/my_strategy/v1/trade_log.parquet")
    equity_curve_path = Path("results/my_strategy/v1/equity_curve.parquet")
    
    try:
        # Load data with validation
        data = engine.load_data(trade_log_path, equity_curve_path, validate=True)
        
        print(f"✓ Data loaded successfully:")
        print(f"  - Trades: {len(data['trade_log'])}")
        print(f"  - Equity points: {len(data['equity_curve'])}")
        
        # Get data summary
        summary = engine.get_data_summary()
        print(f"✓ Data summary: {summary}")
        
        return engine, data
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None


def dashboard_integration_example():
    """
    Shows how to integrate MetricsEngine with dashboard applications.
    """
    print("\n=== Dashboard Integration Example ===")
    
    engine = MetricsEngine(enable_caching=True)
    
    # Simulate dashboard data loading workflow
    def load_backtest_data(strategy_name: str, version: str):
        """Load data for a specific backtest run."""
        results_path = Path(f"results/{strategy_name}/{version}")
        trade_path = results_path / "trade_log.parquet"
        equity_path = results_path / "equity_curve.parquet"
        
        # Validate files exist before loading
        validation_results = engine.validate_data_files(trade_path, equity_path)
        
        if not all(validation_results[key]['valid'] for key in validation_results):
            print(f"✗ Validation failed for {strategy_name}/{version}")
            return None
        
        # Load data
        data = engine.load_data(trade_path, equity_path)
        print(f"✓ Loaded {strategy_name}/{version}")
        return data
    
    # Simulate loading multiple strategy versions
    strategies = [
        ("momentum_strategy", "v1"),
        ("mean_reversion_strategy", "v2"),
        ("breakout_strategy", "v3")
    ]
    
    for strategy, version in strategies:
        data = load_backtest_data(strategy, version)
        if data:
            # Get quick metrics for dashboard display
            trade_count = len(data['trade_log'])
            equity_final = data['equity_curve']['equity'].iloc[-1]
            print(f"  {strategy} {version}: {trade_count} trades, Final Equity: ₹{equity_final:,.2f}")


def performance_optimization_example():
    """
    Demonstrates performance optimization techniques with MetricsEngine.
    """
    print("\n=== Performance Optimization Example ===")
    
    # Initialize with larger cache for better performance
    engine = MetricsEngine(cache_size=20, enable_caching=True)
    
    # Example: Loading the same data multiple times (simulates dashboard refreshes)
    trade_path = Path("results/strategy_a/v1/trade_log.parquet")
    equity_path = Path("results/strategy_a/v1/equity_curve.parquet")
    
    import time
    
    # First load (from disk)
    start_time = time.time()
    data1 = engine.load_data(trade_path, equity_path)
    first_load_time = time.time() - start_time
    
    # Second load (from cache)
    start_time = time.time()
    data2 = engine.load_data(trade_path, equity_path)
    second_load_time = time.time() - start_time
    
    print(f"✓ First load (disk): {first_load_time:.3f}s")
    print(f"✓ Second load (cache): {second_load_time:.3f}s")
    print(f"✓ Speedup: {first_load_time/second_load_time:.1f}x faster")
    
    # Cache statistics
    cache_stats = engine.get_cache_stats()
    print(f"✓ Cache stats: {cache_stats}")


def integration_with_backtest_metrics_example():
    """
    Shows how to integrate MetricsEngine with existing BacktestMetrics.
    """
    print("\n=== Integration with BacktestMetrics ===")
    
    engine = MetricsEngine()
    
    # Load data
    trade_path = Path("results/my_strategy/v1/trade_log.parquet")
    equity_path = Path("results/my_strategy/v1/equity_curve.parquet")
    
    try:
        data = engine.load_data(trade_path, equity_path)
        
        # Prepare data for BacktestMetrics
        equity_df = data['equity_curve']
        equity_series = equity_df.set_index('date')['equity']
        
        # Initialize BacktestMetrics
        metrics_calculator = BacktestMetrics(
            equity_curve=equity_series,
            trade_log=data['trade_log']
        )
        
        # Calculate comprehensive metrics
        metrics = metrics_calculator.calculate_all_metrics()
        
        print("✓ Key Performance Metrics:")
        key_metrics = [
            'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)',
            'Win Rate (%)', 'Profit Factor', 'Total Trades'
        ]
        
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
        
    except Exception as e:
        print(f"✗ Error in metrics calculation: {e}")


def error_handling_example():
    """
    Demonstrates robust error handling patterns.
    """
    print("\n=== Error Handling Example ===")
    
    engine = MetricsEngine()
    
    # Example 1: File not found
    try:
        data = engine.load_data("nonexistent_trades.parquet", "nonexistent_equity.parquet")
    except Exception as e:
        print(f"✓ Handled file not found: {type(e).__name__}")
    
    # Example 2: Validation with schema checking
    def validate_backtest_files(strategy_name: str, version: str):
        """Validate backtest files before processing."""
        results_path = Path(f"results/{strategy_name}/{version}")
        trade_path = results_path / "trade_log.parquet"
        equity_path = results_path / "equity_curve.parquet"
        
        # Pre-validate files
        validation_results = engine.validate_data_files(trade_path, equity_path)
        
        print(f"Validation results for {strategy_name}/{version}:")
        for data_type, result in validation_results.items():
            status = "✓ Valid" if result['valid'] else "✗ Invalid"
            print(f"  {data_type}: {status}")
            
            if not result['valid']:
                for error in result['errors']:
                    print(f"    Error: {error}")
        
        return all(result['valid'] for result in validation_results.values())
    
    # Test validation
    valid = validate_backtest_files("test_strategy", "v1")
    print(f"Overall validation: {'✓ Passed' if valid else '✗ Failed'}")


def threading_safety_example():
    """
    Demonstrates thread-safe usage for concurrent dashboard access.
    """
    print("\n=== Thread Safety Example ===")
    
    import threading
    import concurrent.futures
    
    engine = MetricsEngine(cache_size=5, enable_caching=True)
    
    def simulate_dashboard_user(user_id: int):
        """Simulate a dashboard user accessing data."""
        try:
            # Each user loads data and performs operations
            trade_path = Path("results/shared_strategy/v1/trade_log.parquet")
            equity_path = Path("results/shared_strategy/v1/equity_curve.parquet")
            
            # Load data (thread-safe)
            data = engine.load_data(trade_path, equity_path)
            
            # Perform typical dashboard operations
            summary = engine.get_data_summary()
            trade_data = engine.get_trade_data()
            equity_data = engine.get_equity_data()
            
            return {
                'user_id': user_id,
                'trade_count': len(trade_data) if trade_data else 0,
                'summary': summary
            }
            
        except Exception as e:
            return {'user_id': user_id, 'error': str(e)}
    
    # Simulate multiple concurrent users
    num_users = 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [executor.submit(simulate_dashboard_user, i) for i in range(num_users)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    print(f"✓ Processed {len(results)} concurrent users:")
    for result in results:
        if 'error' in result:
            print(f"  User {result['user_id']}: Error - {result['error']}")
        else:
            print(f"  User {result['user_id']}: {result['trade_count']} trades")


def memory_management_example():
    """
    Shows memory-efficient usage patterns for large datasets.
    """
    print("\n=== Memory Management Example ===")
    
    # Configure for memory efficiency
    engine = MetricsEngine(cache_size=3, enable_caching=True)  # Smaller cache
    
    def process_large_dataset(file_path_pairs):
        """Process multiple large datasets efficiently."""
        for i, (trade_path, equity_path) in enumerate(file_path_pairs):
            print(f"Processing dataset {i+1}/{len(file_path_pairs)}")
            
            try:
                # Load data
                data = engine.load_data(trade_path, equity_path)
                
                # Process data (simulate analysis)
                trade_count = len(data['trade_log'])
                equity_points = len(data['equity_curve'])
                
                print(f"  ✓ Processed: {trade_count} trades, {equity_points} equity points")
                
                # Get memory usage stats
                cache_stats = engine.get_cache_stats()
                print(f"  Cache size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
                
            except Exception as e:
                print(f"  ✗ Error processing dataset {i+1}: {e}")
    
    # Example dataset paths
    datasets = [
        ("results/strategy_1/v1/trade_log.parquet", "results/strategy_1/v1/equity_curve.parquet"),
        ("results/strategy_2/v1/trade_log.parquet", "results/strategy_2/v1/equity_curve.parquet"),
        ("results/strategy_3/v1/trade_log.parquet", "results/strategy_3/v1/equity_curve.parquet"),
    ]
    
    process_large_dataset(datasets)
    
    # Clear cache when done to free memory
    engine.clear_cache()
    print("✓ Cache cleared")


def create_sample_data():
    """
    Creates sample data files for testing the examples.
    This function helps users get started by creating realistic test data.
    """
    print("\n=== Creating Sample Data ===")
    
    # Create results directory structure
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample trade data
    np.random.seed(42)  # For reproducible examples
    
    trade_data = pd.DataFrame({
        'trade_id': [f'T{i:03d}' for i in range(1, 101)],
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], 100),
        'entry_timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
        'exit_timestamp': pd.date_range('2024-01-02', periods=100, freq='D'),
        'entry_price': np.random.normal(200, 50, 100),
        'exit_price': np.random.normal(202, 52, 100),
        'quantity': np.random.choice([10, 25, 50, 100], 100),
        'pnl': np.random.normal(50, 200, 100)
    })
    
    # Create sample equity data
    equity_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=150, freq='D'),
        'equity': 100000 + np.cumsum(np.random.normal(10, 500, 150))
    })
    
    # Save sample data
    trade_path = sample_dir / "trade_log.parquet"
    equity_path = sample_dir / "equity_curve.parquet"
    
    trade_data.to_parquet(trade_path)
    equity_data.to_parquet(equity_path)
    
    print(f"✓ Sample data created:")
    print(f"  Trade log: {trade_path}")
    print(f"  Equity curve: {equity_path}")
    print(f"  Use these files to test the examples above!")
    
    return trade_path, equity_path


def main():
    """
    Main function that runs all examples.
    """
    print("MetricsEngine Usage Examples")
    print("=" * 50)
    
    # Create sample data first
    sample_trade_path, sample_equity_path = create_sample_data()
    
    # Run examples (Note: Most will fail without real data files)
    print("\nNote: Most examples will show error handling since sample")
    print("data paths don't match the examples. This is intentional to")
    print("demonstrate error handling patterns.")
    
    basic_usage_example()
    dashboard_integration_example()
    performance_optimization_example()
    integration_with_backtest_metrics_example()
    error_handling_example()
    threading_safety_example()
    memory_management_example()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the sample_data/ directory")
    print("for test files you can use with MetricsEngine.")


if __name__ == "__main__":
    main()