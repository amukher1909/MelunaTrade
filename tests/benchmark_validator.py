"""Performance benchmark for DataValidator - validates < 100ms for 10k bars."""

import pandas as pd
import numpy as np
import time
from meluna.utils.data_validation import DataValidator


def create_test_data(num_bars: int) -> pd.DataFrame:
    """Create synthetic OHLCV data for benchmarking."""
    np.random.seed(42)
    base_price = 100

    # Generate realistic price movements
    returns = np.random.normal(0, 0.02, num_bars)
    closes = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=num_bars, freq='D'),
        'open': closes * (1 + np.random.uniform(-0.01, 0.01, num_bars)),
        'high': closes * (1 + np.random.uniform(0.01, 0.03, num_bars)),
        'low': closes * (1 + np.random.uniform(-0.03, -0.01, num_bars)),
        'close': closes,
        'volume': np.random.uniform(1000, 5000, num_bars)
    })

    return df


def benchmark_validation(num_bars: int, num_runs: int = 10) -> dict:
    """Benchmark validation performance."""
    df = create_test_data(num_bars)
    validator = DataValidator(interval='1d', strictness='warn')

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        errors = validator.validate(df)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    return {
        'num_bars': num_bars,
        'avg_time_ms': np.mean(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'std_time_ms': np.std(times)
    }


if __name__ == '__main__':
    print("=" * 60)
    print("DataValidator Performance Benchmark")
    print("=" * 60)
    print()

    test_sizes = [100, 1000, 5000, 10000, 20000]

    for size in test_sizes:
        result = benchmark_validation(size, num_runs=10)
        print(f"Bars: {result['num_bars']:>6} | "
              f"Avg: {result['avg_time_ms']:>6.2f}ms | "
              f"Min: {result['min_time_ms']:>6.2f}ms | "
              f"Max: {result['max_time_ms']:>6.2f}ms | "
              f"Std: {result['std_time_ms']:>5.2f}ms")

    print()
    print("=" * 60)
    print("Acceptance Criteria: < 100ms for 10,000 bars")

    result_10k = benchmark_validation(10000, num_runs=20)
    passed = result_10k['avg_time_ms'] < 100

    print(f"Result: {result_10k['avg_time_ms']:.2f}ms (avg over 20 runs)")
    print(f"Status: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60)
