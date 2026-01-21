"""
Performance Benchmarking for Streaming vs Pandas Indicators

This module provides comprehensive performance benchmarking tools to compare
streaming indicators with traditional pandas-based indicator calculations.
It measures execution time, memory usage, and signal generation accuracy.

Features:
- Time-series benchmark with varying data sizes
- Memory usage profiling
- Signal accuracy validation
- Performance report generation
- Statistical analysis of performance differences

Example Usage:
    from utils.performance_benchmark import PerformanceBenchmark
    
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.generate_report(results)
"""

import time
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import statistics
import logging
import gc
import tracemalloc

# Import our modules
from Strategies.MA_crossover import MovingAverageCrossoverStrategy
from meluna.events import MarketEvent, SignalEvent

logger = logging.getLogger(__name__)


@dataclass 
class BenchmarkMetrics:
    """Metrics collected during performance benchmarking."""
    execution_time: float
    peak_memory_mb: float
    avg_memory_mb: float
    signal_count: int
    data_points_processed: int
    setup_time: float = 0.0
    signals_per_second: float = 0.0
    memory_per_datapoint_kb: float = 0.0


@dataclass
class ComparisonResult:
    """Result of comparing streaming vs pandas performance."""
    data_size: int
    streaming_metrics: BenchmarkMetrics
    pandas_metrics: BenchmarkMetrics
    time_improvement_factor: float = 0.0
    memory_improvement_factor: float = 0.0
    signals_match: bool = False
    signal_differences: List[Dict[str, Any]] = field(default_factory=list)


class MockDataHandler:
    """Mock data handler for benchmarking."""
    pass


class DataGenerator:
    """Generate realistic market data for benchmarking."""
    
    @staticmethod
    def generate_ohlcv_data(size: int, symbol: str = 'BENCH', 
                           start_price: float = 100.0, 
                           volatility: float = 0.02) -> pd.DataFrame:
        """
        Generate realistic OHLCV data for benchmarking.
        
        Args:
            size (int): Number of data points to generate
            symbol (str): Symbol name
            start_price (float): Starting price
            volatility (float): Price volatility factor
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        # Generate time series
        start_time = datetime(2023, 1, 1, 9, 30)
        timestamps = [start_time + timedelta(minutes=i) for i in range(size)]
        
        # Generate realistic price movements
        returns = np.random.normal(0, volatility, size)
        
        # Create cumulative prices with some mean reversion
        log_prices = np.log(start_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Generate OHLC from closes
        opens = np.roll(prices, 1)
        opens[0] = start_price
        
        # Add some intraday variation
        high_variation = np.random.uniform(0.001, 0.01, size)
        low_variation = np.random.uniform(-0.01, -0.001, size)
        
        highs = prices * (1 + high_variation)
        lows = prices * (1 + low_variation)
        
        # Ensure OHLC relationship is maintained
        for i in range(size):
            high_candidate = max(opens[i], prices[i], highs[i])
            low_candidate = min(opens[i], prices[i], lows[i])
            highs[i] = high_candidate
            lows[i] = low_candidate
        
        volumes = np.random.randint(1000, 10000, size)
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=timestamps)


class MemoryProfiler:
    """Profile memory usage during benchmark execution."""
    
    def __init__(self):
        self.measurements = []
        self.peak_memory = 0
        
    def start_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
        gc.collect()  # Clean up before measurement
        
    def stop_profiling(self) -> Tuple[float, float]:
        """
        Stop memory profiling and return metrics.
        
        Returns:
            Tuple[float, float]: (peak_memory_mb, avg_memory_mb)
        """
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / (1024 * 1024)
        current_mb = current / (1024 * 1024)
        
        return peak_mb, current_mb
        
    def measure_current(self):
        """Measure current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.measurements.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
        
    def get_average(self) -> float:
        """Get average memory usage."""
        return statistics.mean(self.measurements) if self.measurements else 0.0


class PerformanceBenchmark:
    """Main benchmarking class."""
    
    def __init__(self):
        self.data_generator = DataGenerator()
        self.memory_profiler = MemoryProfiler()
        
    def benchmark_streaming_strategy(self, data: pd.DataFrame, symbol: str = 'BENCH') -> BenchmarkMetrics:
        """
        Benchmark streaming indicator strategy.
        
        Args:
            data (pd.DataFrame): Market data
            symbol (str): Symbol name
            
        Returns:
            BenchmarkMetrics: Performance metrics
        """
        # Initialize strategy
        params = {
            'fast_ma_period': 10,
            'slow_ma_period': 20,
            'use_streaming_indicators': True
        }
        
        data_handler = MockDataHandler()
        
        # Measure setup time
        setup_start = time.perf_counter()
        strategy = MovingAverageCrossoverStrategy(params, data_handler, [symbol])
        setup_time = time.perf_counter() - setup_start
        
        # Start profiling
        self.memory_profiler.start_profiling()
        gc.collect()
        
        # Process data points
        start_time = time.perf_counter()
        all_signals = []
        
        for idx, row in data.iterrows():
            market_event = MarketEvent(
                symbol=symbol,
                timestamp=idx,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume'])
            )
            
            signals = strategy.on_market_data(market_event)
            all_signals.extend(signals)
            
            # Periodic memory measurement
            if len(all_signals) % 100 == 0:
                self.memory_profiler.measure_current()
        
        execution_time = time.perf_counter() - start_time
        
        # Stop profiling
        peak_memory, avg_memory = self.memory_profiler.stop_profiling()
        
        return BenchmarkMetrics(
            execution_time=execution_time,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            signal_count=len(all_signals),
            data_points_processed=len(data),
            setup_time=setup_time,
            signals_per_second=len(data) / execution_time if execution_time > 0 else 0,
            memory_per_datapoint_kb=(peak_memory * 1024) / len(data) if len(data) > 0 else 0
        )
        
    def benchmark_pandas_strategy(self, data: pd.DataFrame, symbol: str = 'BENCH') -> BenchmarkMetrics:
        """
        Benchmark pandas-based indicator strategy.
        
        Args:
            data (pd.DataFrame): Market data
            symbol (str): Symbol name
            
        Returns:
            BenchmarkMetrics: Performance metrics
        """
        # Initialize strategy  
        params = {
            'fast_ma_period': 10,
            'slow_ma_period': 20,
            'use_streaming_indicators': False
        }
        
        data_handler = MockDataHandler()
        
        # Measure setup and precomputation time
        setup_start = time.perf_counter()
        strategy = MovingAverageCrossoverStrategy(params, data_handler, [symbol])
        strategy.precompute_indicators({symbol: data})
        setup_time = time.perf_counter() - setup_start
        
        # Start profiling
        self.memory_profiler.start_profiling()
        gc.collect()
        
        # Process data points
        start_time = time.perf_counter()
        all_signals = []
        
        for idx, row in data.iterrows():
            market_event = MarketEvent(
                symbol=symbol,
                timestamp=idx,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume'])
            )
            
            signals = strategy.on_market_data(market_event)
            all_signals.extend(signals)
            
            # Periodic memory measurement
            if len(all_signals) % 100 == 0:
                self.memory_profiler.measure_current()
        
        execution_time = time.perf_counter() - start_time
        
        # Stop profiling
        peak_memory, avg_memory = self.memory_profiler.stop_profiling()
        
        return BenchmarkMetrics(
            execution_time=execution_time,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            signal_count=len(all_signals),
            data_points_processed=len(data),
            setup_time=setup_time,
            signals_per_second=len(data) / execution_time if execution_time > 0 else 0,
            memory_per_datapoint_kb=(peak_memory * 1024) / len(data) if len(data) > 0 else 0
        )
    
    def compare_strategies(self, data: pd.DataFrame, symbol: str = 'BENCH') -> ComparisonResult:
        """
        Compare streaming vs pandas strategies on the same data.
        
        Args:
            data (pd.DataFrame): Market data
            symbol (str): Symbol name
            
        Returns:
            ComparisonResult: Comparison metrics
        """
        logger.info(f"Benchmarking strategies on {len(data)} data points")
        
        # Benchmark streaming strategy
        logger.info("Benchmarking streaming strategy...")
        streaming_metrics = self.benchmark_streaming_strategy(data, symbol)
        
        # Force garbage collection between benchmarks
        gc.collect()
        
        # Benchmark pandas strategy
        logger.info("Benchmarking pandas strategy...")
        pandas_metrics = self.benchmark_pandas_strategy(data, symbol)
        
        # Calculate improvement factors
        time_improvement = (pandas_metrics.execution_time / streaming_metrics.execution_time 
                           if streaming_metrics.execution_time > 0 else 0)
        
        memory_improvement = (pandas_metrics.peak_memory_mb / streaming_metrics.peak_memory_mb
                             if streaming_metrics.peak_memory_mb > 0 else 0)
        
        # Check signal accuracy
        signals_match = streaming_metrics.signal_count == pandas_metrics.signal_count
        
        return ComparisonResult(
            data_size=len(data),
            streaming_metrics=streaming_metrics,
            pandas_metrics=pandas_metrics,
            time_improvement_factor=time_improvement,
            memory_improvement_factor=memory_improvement,
            signals_match=signals_match,
            signal_differences=[] if signals_match else [
                {
                    'streaming_signals': streaming_metrics.signal_count,
                    'pandas_signals': pandas_metrics.signal_count,
                    'difference': abs(streaming_metrics.signal_count - pandas_metrics.signal_count)
                }
            ]
        )
    
    def run_scalability_test(self, data_sizes: List[int] = None) -> List[ComparisonResult]:
        """
        Run scalability test with varying data sizes.
        
        Args:
            data_sizes (List[int]): List of data sizes to test
            
        Returns:
            List[ComparisonResult]: Results for each data size
        """
        if data_sizes is None:
            data_sizes = [100, 500, 1000, 2000, 5000, 10000]
        
        results = []
        
        for size in data_sizes:
            logger.info(f"Testing scalability with {size} data points")
            
            # Generate test data
            data = self.data_generator.generate_ohlcv_data(size)
            
            # Compare strategies
            result = self.compare_strategies(data)
            results.append(result)
            
            logger.info(f"Size {size}: Streaming {result.streaming_metrics.execution_time:.4f}s, "
                       f"Pandas {result.pandas_metrics.execution_time:.4f}s, "
                       f"Improvement: {result.time_improvement_factor:.2f}x")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Returns:
            Dict[str, Any]: Complete benchmark results
        """
        logger.info("Starting comprehensive performance benchmark")
        
        results = {
            'timestamp': datetime.now(),
            'scalability_results': [],
            'memory_analysis': {},
            'performance_summary': {}
        }
        
        # Run scalability tests
        logger.info("Running scalability tests...")
        scalability_results = self.run_scalability_test()
        results['scalability_results'] = scalability_results
        
        # Analyze memory usage patterns
        logger.info("Analyzing memory usage patterns...")
        memory_analysis = self._analyze_memory_patterns(scalability_results)
        results['memory_analysis'] = memory_analysis
        
        # Generate performance summary
        logger.info("Generating performance summary...")
        performance_summary = self._generate_performance_summary(scalability_results)
        results['performance_summary'] = performance_summary
        
        logger.info("Comprehensive benchmark completed")
        return results
    
    def _analyze_memory_patterns(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze memory usage patterns across different data sizes."""
        streaming_memory = [r.streaming_metrics.peak_memory_mb for r in results]
        pandas_memory = [r.pandas_metrics.peak_memory_mb for r in results]
        data_sizes = [r.data_size for r in results]
        
        return {
            'streaming_memory_growth': {
                'sizes': data_sizes,
                'memory_mb': streaming_memory,
                'memory_per_datapoint_trend': [m/s for m, s in zip(streaming_memory, data_sizes)]
            },
            'pandas_memory_growth': {
                'sizes': data_sizes,
                'memory_mb': pandas_memory,
                'memory_per_datapoint_trend': [m/s for m, s in zip(pandas_memory, data_sizes)]
            },
            'memory_efficiency': {
                'avg_improvement_factor': statistics.mean([r.memory_improvement_factor for r in results]),
                'max_improvement_factor': max([r.memory_improvement_factor for r in results]),
                'min_improvement_factor': min([r.memory_improvement_factor for r in results])
            }
        }
    
    def _generate_performance_summary(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate overall performance summary."""
        time_improvements = [r.time_improvement_factor for r in results]
        memory_improvements = [r.memory_improvement_factor for r in results]
        signal_accuracy = [r.signals_match for r in results]
        
        return {
            'time_performance': {
                'avg_improvement_factor': statistics.mean(time_improvements),
                'max_improvement_factor': max(time_improvements),
                'min_improvement_factor': min(time_improvements),
                'std_deviation': statistics.stdev(time_improvements) if len(time_improvements) > 1 else 0
            },
            'memory_performance': {
                'avg_improvement_factor': statistics.mean(memory_improvements),
                'max_improvement_factor': max(memory_improvements),
                'min_improvement_factor': min(memory_improvements),
                'std_deviation': statistics.stdev(memory_improvements) if len(memory_improvements) > 1 else 0
            },
            'signal_accuracy': {
                'accuracy_rate': sum(signal_accuracy) / len(signal_accuracy) if signal_accuracy else 0,
                'total_tests': len(signal_accuracy),
                'accurate_tests': sum(signal_accuracy)
            },
            'scalability_analysis': {
                'linear_scaling': self._analyze_scaling_behavior(results),
                'performance_degradation': self._analyze_performance_degradation(results)
            }
        }
    
    def _analyze_scaling_behavior(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze how performance scales with data size."""
        if len(results) < 2:
            return {'analysis': 'Insufficient data for scaling analysis'}
        
        sizes = [r.data_size for r in results]
        streaming_times = [r.streaming_metrics.execution_time for r in results]
        pandas_times = [r.pandas_metrics.execution_time for r in results]
        
        # Simple linear regression to check scaling
        def calculate_scaling_factor(sizes, times):
            if len(sizes) < 2:
                return 0
            # Calculate approximate scaling factor
            size_ratios = [sizes[i]/sizes[i-1] for i in range(1, len(sizes))]
            time_ratios = [times[i]/times[i-1] for i in range(1, len(times)) if times[i-1] > 0]
            
            if time_ratios:
                return statistics.mean(time_ratios) / statistics.mean(size_ratios)
            return 0
        
        streaming_scaling = calculate_scaling_factor(sizes, streaming_times)
        pandas_scaling = calculate_scaling_factor(sizes, pandas_times)
        
        return {
            'streaming_scaling_factor': streaming_scaling,
            'pandas_scaling_factor': pandas_scaling,
            'streaming_is_linear': abs(streaming_scaling - 1.0) < 0.2,
            'pandas_is_linear': abs(pandas_scaling - 1.0) < 0.2
        }
    
    def _analyze_performance_degradation(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze performance degradation patterns."""
        time_improvements = [r.time_improvement_factor for r in results]
        
        if len(time_improvements) < 2:
            return {'analysis': 'Insufficient data for degradation analysis'}
        
        # Check if improvement factor is stable across sizes
        improvement_stability = statistics.stdev(time_improvements) / statistics.mean(time_improvements)
        
        return {
            'improvement_stability': improvement_stability,
            'stable_performance': improvement_stability < 0.2,
            'performance_trend': 'improving' if time_improvements[-1] > time_improvements[0] else 'degrading'
        }
    
    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate human-readable performance report.
        
        Args:
            results (Dict[str, Any]): Benchmark results
            output_file (Optional[str]): Output file path
            
        Returns:
            str: Generated report
        """
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("STREAMING VS PANDAS INDICATORS PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {results['timestamp']}")
        report_lines.append("")
        
        # Performance Summary
        summary = results['performance_summary']
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        time_perf = summary['time_performance']
        report_lines.append(f"Average Time Improvement: {time_perf['avg_improvement_factor']:.2f}x faster")
        report_lines.append(f"Maximum Time Improvement: {time_perf['max_improvement_factor']:.2f}x faster")
        report_lines.append(f"Minimum Time Improvement: {time_perf['min_improvement_factor']:.2f}x faster")
        
        memory_perf = summary['memory_performance']
        report_lines.append(f"Average Memory Improvement: {memory_perf['avg_improvement_factor']:.2f}x more efficient")
        report_lines.append(f"Maximum Memory Improvement: {memory_perf['max_improvement_factor']:.2f}x more efficient")
        
        accuracy = summary['signal_accuracy']
        report_lines.append(f"Signal Accuracy: {accuracy['accuracy_rate']*100:.1f}% ({accuracy['accurate_tests']}/{accuracy['total_tests']} tests)")
        report_lines.append("")
        
        # Detailed Results
        report_lines.append("DETAILED RESULTS BY DATA SIZE")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Size':<8} {'Stream(s)':<10} {'Pandas(s)':<10} {'Time Imp':<10} {'Mem Imp':<10} {'Signals':<10}")
        report_lines.append("-" * 68)
        
        for result in results['scalability_results']:
            size = result.data_size
            stream_time = result.streaming_metrics.execution_time
            pandas_time = result.pandas_metrics.execution_time
            time_imp = result.time_improvement_factor
            mem_imp = result.memory_improvement_factor
            signals_match = "✓" if result.signals_match else "✗"
            
            report_lines.append(f"{size:<8} {stream_time:<10.4f} {pandas_time:<10.4f} {time_imp:<10.2f} {mem_imp:<10.2f} {signals_match:<10}")
        
        report_lines.append("")
        
        # Memory Analysis
        memory_analysis = results['memory_analysis']
        report_lines.append("MEMORY USAGE ANALYSIS")
        report_lines.append("-" * 40)
        
        efficiency = memory_analysis['memory_efficiency']
        report_lines.append(f"Average Memory Efficiency Gain: {efficiency['avg_improvement_factor']:.2f}x")
        report_lines.append(f"Peak Memory Efficiency Gain: {efficiency['max_improvement_factor']:.2f}x")
        report_lines.append("")
        
        # Scalability Analysis
        scaling = summary['scalability_analysis']
        report_lines.append("SCALABILITY ANALYSIS")
        report_lines.append("-" * 40)
        
        linear_scaling = scaling['linear_scaling']
        report_lines.append(f"Streaming Linear Scaling: {'Yes' if linear_scaling.get('streaming_is_linear') else 'No'}")
        report_lines.append(f"Pandas Linear Scaling: {'Yes' if linear_scaling.get('pandas_is_linear') else 'No'}")
        
        degradation = scaling['performance_degradation']
        report_lines.append(f"Performance Stability: {'High' if degradation.get('stable_performance') else 'Variable'}")
        report_lines.append(f"Performance Trend: {degradation.get('performance_trend', 'Unknown')}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        avg_time_improvement = time_perf['avg_improvement_factor']
        if avg_time_improvement > 5:
            report_lines.append("✓ Strong recommendation: Use streaming indicators for all new strategies")
        elif avg_time_improvement > 2:
            report_lines.append("✓ Recommendation: Consider streaming indicators for performance-critical applications")
        else:
            report_lines.append("⚠ Mixed results: Evaluate case-by-case benefits")
        
        if accuracy['accuracy_rate'] == 1.0:
            report_lines.append("✓ Signal accuracy is perfect - safe to migrate")
        else:
            report_lines.append("⚠ Signal differences detected - review migration carefully")
        
        if efficiency['avg_improvement_factor'] > 1.5:
            report_lines.append("✓ Significant memory savings - beneficial for large datasets")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = '\n'.join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text


def main():
    """Example usage of performance benchmarking."""
    logging.basicConfig(level=logging.INFO)
    
    benchmark = PerformanceBenchmark()
    
    print("Starting comprehensive performance benchmark...")
    print("This may take several minutes...")
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate and display report
    report = benchmark.generate_report(results)
    print(report)
    
    # Save report to file
    report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    benchmark.generate_report(results, report_file)
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()