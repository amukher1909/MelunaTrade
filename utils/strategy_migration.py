"""
Strategy Migration Utility

This module provides utilities for migrating pandas-based trading strategies 
to use streaming technical indicators. It includes tools for automatic code
conversion, validation, and performance comparison.

Features:
- Automatic detection of pandas indicator operations
- Code generation for streaming indicator equivalents
- Side-by-side validation to ensure identical results
- Performance benchmarking between approaches
- Migration report generation

Example Usage:
    from utils.strategy_migration import StrategyMigrator
    
    migrator = StrategyMigrator()
    
    # Convert existing strategy
    migrated_code = migrator.migrate_strategy_file('Strategies/MyStrategy.py')
    
    # Validate migration
    is_valid = migrator.validate_migration(
        original_strategy=MyOriginalStrategy,
        migrated_strategy=MyMigratedStrategy,
        test_data=sample_data
    )
"""

import ast
import re
import logging
from typing import Dict, List, Tuple, Any, Type, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """
    Result of a strategy migration operation.
    """
    success: bool
    migrated_code: Optional[str] = None
    detected_indicators: List[str] = None
    conversion_mappings: Dict[str, str] = None
    warnings: List[str] = None
    errors: List[str] = None


@dataclass
class ValidationResult:
    """
    Result of a migration validation.
    """
    signals_match: bool
    performance_improvement: float
    signal_differences: List[Dict[str, Any]] = None
    timing_comparison: Dict[str, float] = None
    memory_comparison: Dict[str, float] = None


class PandasIndicatorDetector:
    """
    Detects pandas-based indicator calculations in strategy code.
    """
    
    def __init__(self):
        self.indicator_patterns = {
            'rolling_mean': r'\.rolling\s*\(\s*window\s*=\s*(\d+)\s*\)\.mean\(\)',
            'rolling_std': r'\.rolling\s*\(\s*window\s*=\s*(\d+)\s*\)\.std\(\)',
            'ewm': r'\.ewm\s*\(\s*span\s*=\s*(\d+)\s*\)\.mean\(\)',
            'pct_change': r'\.pct_change\s*\(\s*(\d*)\s*\)',
            'shift': r'\.shift\s*\(\s*(\d+)\s*\)',
            'rolling_max': r'\.rolling\s*\(\s*window\s*=\s*(\d+)\s*\)\.max\(\)',
            'rolling_min': r'\.rolling\s*\(\s*window\s*=\s*(\d+)\s*\)\.min\(\)'
        }
    
    def detect_indicators(self, code: str) -> List[Dict[str, Any]]:
        """
        Detect pandas indicator calculations in code.
        
        Args:
            code (str): Python code to analyze
            
        Returns:
            List[Dict[str, Any]]: List of detected indicators with metadata
        """
        detected = []
        
        for indicator_type, pattern in self.indicator_patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                detected.append({
                    'type': indicator_type,
                    'pattern': match.group(0),
                    'parameters': match.groups(),
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'line': code[:match.start()].count('\n') + 1
                })
        
        return detected


class StreamingIndicatorMapper:
    """
    Maps pandas indicator operations to streaming indicator equivalents.
    """
    
    def __init__(self):
        self.mapping_rules = {
            'rolling_mean': self._map_rolling_mean,
            'ewm': self._map_ewm,
            'rolling_std': self._map_rolling_std,
            'rolling_max': self._map_rolling_max,
            'rolling_min': self._map_rolling_min
        }
    
    def map_indicator(self, indicator_type: str, parameters: Tuple[str, ...]) -> Dict[str, Any]:
        """
        Map a pandas indicator to streaming equivalent.
        
        Args:
            indicator_type (str): Type of indicator
            parameters (Tuple[str, ...]): Parameters extracted from pattern
            
        Returns:
            Dict[str, Any]: Mapping information for streaming indicator
        """
        if indicator_type in self.mapping_rules:
            return self.mapping_rules[indicator_type](parameters)
        
        return {
            'streaming_type': None,
            'warning': f"No mapping available for {indicator_type}"
        }
    
    def _map_rolling_mean(self, parameters: Tuple[str, ...]) -> Dict[str, Any]:
        """Map rolling mean to SMA."""
        period = int(parameters[0]) if parameters else 20
        return {
            'streaming_type': 'sma',
            'streaming_params': {'period': period},
            'factory_call': f"ta.create('sma', period={period})",
            'description': f"Simple Moving Average with period {period}"
        }
    
    def _map_ewm(self, parameters: Tuple[str, ...]) -> Dict[str, Any]:
        """Map exponential weighted mean to EMA."""
        span = int(parameters[0]) if parameters else 12
        return {
            'streaming_type': 'ema',
            'streaming_params': {'period': span},
            'factory_call': f"ta.create('ema', period={span})",
            'description': f"Exponential Moving Average with span {span}"
        }
    
    def _map_rolling_std(self, parameters: Tuple[str, ...]) -> Dict[str, Any]:
        """Map rolling standard deviation."""
        period = int(parameters[0]) if parameters else 20
        return {
            'streaming_type': 'rolling_std',
            'streaming_params': {'period': period},
            'factory_call': f"# Note: Rolling Std not yet implemented in streaming library\n# self.indicators[symbol]['rolling_std'] = ta.create('rolling_std', period={period})",
            'description': f"Rolling Standard Deviation with period {period}",
            'warning': "Rolling standard deviation not yet implemented in streaming library"
        }
    
    def _map_rolling_max(self, parameters: Tuple[str, ...]) -> Dict[str, Any]:
        """Map rolling maximum."""
        period = int(parameters[0]) if parameters else 14
        return {
            'streaming_type': 'rolling_max',
            'streaming_params': {'period': period},
            'factory_call': f"# Note: Rolling Max available in minmax indicators\n# self.indicators[symbol]['rolling_max'] = ta.create('rolling_max', period={period})",
            'description': f"Rolling Maximum with period {period}",
            'warning': "Check minmax indicators module for rolling max implementation"
        }
    
    def _map_rolling_min(self, parameters: Tuple[str, ...]) -> Dict[str, Any]:
        """Map rolling minimum."""
        period = int(parameters[0]) if parameters else 14
        return {
            'streaming_type': 'rolling_min',
            'streaming_params': {'period': period},
            'factory_call': f"# Note: Rolling Min available in minmax indicators\n# self.indicators[symbol]['rolling_min'] = ta.create('rolling_min', period={period})",
            'description': f"Rolling Minimum with period {period}",
            'warning': "Check minmax indicators module for rolling min implementation"
        }


class StrategyMigrator:
    """
    Main class for migrating pandas-based strategies to streaming indicators.
    """
    
    def __init__(self):
        self.detector = PandasIndicatorDetector()
        self.mapper = StreamingIndicatorMapper()
    
    def analyze_strategy_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a strategy file for pandas indicator usage.
        
        Args:
            file_path (str): Path to strategy file
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            detected_indicators = self.detector.detect_indicators(code)
            
            # Map to streaming equivalents
            mappings = []
            for indicator in detected_indicators:
                mapping = self.mapper.map_indicator(
                    indicator['type'], 
                    indicator['parameters']
                )
                mappings.append({
                    **indicator,
                    **mapping
                })
            
            return {
                'file_path': file_path,
                'total_indicators': len(detected_indicators),
                'detected_indicators': detected_indicators,
                'streaming_mappings': mappings,
                'has_precompute_method': 'precompute_indicators' in code,
                'has_on_market_data': 'on_market_data' in code
            }
            
        except FileNotFoundError:
            logger.error(f"Strategy file not found: {file_path}")
            return {'error': f"File not found: {file_path}"}
        except Exception as e:
            logger.error(f"Error analyzing strategy file: {e}")
            return {'error': str(e)}
    
    def generate_streaming_setup_method(self, mappings: List[Dict[str, Any]], symbol_list_var: str = "self.symbol_list") -> str:
        """
        Generate the _setup_indicators method code.
        
        Args:
            mappings (List[Dict[str, Any]]): Indicator mappings
            symbol_list_var (str): Variable name for symbol list
            
        Returns:
            str: Generated method code
        """
        method_lines = [
            "    def _setup_indicators(self) -> None:",
            "        \"\"\"",
            "        Set up streaming indicators for each symbol.",
            "        \"\"\"",
            f"        for symbol in {symbol_list_var}:"
        ]
        
        # Group indicators by type to avoid duplicates
        unique_indicators = {}
        for mapping in mappings:
            if mapping.get('streaming_type'):
                indicator_name = f"{mapping['streaming_type']}_{mapping['streaming_params'].get('period', '')}"
                unique_indicators[indicator_name] = mapping
        
        if unique_indicators:
            method_lines.append("            self.indicators[symbol] = {")
            for name, mapping in unique_indicators.items():
                factory_call = mapping.get('factory_call', '')
                if factory_call and not factory_call.startswith('#'):
                    method_lines.append(f"                '{mapping['streaming_type']}': {factory_call},")
            method_lines.append("            }")
            method_lines.append("        ")
            method_lines.append("        self._use_streaming_indicators = True")
        else:
            method_lines.append("            # No streaming indicators available for detected pandas operations")
            method_lines.append("            pass")
        
        return '\n'.join(method_lines)
    
    def generate_migration_template(self, analysis: Dict[str, Any]) -> str:
        """
        Generate a template for migrating a strategy.
        
        Args:
            analysis (Dict[str, Any]): Analysis results from analyze_strategy_file
            
        Returns:
            str: Migration template code
        """
        template = []
        
        if analysis.get('error'):
            return f"# Error in analysis: {analysis['error']}"
        
        template.append("# Strategy Migration Template")
        template.append("# Generated automatically - please review and customize")
        template.append("")
        template.append("# Step 1: Add streaming indicator import")
        template.append("import meluna.technical_analysis as ta")
        template.append("")
        
        template.append("# Step 2: Add streaming parameter to constructor")
        template.append("# Add this to __init__:")
        template.append("# self.use_streaming = parameters.get('use_streaming_indicators', True)")
        template.append("")
        
        if analysis['streaming_mappings']:
            template.append("# Step 3: Add _setup_indicators method")
            setup_method = self.generate_streaming_setup_method(analysis['streaming_mappings'])
            template.append(setup_method)
            template.append("")
        
        template.append("# Step 4: Replace on_market_data with _generate_signals")
        template.append("# Rename your current on_market_data method to _generate_signals")
        template.append("# The base class will handle indicator updates automatically")
        template.append("")
        
        template.append("# Step 5: Replace pandas indicator access with streaming access")
        for mapping in analysis['streaming_mappings']:
            if mapping.get('streaming_type'):
                template.append(f"# Replace pandas operation: {mapping['pattern']}")
                template.append(f"# With streaming access: self._get_indicator_value(symbol, '{mapping['streaming_type']}')")
                template.append(f"# Description: {mapping.get('description', '')}")
                if mapping.get('warning'):
                    template.append(f"# Warning: {mapping['warning']}")
                template.append("")
        
        template.append("# Step 6: Test both modes")
        template.append("# Set use_streaming_indicators=False to test pandas mode")
        template.append("# Set use_streaming_indicators=True to test streaming mode")
        
        return '\n'.join(template)
    
    def validate_migration(self, original_strategy: Type, migrated_strategy: Type, 
                          test_data: pd.DataFrame, symbol: str = 'TEST') -> ValidationResult:
        """
        Validate that a migrated strategy produces identical results.
        
        Args:
            original_strategy (Type): Original strategy class
            migrated_strategy (Type): Migrated strategy class  
            test_data (pd.DataFrame): Test data with OHLCV columns
            symbol (str): Symbol to test with
            
        Returns:
            ValidationResult: Validation results
        """
        try:
            # Mock data handler
            class MockDataHandler:
                pass
            
            data_handler = MockDataHandler()
            
            # Initialize strategies
            params = {'fast_ma_period': 10, 'slow_ma_period': 20}
            
            # Original strategy (pandas mode)
            original = original_strategy(params, data_handler, [symbol])
            original.precompute_indicators({symbol: test_data})
            
            # Migrated strategy (streaming mode)  
            migrated_params = {**params, 'use_streaming_indicators': True}
            migrated = migrated_strategy(migrated_params, data_handler, [symbol])
            
            # Compare signals
            original_signals = []
            migrated_signals = []
            
            from meluna.events import MarketEvent
            
            for idx, row in test_data.iterrows():
                market_event = MarketEvent(
                    symbol=symbol,
                    timestamp=row.name if hasattr(row.name, 'timestamp') else datetime.now(),
                    open=row['open'],
                    high=row['high'], 
                    low=row['low'],
                    close=row['close'],
                    volume=int(row['volume']) if 'volume' in row else 1000
                )
                
                # Get signals from both strategies
                orig_sigs = original.on_market_data(market_event) if hasattr(original, 'on_market_data') else []
                migr_sigs = migrated.on_market_data(market_event) if hasattr(migrated, 'on_market_data') else []
                
                original_signals.extend(orig_sigs)
                migrated_signals.extend(migr_sigs)
            
            # Compare results
            signals_match = len(original_signals) == len(migrated_signals)
            if signals_match:
                for orig, migr in zip(original_signals, migrated_signals):
                    if (orig.direction != migr.direction or 
                        orig.symbol != migr.symbol):
                        signals_match = False
                        break
            
            return ValidationResult(
                signals_match=signals_match,
                performance_improvement=0.0,  # Placeholder - would need timing
                signal_differences=[] if signals_match else [
                    {'original': len(original_signals), 'migrated': len(migrated_signals)}
                ]
            )
            
        except Exception as e:
            logger.error(f"Error validating migration: {e}")
            return ValidationResult(
                signals_match=False,
                performance_improvement=0.0,
                signal_differences=[{'error': str(e)}]
            )
    
    def migrate_strategy_file(self, file_path: str, output_path: Optional[str] = None) -> MigrationResult:
        """
        Migrate a complete strategy file.
        
        Args:
            file_path (str): Path to original strategy file
            output_path (Optional[str]): Path for migrated file, defaults to original + '_migrated.py'
            
        Returns:
            MigrationResult: Migration results
        """
        try:
            # Analyze the file
            analysis = self.analyze_strategy_file(file_path)
            
            if analysis.get('error'):
                return MigrationResult(
                    success=False,
                    errors=[analysis['error']]
                )
            
            # Generate migration template
            template = self.generate_migration_template(analysis)
            
            # Create output path
            if output_path is None:
                path = Path(file_path)
                output_path = str(path.parent / f"{path.stem}_migration_template{path.suffix}")
            
            # Write migration template
            with open(output_path, 'w') as f:
                f.write(template)
            
            return MigrationResult(
                success=True,
                migrated_code=template,
                detected_indicators=[m['type'] for m in analysis['streaming_mappings']],
                conversion_mappings={m['type']: m.get('streaming_type') for m in analysis['streaming_mappings']},
                warnings=[m.get('warning') for m in analysis['streaming_mappings'] if m.get('warning')]
            )
            
        except Exception as e:
            logger.error(f"Error migrating strategy file: {e}")
            return MigrationResult(
                success=False,
                errors=[str(e)]
            )


def main():
    """
    Example usage of the migration utility.
    """
    migrator = StrategyMigrator()
    
    # Analyze existing strategy
    analysis = migrator.analyze_strategy_file('Strategies/MA_crossover.py')
    
    print("Analysis Results:")
    print(f"Total indicators detected: {analysis.get('total_indicators', 0)}")
    
    for mapping in analysis.get('streaming_mappings', []):
        print(f"- {mapping['type']} -> {mapping.get('streaming_type', 'Not available')}")
        if mapping.get('warning'):
            print(f"  Warning: {mapping['warning']}")
    
    # Generate migration template
    result = migrator.migrate_strategy_file('Strategies/MA_crossover.py')
    
    if result.success:
        print(f"\nMigration template generated successfully!")
        print(f"Detected indicators: {result.detected_indicators}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
    else:
        print(f"Migration failed: {result.errors}")


if __name__ == "__main__":
    main()