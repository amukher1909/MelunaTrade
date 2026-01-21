#!/usr/bin/env python3
"""
Binance Calculator Validator for Liquidation UAT (Issue #152).

This utility helps organize manual validation data against Binance Futures Calculator.
Since Binance calculator is web-based (no API), this class generates test scenarios
and DataFrame structures for manual validation.

Binance Futures Calculator: https://www.binance.com/en/futures/tools/calculator

Usage:
    validator = BinanceCalculatorValidator()
    test_cases = validator.generate_validation_matrix()
    df = validator.create_binance_validation_df(test_cases)
    # Export to Excel with empty "Binance Calculator" column
    # User manually enters Binance values
    # Re-run validation to compare
"""

import pandas as pd
from typing import Dict, List, Any, Optional


class BinanceCalculatorValidator:
    """
    Cross-validates liquidation prices against Binance Futures Calculator.

    Note: Binance calculator is web-based, so this utility helps
    organize manual validation data.
    """

    # Default maintenance margin rate (Binance default for most symbols)
    DEFAULT_MMR = 0.004  # 0.4%

    @staticmethod
    def get_expected_liquidation_price(
        entry_price: float,
        leverage: float,
        direction: str,
        mmr: float = DEFAULT_MMR
    ) -> Dict[str, Any]:
        """
        Calculate expected liquidation price using Binance formula.

        Binance Isolated Margin Liquidation Formula:
        - LONG:  liq_price = entry_price * (1 - 1/leverage + MMR)
        - SHORT: liq_price = entry_price * (1 + 1/leverage - MMR)

        Args:
            entry_price: Position entry price
            leverage: Leverage multiplier (e.g., 10.0 for 10x)
            direction: 'LONG' or 'SHORT'
            mmr: Maintenance margin rate (default 0.004 = 0.4%)

        Returns:
            Dictionary with:
            - liquidation_price: Calculated liquidation price
            - formula: Formula used for calculation
            - calculation_steps: Step-by-step breakdown
            - mmr: MMR used in calculation

        Example:
            >>> validator = BinanceCalculatorValidator()
            >>> result = validator.get_expected_liquidation_price(40000, 10.0, 'LONG')
            >>> result['liquidation_price']
            36160.0
        """
        if direction == 'LONG':
            # LONG liquidation: price drops below entry
            liq_price = entry_price * (1 - 1/leverage + mmr)
            formula = f"entry × (1 - 1/leverage + MMR)"
            multiplier = f"(1 - {1/leverage:.6f} + {mmr})"
        elif direction == 'SHORT':
            # SHORT liquidation: price rises above entry
            liq_price = entry_price * (1 + 1/leverage - mmr)
            formula = f"entry × (1 + 1/leverage - MMR)"
            multiplier = f"(1 + {1/leverage:.6f} - {mmr})"
        else:
            raise ValueError(f"Invalid direction '{direction}'. Must be 'LONG' or 'SHORT'.")

        calculation_steps = [
            f"Direction: {direction}",
            f"Entry Price: ${entry_price:,.2f}",
            f"Leverage: {leverage}x",
            f"MMR: {mmr*100}%",
            f"Formula: {formula}",
            f"Calculation: ${entry_price:,.2f} × {multiplier}",
            f"Result: ${liq_price:,.2f}"
        ]

        return {
            'liquidation_price': liq_price,
            'formula': formula,
            'calculation_steps': calculation_steps,
            'mmr': mmr
        }

    @staticmethod
    def generate_validation_matrix() -> List[Dict[str, Any]]:
        """
        Generate strategic test scenarios for Binance validation.

        Returns 12 test scenarios covering:
        - Core leverage levels: 1x, 10x, 125x (LONG/SHORT)
        - Extreme prices: $10, $100,000 (LONG/SHORT)
        - Different MMR values: 0.5%, 1.0%

        Returns:
            List of test scenario dictionaries

        Example:
            >>> validator = BinanceCalculatorValidator()
            >>> scenarios = validator.generate_validation_matrix()
            >>> len(scenarios)
            12
        """
        test_scenarios = []

        # Core leverage levels (1x, 10x, 125x) × LONG/SHORT
        standard_price = 40000.0
        leverage_levels = [1.0, 10.0, 125.0]

        for leverage in leverage_levels:
            for direction in ['LONG', 'SHORT']:
                test_scenarios.append({
                    'entry_price': standard_price,
                    'leverage': leverage,
                    'direction': direction,
                    'mmr': BinanceCalculatorValidator.DEFAULT_MMR,
                    'scenario_type': 'core'
                })

        # Extreme price scenarios
        extreme_prices = [10.0, 100000.0]
        standard_leverage = 10.0

        for price in extreme_prices:
            for direction in ['LONG', 'SHORT']:
                test_scenarios.append({
                    'entry_price': price,
                    'leverage': standard_leverage,
                    'direction': direction,
                    'mmr': BinanceCalculatorValidator.DEFAULT_MMR,
                    'scenario_type': 'extreme_price'
                })

        # Note: Different MMR testing removed for now
        # Portfolio currently uses fixed MMR (0.4%)
        # Will be added when per-symbol MMR configuration is implemented

        return test_scenarios

    @staticmethod
    def create_binance_validation_df(test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate DataFrame for manual Binance calculator validation.

        The DataFrame includes columns for:
        - Test inputs (entry price, leverage, direction, MMR)
        - Expected values (from our formula)
        - Actual values (from Portfolio implementation)
        - Binance Calculator values (empty for manual entry)
        - Error calculations
        - Status (PASS/FAIL)

        Args:
            test_cases: List of test case dictionaries with calculated results

        Returns:
            DataFrame ready for Excel export with empty Binance column

        Example:
            >>> validator = BinanceCalculatorValidator()
            >>> scenarios = validator.generate_validation_matrix()
            >>> # Add 'actual' values from Portfolio to scenarios
            >>> df = validator.create_binance_validation_df(scenarios)
            >>> 'Binance Calculator' in df.columns
            True
        """
        rows = []

        for tc in test_cases:
            # Get expected from formula
            expected_result = BinanceCalculatorValidator.get_expected_liquidation_price(
                tc['entry_price'],
                tc['leverage'],
                tc['direction'],
                tc.get('mmr', BinanceCalculatorValidator.DEFAULT_MMR)
            )

            # Get actual from test case (if provided)
            actual_liq_price = tc.get('actual', None)

            # Calculate error vs formula (if actual is provided)
            error_formula = None
            if actual_liq_price is not None:
                error_formula = abs(actual_liq_price - expected_result['liquidation_price']) / expected_result['liquidation_price'] * 100

            rows.append({
                'Scenario Type': tc.get('scenario_type', 'unknown'),
                'Entry Price': tc['entry_price'],
                'Leverage': f"{tc['leverage']}x",
                'Direction': tc['direction'],
                'MMR': f"{tc.get('mmr', BinanceCalculatorValidator.DEFAULT_MMR)*100}%",
                'Expected (Formula)': expected_result['liquidation_price'],
                'Our Calculation': actual_liq_price,
                'Binance Calculator': None,  # Manual entry column
                'Error vs Formula %': error_formula,
                'Error vs Binance %': None,  # Calculated after manual entry
                'Status': None  # Determined after manual validation
            })

        return pd.DataFrame(rows)

    @staticmethod
    def validate_against_binance(
        df: pd.DataFrame,
        tolerance: float = 0.1
    ) -> pd.DataFrame:
        """
        Validate our calculations against manually-entered Binance values.

        Args:
            df: DataFrame with 'Our Calculation' and 'Binance Calculator' columns filled
            tolerance: Error tolerance percentage (default 0.1% = acceptance criteria)

        Returns:
            Updated DataFrame with error calculations and PASS/FAIL status

        Example:
            >>> # After user manually enters Binance values in Excel
            >>> df = pd.read_excel('liquidation_uat.xlsx', sheet_name='Binance Validation')
            >>> validated_df = BinanceCalculatorValidator.validate_against_binance(df)
        """
        # Calculate error vs Binance (if Binance values are provided)
        def calc_error(row):
            if pd.isna(row['Binance Calculator']) or pd.isna(row['Our Calculation']):
                return None
            binance_val = float(row['Binance Calculator'])
            our_val = float(row['Our Calculation'])
            return abs(our_val - binance_val) / binance_val * 100

        def determine_status(row):
            if pd.isna(row['Error vs Binance %']):
                return 'PENDING'
            return 'PASS' if row['Error vs Binance %'] < tolerance else 'FAIL'

        df['Error vs Binance %'] = df.apply(calc_error, axis=1)
        df['Status'] = df.apply(determine_status, axis=1)

        return df

    @staticmethod
    def format_calculation_steps(steps: List[str]) -> str:
        """
        Format calculation steps for Excel display.

        Args:
            steps: List of calculation step strings

        Returns:
            Formatted string with newlines for Excel cell display
        """
        return '\n'.join(steps)
