#!/usr/bin/env python3
"""
Reference margin calculator for UAT validation.

This module provides standalone reference implementations of margin calculations
that EXACTLY match the production Portfolio code. These are used as the source
of truth for validating actual Portfolio behavior.

CRITICAL: All formulas must match Portfolio.py production implementation exactly.
"""

from typing import Literal


class MarginCalculator:
    """
    Reference implementation of margin calculations matching Portfolio production code.

    All methods here replicate the exact formulas from meluna/portfolio.py to serve
    as independent validation references.
    """

    # Constants matching production (Portfolio.py)
    DEFAULT_MMR = 0.004  # 0.4% maintenance margin rate (line 148)
    MAX_LEVERAGE = 125.0  # Maximum leverage (line 94)

    @staticmethod
    def calculate_margin(quantity: int, price: float, leverage: float) -> float:
        """
        Calculate initial margin required for a position.

        Matches: Portfolio._calculate_required_margin() (lines 206-226)

        Formula: initial_margin = (quantity * price) / leverage

        Args:
            quantity: Position size (absolute value)
            price: Entry price
            leverage: Leverage multiplier (1.0 - 125.0)

        Returns:
            Required initial margin

        Examples:
            >>> MarginCalculator.calculate_margin(1, 40000.0, 10.0)
            4000.0
            >>> MarginCalculator.calculate_margin(10, 42000.0, 1.0)
            420000.0
        """
        position_value = abs(quantity) * price
        return position_value / leverage

    @staticmethod
    def calculate_pnl_long(entry_price: float, exit_price: float, quantity: int) -> float:
        """
        Calculate P&L for LONG position.

        Matches: Portfolio._close_position_futures() (lines 1014-1015)

        Formula: pnl = (exit_price - entry_price) * quantity

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity

        Returns:
            Profit/Loss (positive = profit, negative = loss)

        Examples:
            >>> MarginCalculator.calculate_pnl_long(40000.0, 45000.0, 1)
            5000.0
            >>> MarginCalculator.calculate_pnl_long(40000.0, 38000.0, 1)
            -2000.0
        """
        return (exit_price - entry_price) * quantity

    @staticmethod
    def calculate_pnl_short(entry_price: float, exit_price: float, quantity: int) -> float:
        """
        Calculate P&L for SHORT position.

        Matches: Portfolio._close_position_futures() (line 1017)

        Formula: pnl = (entry_price - exit_price) * quantity

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity

        Returns:
            Profit/Loss (positive = profit, negative = loss)

        Examples:
            >>> MarginCalculator.calculate_pnl_short(40000.0, 38000.0, 1)
            2000.0
            >>> MarginCalculator.calculate_pnl_short(40000.0, 45000.0, 1)
            -5000.0
        """
        return (entry_price - exit_price) * quantity

    @staticmethod
    def calculate_liquidation_price_long(
        entry_price: float,
        leverage: float,
        mmr: float = DEFAULT_MMR
    ) -> float:
        """
        Calculate liquidation price for LONG position.

        Matches: Portfolio._calculate_liquidation_price() (lines 275-277)

        Formula: liq_price = entry_price * (1 - 1/leverage + MMR)

        Args:
            entry_price: Entry price
            leverage: Leverage multiplier
            mmr: Maintenance margin rate (default: 0.004 = 0.4%)

        Returns:
            Liquidation price

        Examples:
            >>> MarginCalculator.calculate_liquidation_price_long(40000.0, 10.0)
            36160.0
            >>> MarginCalculator.calculate_liquidation_price_long(40000.0, 1.0)
            160.0
        """
        return entry_price * (1 - 1/leverage + mmr)

    @staticmethod
    def calculate_liquidation_price_short(
        entry_price: float,
        leverage: float,
        mmr: float = DEFAULT_MMR
    ) -> float:
        """
        Calculate liquidation price for SHORT position.

        Matches: Portfolio._calculate_liquidation_price() (lines 279-280)

        Formula: liq_price = entry_price * (1 + 1/leverage - MMR)

        Args:
            entry_price: Entry price
            leverage: Leverage multiplier
            mmr: Maintenance margin rate (default: 0.004 = 0.4%)

        Returns:
            Liquidation price

        Examples:
            >>> MarginCalculator.calculate_liquidation_price_short(40000.0, 10.0)
            43840.0
            >>> MarginCalculator.calculate_liquidation_price_short(40000.0, 1.0)
            79840.0
        """
        return entry_price * (1 + 1/leverage - mmr)

    @staticmethod
    def calculate_expected_state_after_open(
        initial_equity: float,
        quantity: int,
        price: float,
        leverage: float,
        commission: float
    ) -> dict:
        """
        Calculate expected portfolio state after opening a position.

        This consolidates the margin accounting logic from Portfolio._open_position_futures()
        (lines 909-984) for easy validation.

        Args:
            initial_equity: Starting total equity
            quantity: Position quantity
            price: Entry price
            leverage: Leverage multiplier
            commission: Commission charged

        Returns:
            Dictionary with expected state:
            {
                'used_margin': float,
                'available_margin': float,
                'total_equity': float,
                'required_margin': float
            }

        Examples:
            >>> state = MarginCalculator.calculate_expected_state_after_open(
            ...     100000.0, 1, 40000.0, 10.0, 40.0
            ... )
            >>> state['used_margin']
            4000.0
            >>> state['available_margin']
            95960.0
            >>> state['total_equity']
            99960.0
        """
        required_margin = MarginCalculator.calculate_margin(quantity, price, leverage)

        # Margin accounting (matches lines 940-945)
        used_margin = required_margin
        total_equity = initial_equity - commission
        available_margin = total_equity - used_margin

        return {
            'used_margin': used_margin,
            'available_margin': available_margin,
            'total_equity': total_equity,
            'required_margin': required_margin
        }

    @staticmethod
    def calculate_expected_state_after_close(
        current_equity: float,
        current_used_margin: float,
        current_available_margin: float,
        position_margin: float,
        pnl: float,
        commission: float
    ) -> dict:
        """
        Calculate expected portfolio state after closing a position.

        This consolidates the margin accounting logic from Portfolio._close_position_futures()
        (lines 1019-1029) for easy validation.

        Args:
            current_equity: Current total equity
            current_used_margin: Current used margin
            current_available_margin: Current available margin
            position_margin: Margin to return from closed position
            pnl: Profit/Loss from position
            commission: Commission charged on close

        Returns:
            Dictionary with expected state:
            {
                'used_margin': float,
                'available_margin': float,
                'total_equity': float
            }

        Examples:
            >>> state = MarginCalculator.calculate_expected_state_after_close(
            ...     99960.0, 4000.0, 95960.0, 4000.0, 5000.0, 45.0
            ... )
            >>> state['used_margin']
            0.0
            >>> state['available_margin']
            104915.0
            >>> state['total_equity']
            104915.0
        """
        # Return margin (line 1020-1021)
        used_margin = current_used_margin - position_margin
        available_margin = current_available_margin + position_margin

        # Apply P&L (lines 1024-1025)
        total_equity = current_equity + pnl
        available_margin = available_margin + pnl

        # Deduct commission (lines 1028-1029)
        total_equity = total_equity - commission
        available_margin = available_margin - commission

        return {
            'used_margin': used_margin,
            'available_margin': available_margin,
            'total_equity': total_equity
        }

    @staticmethod
    def validate_margin_equation(
        used_margin: float,
        available_margin: float,
        total_equity: float,
        tolerance: float = 0.01
    ) -> bool:
        """
        Validate the critical margin equation.

        Matches: Portfolio._validate_margin_equation() (lines 503-532)

        Equation: used_margin + available_margin = total_equity (±tolerance)

        Args:
            used_margin: Used margin
            available_margin: Available margin
            total_equity: Total equity
            tolerance: Allowed difference (default: $0.01)

        Returns:
            True if equation holds, False otherwise

        Examples:
            >>> MarginCalculator.validate_margin_equation(4000.0, 95960.0, 99960.0)
            True
            >>> MarginCalculator.validate_margin_equation(4000.0, 95960.0, 100000.0)
            False
        """
        calculated_total = used_margin + available_margin
        return abs(calculated_total - total_equity) <= tolerance
