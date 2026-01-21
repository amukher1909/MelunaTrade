# meluna/utils/binance_symbol_parser.py

"""
Parse and validate Binance futures symbols.

This module provides symbol parsing for Binance futures contracts to support:
- Cache file organization (different contracts need separate files)
- Contract rollover detection (quarterly expiry dates)
- Symbol validation before API calls

Examples:
    >>> parser = BinanceSymbolParser()
    >>> parser.parse('BTCUSDT')
    {'type': 'perpetual', 'base': 'BTC', 'quote': 'USDT', 'base_pair': 'BTCUSDT'}

    >>> parser.parse('BTCUSDT_250328')
    {'type': 'quarterly', 'base': 'BTC', 'quote': 'USDT', 'base_pair': 'BTCUSDT',
     'expiry': date(2025, 3, 28), 'expiry_suffix': '250328'}
"""

import re
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BinanceSymbolParser:
    """
    Parse and validate Binance futures symbols.

    Supports:
    - Perpetual USDT-margined (e.g., BTCUSDT)
    - Quarterly USDT-margined (e.g., BTCUSDT_250328)
    - Perpetual coin-margined (e.g., BTCUSD_PERP)
    - Quarterly coin-margined (e.g., BTCUSD_0328)

    Note: Binance coin-margined quarterly uses MMDD format (e.g., 0328 = March 28),
    not MMYY. The exact expiry year is inferred (current or next year).
    """

    # Regex patterns for symbol matching
    PERPETUAL_USDT_PATTERN = re.compile(r'^([A-Z]+)(USDT)$')
    QUARTERLY_USDT_PATTERN = re.compile(r'^([A-Z]+)(USDT)_(\d{6})$')
    PERPETUAL_COIN_PATTERN = re.compile(r'^([A-Z]+)(USD)_PERP$')
    QUARTERLY_COIN_PATTERN = re.compile(r'^([A-Z]+)(USD)_(\d{4})$')

    def parse(self, symbol: str, validate_expiry: bool = True) -> Dict[str, Any]:
        """
        Parse symbol into structured metadata.

        Args:
            symbol: Binance futures symbol
            validate_expiry: If True, raises error for expired contracts (default: True)

        Returns:
            Dictionary with keys:
                - type: 'perpetual', 'quarterly', 'perpetual_coin', 'quarterly_coin'
                - base: Base asset (e.g., 'BTC')
                - quote: Quote asset (e.g., 'USDT')
                - base_pair: Full pair (e.g., 'BTCUSDT')
                - expiry: datetime.date (only for quarterly)
                - expiry_suffix: Raw suffix (only for quarterly)

        Raises:
            ValueError: If symbol format is invalid or contract expired

        Examples:
            >>> parser = BinanceSymbolParser()
            >>> parser.parse('BTCUSDT')
            {'type': 'perpetual', 'base': 'BTC', 'quote': 'USDT', 'base_pair': 'BTCUSDT'}

            >>> parser.parse('BTCUSDT_250328')
            {'type': 'quarterly', 'base': 'BTC', 'quote': 'USDT', 'base_pair': 'BTCUSDT',
             'expiry': date(2025, 3, 28), 'expiry_suffix': '250328'}
        """
        # Try perpetual USDT
        match = self.PERPETUAL_USDT_PATTERN.match(symbol)
        if match:
            base, quote = match.groups()
            return {
                'type': 'perpetual',
                'base': base,
                'quote': quote,
                'base_pair': symbol
            }

        # Try quarterly USDT
        match = self.QUARTERLY_USDT_PATTERN.match(symbol)
        if match:
            base, quote, expiry_suffix = match.groups()
            expiry_date = self._parse_usdt_expiry(expiry_suffix)

            # Validate not expired (if requested)
            if validate_expiry and expiry_date < date.today():
                raise ValueError(f"Contract {symbol} expired on {expiry_date}")

            # Warn if near expiry
            if validate_expiry:
                days_until_expiry = (expiry_date - date.today()).days
                if days_until_expiry < 7:
                    logger.warning(f"Contract {symbol} expires in {days_until_expiry} days ({expiry_date})")

            return {
                'type': 'quarterly',
                'base': base,
                'quote': quote,
                'base_pair': f"{base}{quote}",
                'expiry': expiry_date,
                'expiry_suffix': expiry_suffix
            }

        # Try perpetual coin
        match = self.PERPETUAL_COIN_PATTERN.match(symbol)
        if match:
            base, quote = match.groups()
            return {
                'type': 'perpetual_coin',
                'base': base,
                'quote': quote,
                'base_pair': f"{base}{quote}"
            }

        # Try quarterly coin
        match = self.QUARTERLY_COIN_PATTERN.match(symbol)
        if match:
            base, quote, expiry_suffix = match.groups()
            expiry_date = self._parse_coin_expiry(expiry_suffix)

            # Validate not expired (if requested)
            if validate_expiry and expiry_date < date.today():
                raise ValueError(f"Contract {symbol} expired on {expiry_date}")

            # Warn if near expiry
            if validate_expiry:
                days_until_expiry = (expiry_date - date.today()).days
                if days_until_expiry < 7:
                    logger.warning(f"Contract {symbol} expires in {days_until_expiry} days ({expiry_date})")

            return {
                'type': 'quarterly_coin',
                'base': base,
                'quote': quote,
                'base_pair': f"{base}{quote}",
                'expiry': expiry_date,
                'expiry_suffix': expiry_suffix
            }

        # No pattern matched
        raise ValueError(
            f"Invalid Binance futures symbol: {symbol}. "
            f"Expected formats: BTCUSDT, BTCUSDT_250328, BTCUSD_PERP, BTCUSD_0328"
        )

    def _parse_usdt_expiry(self, expiry_suffix: str) -> date:
        """
        Parse USDT quarterly expiry suffix (YYMMDD format).

        Args:
            expiry_suffix: 6-digit string (e.g., '250328')

        Returns:
            Expiry date

        Raises:
            ValueError: If format is invalid

        Examples:
            >>> parser._parse_usdt_expiry('250328')
            date(2025, 3, 28)
        """
        try:
            year = 2000 + int(expiry_suffix[:2])
            month = int(expiry_suffix[2:4])
            day = int(expiry_suffix[4:6])
            return date(year, month, day)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid USDT quarterly expiry format: {expiry_suffix}") from e

    def _parse_coin_expiry(self, expiry_suffix: str) -> date:
        """
        Parse coin-margined quarterly expiry suffix (MMDD format).

        Binance coin-margined quarterlies use MMDD format (e.g., 0328 = March 28).
        The year is inferred: if the date is in the past, assume next year.

        Args:
            expiry_suffix: 4-digit string (e.g., '0328')

        Returns:
            Expiry date

        Raises:
            ValueError: If format is invalid

        Examples:
            >>> parser._parse_coin_expiry('0328')  # Assumes current/next year
            date(2025, 3, 28)
        """
        try:
            month = int(expiry_suffix[:2])
            day = int(expiry_suffix[2:4])

            # Start with current year
            current_year = date.today().year
            expiry = date(current_year, month, day)

            # If date is in the past, assume next year
            if expiry < date.today():
                expiry = date(current_year + 1, month, day)

            return expiry

        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid coin quarterly expiry format: {expiry_suffix}") from e

    def is_perpetual(self, symbol: str) -> bool:
        """
        Check if symbol is a perpetual contract.

        Args:
            symbol: Binance futures symbol

        Returns:
            True if perpetual, False otherwise

        Examples:
            >>> parser.is_perpetual('BTCUSDT')
            True
            >>> parser.is_perpetual('BTCUSDT_250328')
            False
        """
        try:
            info = self.parse(symbol, validate_expiry=False)
            return info['type'] in ['perpetual', 'perpetual_coin']
        except ValueError:
            return False

    def is_quarterly(self, symbol: str) -> bool:
        """
        Check if symbol is a quarterly contract.

        Args:
            symbol: Binance futures symbol

        Returns:
            True if quarterly, False otherwise

        Examples:
            >>> parser.is_quarterly('BTCUSDT_250328')
            True
            >>> parser.is_quarterly('BTCUSDT')
            False
        """
        try:
            info = self.parse(symbol, validate_expiry=False)
            return info['type'] in ['quarterly', 'quarterly_coin']
        except ValueError:
            return False

    def get_expiry_date(self, symbol: str) -> Optional[date]:
        """
        Get expiry date for quarterly contracts.

        Args:
            symbol: Binance futures symbol

        Returns:
            Expiry date or None for perpetual contracts

        Examples:
            >>> parser.get_expiry_date('BTCUSDT')
            None

            >>> parser.get_expiry_date('BTCUSDT_250328')
            date(2025, 3, 28)
        """
        try:
            info = self.parse(symbol, validate_expiry=False)
            return info.get('expiry')
        except ValueError:
            return None

    def to_cache_filename(self, symbol: str, interval: str, env: str = 'mainnet') -> str:
        """
        Generate standardized cache filename.

        Args:
            symbol: Binance futures symbol
            interval: Timeframe (e.g., '1d', '1h')
            env: 'mainnet' or 'testnet'

        Returns:
            Cache filename

        Examples:
            >>> parser.to_cache_filename('BTCUSDT', '1d', 'mainnet')
            'BTCUSDT-binance-mainnet-1d.parquet'

            >>> parser.to_cache_filename('BTCUSDT_250328', '1h', 'mainnet')
            'BTCUSDT_250328-binance-mainnet-1h.parquet'
        """
        return f"{symbol}-binance-{env}-{interval}.parquet"
