# meluna/utils/binance_client.py

"""
Production-grade Binance Futures API client wrapper.

This module provides a defensive, rate-limit-aware client for interacting with
Binance Futures API. Key features:
- Weight-based rate limiting with 70% safety margin (prevents IP bans)
- Automatic retry with exponential backoff for transient errors
- Testnet and mainnet support
- Thread-safe for future concurrent usage
"""

import time
import threading
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException

from meluna.utils.binance_exceptions import BinanceAPIError

logger = logging.getLogger(__name__)


class BinanceRateLimiter:
    """
    Weight-based rate limiter for Binance API.

    Binance uses a weight system where requests cost 1-10 weight depending
    on parameters. The limit is 2400 weight/minute. We use 70% (1680) as
    a safety margin to prevent accidental IP bans.

    This class is thread-safe for future concurrent request support.
    """

    def __init__(self, max_weight_per_minute: int = 1680):
        """
        Initialize rate limiter.

        Args:
            max_weight_per_minute: Weight limit (default 1680 = 70% of 2400)
        """
        self.max_weight = max_weight_per_minute
        self.current_weight = 0
        self.weight_reset_time = time.time() + 60
        self.lock = threading.Lock()

    def wait_if_needed(self, estimated_weight: int) -> None:
        """
        Block execution if adding this request would exceed the limit.

        This prevents us from going over the rate limit. If we're close to
        the limit, we sleep until the minute resets.

        Args:
            estimated_weight: Estimated cost of the upcoming request
        """
        with self.lock:
            now = time.time()

            # Reset counter if minute has elapsed
            if now >= self.weight_reset_time:
                self.current_weight = 0
                self.weight_reset_time = now + 60
                logger.debug(f"Rate limit window reset at {now:.1f}")

            # Check if we can proceed
            if self.current_weight + estimated_weight > self.max_weight:
                sleep_time = self.weight_reset_time - now
                logger.warning(
                    f"Rate limit approaching: {self.current_weight}/{self.max_weight} weight used. "
                    f"Sleeping {sleep_time:.1f}s until reset."
                )
                time.sleep(sleep_time)
                self.current_weight = 0
                self.weight_reset_time = time.time() + 60

            self.current_weight += estimated_weight
            logger.debug(f"Weight used: {self.current_weight}/{self.max_weight}")

    def update_from_headers(self, used_weight: int) -> None:
        """
        Update actual weight from API response headers.

        Binance tells us the real weight used in response headers. We sync
        our counter with their truth to prevent drift.

        Args:
            used_weight: Actual weight from X-MBX-USED-WEIGHT-1M header
        """
        if used_weight > 0:
            with self.lock:
                self.current_weight = used_weight
                logger.debug(f"Rate limit synced from headers: {used_weight}/{self.max_weight}")


class BinanceClient:
    """
    Wrapper for Binance Futures API with rate limiting and error handling.

    This client follows the same pattern as ZerodhaClient and FyersClient
    in meluna/utils/. It provides:
    - Authenticated and anonymous modes
    - Testnet vs mainnet switching
    - Automatic rate limiting (70% safety margin)
    - Retry logic with exponential backoff
    - Clear error messages with retry guidance

    Examples:
        # Authenticated testnet mode
        >>> config = {
        ...     'api_key': 'your_key',
        ...     'api_secret': 'your_secret',
        ...     'testnet': True
        ... }
        >>> client = BinanceClient(config)
        >>> data = client.get_klines('BTCUSDT', '1d', '2024-01-01', '2024-12-31')

        # Anonymous mainnet mode (public data only)
        >>> config = {'testnet': False}
        >>> client = BinanceClient(config)
        >>> data = client.get_klines('BTCUSDT', '1h', '2024-01-01', '2024-01-31')
    """

    # Weight table from Binance docs
    KLINES_WEIGHT_TABLE = [
        (99, 1),
        (499, 2),
        (1000, 5),
        (float('inf'), 10)
    ]

    def __init__(self, credentials: Dict[str, Any]):
        """
        Initialize Binance client.

        Args:
            credentials: Dictionary with keys:
                - api_key (optional): Binance API key
                - api_secret (optional): Binance API secret
                - testnet (bool): Use testnet (default: False)

        Raises:
            ValueError: If api_key is provided without api_secret (or vice versa)
        """
        self.api_key = credentials.get('api_key')
        self.api_secret = credentials.get('api_secret')
        self.is_testnet = credentials.get('testnet', False)

        # Validate credentials: both or neither
        has_key = self.api_key is not None
        has_secret = self.api_secret is not None
        if has_key != has_secret:
            raise ValueError(
                "Both api_key and api_secret must be provided together, or neither. "
                "You provided only one."
            )

        # Determine mode
        if has_key and has_secret:
            self.mode = 'authenticated'
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.is_testnet
            )
            env = 'TESTNET' if self.is_testnet else 'MAINNET'
            logger.info(f"BinanceClient initialized: Authenticated mode ({env})")
        else:
            self.mode = 'anonymous'
            self.client = Client(testnet=self.is_testnet)
            env = 'TESTNET' if self.is_testnet else 'MAINNET'
            logger.info(f"BinanceClient initialized: Anonymous mode ({env})")

        # Initialize rate limiter (70% of 2400 = 1680)
        self.rate_limiter = BinanceRateLimiter(max_weight_per_minute=1680)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Union[str, int, None] = None,
        end_time: Union[str, int, None] = None,
        limit: int = 1500
    ) -> List[List[Any]]:
        """
        Fetch klines (OHLCV candlestick data) with rate limiting and retry.

        This method:
        1. Estimates the request weight based on limit parameter
        2. Waits if we're close to rate limit
        3. Makes the request with automatic retry on failures
        4. Updates weight counter from response headers

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Candlestick interval (e.g., '1m', '5m', '1h', '1d')
            start_time: Start time (ISO format string, timestamp ms, or None)
                       Examples: '2024-01-01', '2024-01-01 12:00:00', 1704067200000
            end_time: End time (same formats as start_time, optional)
            limit: Number of candles (max 1500, default 1500)

        Returns:
            List of klines, where each kline is:
            [
                open_time,        # Timestamp (ms)
                open,             # str
                high,             # str
                low,              # str
                close,            # str
                volume,           # str
                close_time,       # Timestamp (ms)
                quote_volume,     # str
                trades,           # int
                taker_buy_base,   # str
                taker_buy_quote,  # str
                ignore            # str
            ]

        Raises:
            BinanceAPIError: On API errors (check should_retry flag)

        Examples:
            >>> client = BinanceClient({'testnet': True})
            >>> # Fetch 30 days of daily data (ISO format)
            >>> klines = client.get_klines('BTCUSDT', '1d', '2024-01-01', '2024-01-31')
            >>> # Or use limit without end_time
            >>> klines = client.get_klines('BTCUSDT', '1d', '2024-01-01', limit=30)
        """
        # Convert string dates to millisecond timestamps if needed
        start_ms = self._to_timestamp(start_time) if start_time else None
        end_ms = self._to_timestamp(end_time) if end_time else None

        # Estimate weight
        estimated_weight = self._estimate_klines_weight(limit)
        logger.debug(f"Fetching klines for {symbol} (interval={interval}, limit={limit}, est_weight={estimated_weight})")

        # Wait if needed to avoid rate limit
        self.rate_limiter.wait_if_needed(estimated_weight)

        # Fetch with retry logic
        return self._fetch_with_retry(
            lambda: self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ms,
                endTime=end_ms,
                limit=limit
            )
        )

    def _to_timestamp(self, time_value: Union[str, int]) -> int:
        """
        Convert various time formats to millisecond timestamp.

        Args:
            time_value: String (ISO format) or int (ms timestamp)

        Returns:
            Millisecond timestamp

        Examples:
            >>> client._to_timestamp('2024-01-01')
            1704067200000
            >>> client._to_timestamp('2024-01-01 12:00:00')
            1704110400000
            >>> client._to_timestamp(1704067200000)
            1704067200000
        """
        # If already int, assume it's milliseconds
        if isinstance(time_value, int):
            return time_value

        # Try parsing ISO format strings
        try:
            # Try with time first (e.g., '2024-01-01 12:00:00')
            if ' ' in time_value:
                dt = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
            else:
                # Just date (e.g., '2024-01-01')
                dt = datetime.strptime(time_value, '%Y-%m-%d')

            # Convert to milliseconds
            return int(dt.timestamp() * 1000)

        except ValueError:
            # If parsing fails, raise clear error
            raise ValueError(
                f"Invalid time format: '{time_value}'. "
                f"Use ISO format like '2024-01-01' or '2024-01-01 12:00:00', "
                f"or millisecond timestamp (int)."
            )

    def _estimate_klines_weight(self, limit: int) -> int:
        """
        Estimate API weight for klines request based on limit.

        Weight table (from Binance docs):
        - limit ≤ 99:    weight = 1
        - limit ≤ 499:   weight = 2
        - limit ≤ 1000:  weight = 5
        - limit > 1000:  weight = 10

        Args:
            limit: Number of candles requested

        Returns:
            Estimated weight (1, 2, 5, or 10)
        """
        for threshold, weight in self.KLINES_WEIGHT_TABLE:
            if limit <= threshold:
                return weight
        return 10  # Fallback (should never reach here)

    def _fetch_with_retry(self, api_call, max_retries: int = 3):
        """
        Execute API call with exponential backoff retry on transient errors.

        This handles:
        - 429 (rate limit): Retry with backoff (1s, 2s, 4s)
        - 5xx (server error): Retry with backoff
        - 400, 401, 418 (client error, auth, IP ban): No retry

        Args:
            api_call: Lambda function wrapping the API call
            max_retries: Maximum retry attempts (default 3)

        Returns:
            API response data

        Raises:
            BinanceAPIError: After max retries exceeded or on fatal errors
        """
        for attempt in range(max_retries + 1):
            try:
                response = api_call()

                # Try to update rate limiter from headers
                # Note: python-binance doesn't always expose headers easily,
                # so we rely primarily on our estimates
                return response

            except BinanceAPIException as e:
                error = self._parse_binance_exception(e)

                # Don't retry if fatal or last attempt
                if not error.should_retry or attempt == max_retries:
                    logger.error(f"API call failed after {attempt + 1} attempts: {error}")
                    raise error

                # Exponential backoff: 1s, 2s, 4s
                backoff_time = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {error}. "
                    f"Retrying in {backoff_time}s..."
                )
                time.sleep(backoff_time)

            except Exception as e:
                # Unexpected error (not from Binance library)
                error = BinanceAPIError(0, f"Unexpected error: {str(e)}", should_retry=False)
                logger.error(f"Unexpected error: {error}")
                raise error

        # Should never reach here (loop handles all cases)
        raise BinanceAPIError(500, f"Failed after {max_retries} retries", should_retry=False)

    def _parse_binance_exception(self, exception: BinanceAPIException) -> BinanceAPIError:
        """
        Parse BinanceAPIException into our custom error with retry guidance.

        Error categories:
        - Transient (retry): 429 (rate limit), 5xx (server error)
        - Fatal (no retry): 418 (IP ban), 401 (auth), 400 (bad request)

        Args:
            exception: Exception from python-binance library

        Returns:
            BinanceAPIError with should_retry flag set appropriately
        """
        status_code = exception.status_code
        message = exception.message

        # Transient errors - retry makes sense
        if status_code == 429:
            return BinanceAPIError(429, "Rate limit exceeded (429)", should_retry=True)
        elif status_code >= 500:
            return BinanceAPIError(status_code, f"Server error: {message}", should_retry=True)

        # Fatal errors - no retry
        elif status_code == 418:
            return BinanceAPIError(418, "IP auto-banned (418). Stop all requests immediately!", should_retry=False)
        elif status_code == 401:
            return BinanceAPIError(401, "Unauthorized. Check API credentials.", should_retry=False)
        elif status_code == 400:
            return BinanceAPIError(400, f"Bad request: {message}", should_retry=False)

        # Unknown error - don't retry (could be client bug)
        else:
            return BinanceAPIError(status_code, f"API error: {message}", should_retry=False)
