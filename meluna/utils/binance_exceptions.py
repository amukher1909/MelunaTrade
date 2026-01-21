# meluna/utils/binance_exceptions.py

"""
Custom exceptions for Binance API client.
Provides clear error categorization and retry guidance.
"""

class BinanceAPIError(Exception):
    """
    Custom exception for Binance API errors.

    Attributes:
        status_code: HTTP status code (e.g., 429, 418, 500)
        message: Human-readable error description
        should_retry: Whether this error is transient (retry makes sense)

    Examples:
        >>> try:
        ...     client.get_klines(...)
        ... except BinanceAPIError as e:
        ...     if e.should_retry:
        ...         print("Temporary issue, try again later")
        ...     else:
        ...         print("Fatal error, check your code/credentials")
    """

    def __init__(self, status_code: int, message: str, should_retry: bool = False):
        """
        Initialize Binance API error.

        Args:
            status_code: HTTP status code from API response
            message: Error description
            should_retry: True for transient errors (429, 5xx), False for fatal (400, 401, 418)
        """
        self.status_code = status_code
        self.message = message
        self.should_retry = should_retry
        super().__init__(f"[{status_code}] {message}")
