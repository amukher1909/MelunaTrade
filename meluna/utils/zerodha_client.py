# meluna/utils/zerodha_client.py

import logging
import pandas as pd
import os
from pathlib import Path
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

class ZerodhaClient:
    """
    (Robust Version) A wrapper for the Zerodha Kite Connect API that
    efficiently downloads and caches the instrument list for fast lookups.
    """
    def __init__(self, credentials: dict, instrument_cache_path: str):
        self.api_key = credentials.get('api_key')
        self.access_token = credentials.get('access_token')
        self.instrument_cache_path = instrument_cache_path
        self.instruments_df = None  # To hold the cached instruments

        try:
            self.kite = KiteConnect(api_key=self.api_key)
            if self.access_token:
                self.kite.set_access_token(self.access_token)
            logger.info("✅ ZerodhaClient initialized.")
            
            # Load instruments into memory on startup
            self._load_or_fetch_instruments()

        except Exception as e:
            logger.exception(f"Failed to initialize KiteConnect client: {e}")
            raise

    def _load_or_fetch_instruments(self):
        """
        Loads the instrument list from a local cache if it exists,
        otherwise fetches it from the API and creates the cache.
        """
        cache_path = Path(self.instrument_cache_path)
        try:
            if cache_path.exists():
                logger.info(f"Loading instruments from cache: {cache_path}")
                self.instruments_df = pd.read_csv(cache_path)
            else:
                logger.info("Instrument cache not found. Fetching from Zerodha API...")
                instrument_dump = self.kite.instruments()
                self.instruments_df = pd.DataFrame(instrument_dump)
                
                # Create the directory if it doesn't exist
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.instruments_df.to_csv(cache_path, index=False)
                logger.info(f"Successfully cached instruments to {cache_path}")

        except Exception as e:
            logger.exception(f"Failed to load or fetch instrument list: {e}")
            self.instruments_df = pd.DataFrame() # Ensure it's an empty df on failure

    def get_instrument_token(self, symbol_with_exchange: str) -> int:
        """
        (Definitive Version) Performs a robust, local lookup for the instrument token.
        It correctly handles the 'EXCHANGE:SYMBOL-SERIES' format by parsing the
        exchange and base symbol before searching the cached instrument list.
        """
        if self.instruments_df.empty:
            logger.error("Instrument list is not loaded. Cannot find token.")
            return None

        try:
            # --- 1. Parse the input string ---
            # Example: "NSE:RELIANCE-EQ" -> exchange="NSE", trading_symbol="RELIANCE-EQ"
            parts = symbol_with_exchange.split(':')
            if len(parts) != 2:
                logger.error(f"Invalid symbol format: '{symbol_with_exchange}'. Expected 'EXCHANGE:SYMBOL'.")
                return None
            exchange, trading_symbol = parts

            # --- 2. First, try for an exact match ---
            # This handles symbols that might not have a series suffix (e.g., NIFTY 50)
            match = self.instruments_df[
                (self.instruments_df['tradingsymbol'] == trading_symbol) &
                (self.instruments_df['exchange'] == exchange)
            ]

            # --- 3. If no exact match, strip the suffix and try the base symbol ---
            # Example: "RELIANCE-EQ" -> base_symbol="RELIANCE"
            if match.empty and trading_symbol.endswith('-EQ'):
                base_symbol = trading_symbol.rsplit('-', 1)[0]
                match = self.instruments_df[
                    (self.instruments_df['tradingsymbol'] == base_symbol) &
                    (self.instruments_df['exchange'] == exchange)
                ]

            if not match.empty:
                token = int(match.iloc[0]['instrument_token'])
                logger.info(f"Successfully found instrument token {token} for {symbol_with_exchange}.")
                return token

        except Exception as e:
            logger.exception(f"An error occurred while finding token for {symbol_with_exchange}: {e}")

        logger.error(f"Could not find any valid instrument token for {symbol_with_exchange}.")
        return None

    def historical_data(self, instrument_token: int, from_date, to_date, interval: str):
        """
        Fetch historical data with improved error handling and validation.
        
        Args:
            instrument_token: Zerodha instrument token
            from_date: Start date (datetime object)
            to_date: End date (datetime object)
            interval: Data interval ('day', 'minute', etc.)
            
        Returns:
            List of OHLCV dictionaries or empty list on failure
        """
        try:
            logger.info(f"ZerodhaClient: Requesting {interval} data for token {instrument_token} "
                       f"from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
            
            # Validate access token before making API call
            if not self.access_token:
                logger.error("No access token available for Zerodha API")
                return []
            
            response = self.kite.historical_data(instrument_token, from_date, to_date, interval)
            
            if response:
                logger.info(f"Successfully fetched {len(response)} data points for token {instrument_token}")
                return response
            else:
                logger.warning(f"No data returned for token {instrument_token}")
                return []
                
        except Exception as e:
            # Log detailed error information
            error_msg = str(e)
            if "Incorrect `api_key` or `access_token`" in error_msg:
                logger.error(f"Authentication failed for token {instrument_token}. "
                           f"Please verify API credentials and access token validity.")
            elif "rate limit" in error_msg.lower():
                logger.error(f"Rate limit exceeded for token {instrument_token}. Consider adding delays.")
            else:
                logger.error(f"API error for token {instrument_token}: {error_msg}")
            
            return []