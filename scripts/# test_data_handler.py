# test_data_handler.py
import yaml
from pathlib import Path
from meluna.core import ConfigLoader
from data.fyers_handler import FyersDataHandler

def run_simple_test():
    """
    A minimal test to isolate and debug the FyersDataHandler.
    """
    print("--- Running Minimal DataHandler Isolation Test ---")
    try:
        # 1. Initialize Config
        config_path = 'c'
        config_loader = ConfigLoader(config_path=str(config_path))
        print("ConfigLoader initialized.")

        # 2. Initialize DataHandler
        data_handler = FyersDataHandler(config_loader=config_loader)
        print("FyersDataHandler initialized.")

        # 3. Fetch data for ONE symbol from the config
        symbol_to_test = config_loader.get('data_settings')['symbol_list'][0]
        start_date = config_loader.get('backtest_settings')['start_date']
        end_date = config_loader.get('backtest_settings')['end_date']

        print(f"Attempting to fetch and cache data for '{symbol_to_test}'...")
        data_handler.fetch_and_cache_data(
            symbol=symbol_to_test,
            start_date=start_date,
            end_date=end_date
        )

        # 4. Inspect the cache directly
        print("\n--- Cache Inspection ---")
        if symbol_to_test in data_handler.historical_data and not data_handler.historical_data[symbol_to_test].empty:
            print(f"✅ SUCCESS: Symbol '{symbol_to_test}' found in cache with data.")
            df = data_handler.historical_data[symbol_to_test]
            print(f"Cached DataFrame has {len(df)} rows.")
            print("DataFrame Head:")
            print(df.head())
        else:
            print(f"❌ FAILURE: Symbol '{symbol_to_test}' NOT found in cache or its DataFrame is empty.")

        print(f"\nFinal state of cache keys: {list(data_handler.historical_data.keys())}")

    except Exception as e:
        print(f"\n❌ AN ERROR OCCURRED DURING THE TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_simple_test()