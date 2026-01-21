#!/usr/bin/env python
"""
Quick test verification script.
Run this with your Python: python verify_tests.py
"""

import sys
import os

print("=" * 60)
print("TEST VERIFICATION SCRIPT")
print("=" * 60)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

# Test imports
try:
    import pytest
    print(f"✓ pytest {pytest.__version__} installed")
except ImportError:
    print("✗ pytest NOT installed - run: pip install pytest")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__} installed")
except ImportError:
    print("✗ pandas NOT installed")

# Test file existence
test_files = [
    "tests/regression/test_equity_handlers.py",
    "tests/integration/test_binance_pipeline.py",
    "tests/benchmarks/validate_tradingview.py"
]

print("\n" + "=" * 60)
print("CHECKING TEST FILES")
print("=" * 60)

for test_file in test_files:
    if os.path.exists(test_file):
        print(f"✓ {test_file}")
    else:
        print(f"✗ {test_file} NOT FOUND")

# Try importing test modules
print("\n" + "=" * 60)
print("TESTING IMPORTS")
print("=" * 60)

sys.path.insert(0, os.getcwd())

try:
    from data.binance_handler import BinanceDataHandler
    print("✓ BinanceDataHandler imports successfully")
except Exception as e:
    print(f"✗ BinanceDataHandler import failed: {e}")

try:
    from data.fyers_handler import FyersDataHandler
    print("✓ FyersDataHandler imports successfully")
except Exception as e:
    print(f"✗ FyersDataHandler import failed: {e}")

try:
    from data.zerodha_handler import ZerodhaDataHandler
    print("✓ ZerodhaDataHandler imports successfully")
except Exception as e:
    print(f"✗ ZerodhaDataHandler import failed: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("Run regression tests with:")
print(f"  {sys.executable} -m pytest tests/regression/ -v --no-cov")
print()
print("Run integration tests with:")
print(f"  {sys.executable} -m pytest tests/integration/ -v --no-cov -m integration")
print("=" * 60)
