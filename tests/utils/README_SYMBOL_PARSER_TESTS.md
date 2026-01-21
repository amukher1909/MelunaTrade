# Binance Symbol Parser - Testing Guide

## Overview

The Binance Symbol Parser has two test suites:
1. **Unit Tests (pytest)**: Standard pytest tests for CI/CD integration
2. **UAT Tests (standalone)**: User Acceptance Testing with detailed console output

---

## Running UAT Tests (Recommended for Manual Testing)

### Quick Start

**Option 1: Windows Batch File**
```cmd
run_parser_uat.bat
```

**Option 2: Direct Python Execution**
```bash
python test_binance_symbol_parser_uat.py
```

### What You'll See

The UAT script provides detailed output for every test:

```
================================================================================
  TEST SECTION 1: Perpetual USDT-Margined Parsing
================================================================================

🔹 Test 1.1: Parse BTCUSDT (BTC Perpetual)
   Input Symbol: BTCUSDT
   Parsed Result: {'type': 'perpetual', 'base': 'BTC', 'quote': 'USDT', ...}
   Type: perpetual
   Base: BTC
   Quote: USDT
   Base Pair: BTCUSDT
   ✅ PASS
```

### Test Sections Covered

1. **Perpetual USDT-Margined** (3 tests)
   - BTC, ETH, SOL perpetual contracts

2. **Quarterly USDT-Margined** (6 tests)
   - Future dates, expiry validation, near-expiry warnings
   - Invalid date formats, expired contracts

3. **Perpetual Coin-Margined** (2 tests)
   - BTC, ETH coin perpetual contracts

4. **Quarterly Coin-Margined** (2 tests)
   - Future dates, year rollover logic

5. **Invalid Formats** (7 tests)
   - Wrong separators, missing parts, lowercase, invalid lengths

6. **Utility Methods** (5 tests)
   - is_perpetual(), is_quarterly(), get_expiry_date()
   - to_cache_filename(), filename uniqueness

**Total: 25+ comprehensive tests with detailed output**

---

## Running Unit Tests (For Pytest/CI)

### Prerequisites
```bash
pip install pytest
```

### Run Tests
```bash
# Run all symbol parser tests
pytest tests/utils/test_binance_symbol_parser.py -v

# Run with output
pytest tests/utils/test_binance_symbol_parser.py -v -s

# Run specific test class
pytest tests/utils/test_binance_symbol_parser.py::TestPerpetualUSDT -v
```

### Test Classes

- `TestPerpetualUSDT`: Perpetual USDT-margined tests
- `TestQuarterlyUSDT`: Quarterly USDT-margined tests
- `TestPerpetualCoin`: Perpetual coin-margined tests
- `TestQuarterlyCoin`: Quarterly coin-margined tests
- `TestInvalidFormats`: Invalid symbol format tests
- `TestUtilityMethods`: Utility method tests

**Total: 40+ unit tests**

---

## Example Usage

### Basic Parsing
```python
from meluna.utils.binance_symbol_parser import BinanceSymbolParser

parser = BinanceSymbolParser()

# Parse perpetual
info = parser.parse('BTCUSDT')
# {'type': 'perpetual', 'base': 'BTC', 'quote': 'USDT', 'base_pair': 'BTCUSDT'}

# Parse quarterly
info = parser.parse('BTCUSDT_250328')
# {'type': 'quarterly', 'base': 'BTC', 'quote': 'USDT', 'base_pair': 'BTCUSDT',
#  'expiry': date(2025, 3, 28), 'expiry_suffix': '250328'}
```

### Utility Methods
```python
# Check contract type
parser.is_perpetual('BTCUSDT')  # True
parser.is_quarterly('BTCUSDT_250328')  # True

# Get expiry date
parser.get_expiry_date('BTCUSDT')  # None (perpetual)
parser.get_expiry_date('BTCUSDT_250328')  # date(2025, 3, 28)

# Generate cache filename
parser.to_cache_filename('BTCUSDT', '1d', 'mainnet')
# 'BTCUSDT-binance-mainnet-1d.parquet'
```

---

## Expected Test Results

### Success Criteria

✅ **All tests pass** = 100% success rate
- Parser correctly identifies all 4 contract types
- Expiry dates parsed accurately
- Invalid formats raise appropriate errors
- Utility methods return expected values

### Known Edge Cases Covered

1. **Expired contracts**: Raises `ValueError`
2. **Near-expiry warning**: Logs warning for contracts < 7 days
3. **Invalid dates**: Feb 30, Month 13, etc. raise errors
4. **Year rollover**: Coin quarterly with past dates assumes next year
5. **Lowercase symbols**: Currently raise error (Binance uses uppercase)

---

## Troubleshooting

### Python Not Found

If you see "Python not found":

1. **Check Python installation**:
   ```cmd
   python --version
   py --version
   ```

2. **Add Python to PATH** (Windows):
   - Search "Environment Variables"
   - Edit PATH
   - Add Python installation directory

3. **Use full path**:
   ```cmd
   C:\Python39\python.exe test_binance_symbol_parser_uat.py
   ```

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Ensure you're in the project root
cd C:\Users\Aditya\OneDrive\Meluna

# Run from project root
python test_binance_symbol_parser_uat.py
```

### Test Failures

If tests fail:

1. **Check date-related tests**: Some tests use relative dates (today + X days)
2. **Check parser implementation**: Review `meluna/utils/binance_symbol_parser.py`
3. **Check error messages**: UAT output shows detailed failure reasons

---

## Files

| File | Purpose | Usage |
|------|---------|-------|
| `meluna/utils/binance_symbol_parser.py` | Core parser implementation | Import and use in production code |
| `test_binance_symbol_parser_uat.py` | UAT test suite (standalone) | Run directly with Python for manual testing |
| `tests/utils/test_binance_symbol_parser.py` | Unit tests (pytest) | Run with pytest for CI/CD |
| `run_parser_uat.bat` | Windows batch launcher | Double-click to run UAT tests |
| This file | Testing documentation | Reference guide |

---

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# Example: GitHub Actions
- name: Run Symbol Parser Tests
  run: |
    pytest tests/utils/test_binance_symbol_parser.py -v --tb=short
```

---

## Next Steps

After validating the parser:

1. **Integrate with Data Handler** (Issue #2): Use parser for symbol validation
2. **Implement Caching** (Issue #3): Use `to_cache_filename()` for unique cache files
3. **Add Config Validation**: Validate symbols in `config.yml` before backtest
4. **Contract Rollover** (EPIC-004): Use `get_expiry_date()` for rollover logic

---

**Questions?** See main project documentation or raise an issue on GitHub.
