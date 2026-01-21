# Testing Guide

## Test Structure

```
tests/
├── integration/          # Full pipeline tests (requires testnet API)
├── regression/           # Backward compatibility tests
├── benchmarks/           # Manual validation scripts
├── data/                 # Unit tests for data handlers
└── utils/                # Unit tests for utilities
```

## Running Tests

### Unit Tests (Fast, No API)
```bash
pytest tests/data/ tests/utils/ -m "not integration" --no-cov
```

### Integration Tests (Requires Testnet API)
```bash
pytest tests/integration/ -m integration --no-cov
```

### Regression Tests (Fast)
```bash
pytest tests/regression/ --no-cov
```

### All Tests
```bash
pytest tests/ --no-cov
```

## Manual Benchmarks

### TradingView Validation
```bash
python tests/benchmarks/validate_tradingview.py
```

Follow the interactive prompts to compare Binance data with TradingView exports.

## Success Criteria (EPIC-001)

- ✅ TradingView price match < 0.01%
- ✅ 365-day continuity (1 year = 365 bars)
- ✅ Schema consistency (matches Fyers/Zerodha)
- ✅ Zero regression to equity handlers
