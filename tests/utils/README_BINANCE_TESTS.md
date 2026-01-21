# Binance Client Test Suite

## Test Files

1. **`test_binance_client.py`** - Unit tests (mocked, fast)
2. **`test_binance_integration.py`** - Integration tests (real API, slow)

## Running Tests

### Unit Tests (Default)
```bash
pytest tests/utils/test_binance_client.py -v
```
✅ **30+ test cases** covering all functionality with mocked API

### Integration Tests (Real Testnet API)
```bash
pytest tests/utils/test_binance_integration.py -m integration -v
```
⚠️ **Requires internet connection** - hits real Binance testnet

### Performance Benchmarks
```bash
pytest tests/utils/test_binance_integration.py -m benchmark --benchmark-only
```
✅ Validates rate limiter overhead <100ms

### Run All Tests
```bash
pytest tests/utils/test_binance_*.py -v
```

## Integration Test Details

### Test 1: 30-Day BTC Data Fetch
- Fetches 30 days of BTCUSDT daily candles from testnet
- Validates response structure and data integrity
- No authentication required (anonymous mode)

### Test 2: 100 Sequential Requests (Stress Test)
- Makes 100 sequential API calls
- Validates no 429 (rate limit) errors
- Validates no 418 (IP ban) errors
- Tests rate limiter effectiveness

### Test 3: Rate Limiter Performance
- Benchmarks `wait_if_needed()` execution time
- Validates overhead <100ms per request
- Uses pytest-benchmark framework

## Configuration

Integration tests use **testnet only**:
```python
client = BinanceClient({'testnet': True})
```

For authenticated mode (optional), add to `local_config.yml`:
```yaml
binance_credentials:
  api_key: "your_testnet_key"
  api_secret: "your_testnet_secret"
  testnet: true
```

## Expected Results

✅ **Unit tests**: 100% pass rate (all mocked)
✅ **Integration tests**: 95%+ success rate (real API, network dependent)
✅ **Benchmark**: <100ms rate limiter overhead
