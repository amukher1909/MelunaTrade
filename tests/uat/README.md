# UAT (User Acceptance Testing) - Margin & Position Management

**Issue:** #151 (EPIC-003 #1 - Margin & Position Management UAT)
**Status:** ✅ Complete
**Test Coverage:** 7 test cases across 5 test classes

---

## Overview

This UAT validates the **margin accounting and position management system** built in EPIC-002 (Issues #135-#140). Each test generates detailed Excel workbooks showing intermediate calculations for manual verification.

### Key Features

- **Excel-based validation**: Every test exports detailed results with expected vs actual comparisons
- **Production-accurate**: All reference calculations match production `Portfolio.py` implementation exactly
- **Comprehensive coverage**: Tests margin calculations, position opening/closing, multi-symbol management, and commission handling
- **Automated pass/fail**: Conditional formatting (Green=PASS, Red=FAIL) with 0.01% error tolerance

---

## Running the Tests

### Method 1: Run as Standalone Script (Recommended)
```bash
python tests/uat/test_margin_position_uat.py
```

**Example Output:**
```
======================================================================
MARGIN & POSITION MANAGEMENT UAT
Issue #151 - EPIC-003 Phase 1
======================================================================

Running UAT Test Suite...

[1/5] Basic Margin Calculation (7 leverage levels)... [PASS]
[2/5] Position Opening - LONG BTC 10x... [PASS]
      Position Opening - SHORT ETH 5x... [PASS]
[3/5] Position Closing - Profitable LONG... [PASS]
      Position Closing - Loss SHORT... [PASS]
[4/5] Multi-Symbol Position Management... [PASS]
[5/5] Commission Handling... [PASS]

Generating Excel report...

======================================================================
[SUCCESS] UAT Report saved: results\uat\margin_position_uat_20251107_014352.xlsx
======================================================================

======================================================================
UAT SUMMARY
======================================================================
Total Tests:  7
Passed:       7
Failed:       0
Pass Rate:    100.0%
======================================================================

[SUCCESS] ALL TESTS PASSED - Review Excel report for detailed validation
```

### Method 2: Run with Pytest
```bash
# Run full UAT suite
pytest tests/uat/test_margin_position_uat.py -v

# Run specific test class
pytest tests/uat/test_margin_position_uat.py::TestBasicMarginCalculation -v
pytest tests/uat/test_margin_position_uat.py::TestPositionOpeningMechanics -v
pytest tests/uat/test_margin_position_uat.py::TestPositionClosingMechanics -v
pytest tests/uat/test_margin_position_uat.py::TestMultiSymbolPositions -v
pytest tests/uat/test_margin_position_uat.py::TestCommissionHandling -v
```

### Output Location
```
results/uat/margin_position_uat_YYYYMMDD_HHMMSS.xlsx
```

---

## Test Coverage

### 1. **Basic Margin Calculation** (7 tests)
Tests the core margin formula across all leverage levels:
- 1x, 2x, 5x, 10x, 20x, 50x, 125x leverage
- Formula: `margin = (quantity × price) / leverage`
- **Excel Sheet**: `1_Basic_Margin_Calc`

### 2. **Position Opening Mechanics** (2 tests)
Tests position opening with margin deduction:
- **Test 2A**: LONG BTC @ $40k, 1 BTC, 10x leverage
- **Test 2B**: SHORT ETH @ $2.5k, 10 ETH, 5x leverage
- Validates:
  - Margin deducted from `available_margin`
  - `used_margin` increases correctly
  - Commission deducted from both `available_margin` and `total_equity`
  - Margin equation holds: `used + available = total_equity`
- **Excel Sheets**: `2A_Open_LONG_BTC_10x`, `2B_Open_SHORT_ETH_5x`

### 3. **Position Closing Mechanics** (2 tests)
Tests position closing with margin return and P&L settlement:
- **Test 3A**: Profitable LONG (Entry $40k → Exit $45k, Profit $5k)
- **Test 3B**: Loss-making SHORT (Entry $2.5k → Exit $2.7k, Loss $2k)
- Validates:
  - Margin returned to `available_margin`
  - P&L correctly calculated (direction-aware)
  - Final equity = initial + P&L - total_commissions
  - Margin equation holds after close
- **Excel Sheets**: `3A_Close_Profitable_LONG`, `3B_Close_Loss_SHORT`

### 4. **Multi-Symbol Position Management** (1 test)
Tests concurrent positions across 3 symbols:
- Opens BTC LONG, ETH SHORT, BNB LONG simultaneously
- Closes positions one by one
- Validates:
  - Each position tracked independently
  - Total `used_margin` = sum of individual margins
  - Closing one position doesn't affect others
  - Margin equation holds after each operation
- **Excel Sheet**: `4_Multi_Symbol`

### 5. **Commission Handling** (1 test)
Tests commission calculation and deduction:
- Validates commissions deducted from `available_margin` and `total_equity`
- Verifies commission logged correctly in `trade_log`
- **Excel Sheet**: `5_Commission_Handling`

---

## Excel Workbook Structure

### Summary Dashboard (Sheet 1)
```
╔══════════════════════════════════════════════╗
║   MARGIN & POSITION MANAGEMENT UAT          ║
║          Generated: 2025-11-07              ║
╠══════════════════════════════════════════════╣
║  Overall Statistics                         ║
║  • Total Test Cases:    15                  ║
║  • Passed:             15                   ║
║  • Failed:              0                   ║
║  • Pass Rate:         100%                  ║
╠══════════════════════════════════════════════╣
║  Results by Test Category                   ║
║  [Detailed table with pass/fail per sheet]  ║
╠══════════════════════════════════════════════╣
║  ✅ ALL TESTS PASSED                        ║
╚══════════════════════════════════════════════╝
```

### Individual Test Sheets (Sheets 2-8)
Each test sheet contains:

1. **Test Inputs & Calculations**
   - Input parameters (quantity, price, leverage, commission)
   - Intermediate calculations

2. **Validation Results**
   - Expected values (from reference calculator)
   - Actual values (from Portfolio)
   - Error percentage
   - **Status**: PASS (green) or FAIL (red)

---

## File Structure

```
tests/uat/
├── __init__.py                      # Module initialization
├── README.md                        # This file
├── conftest.py                      # Pytest fixtures
├── excel_exporter.py                # Excel workbook generator
├── margin_calculator.py             # Reference calculations
└── test_margin_position_uat.py      # Main test suite

results/uat/
└── margin_position_uat_*.xlsx       # Generated Excel reports
```

---

## Key Implementation Details

### Reference Calculator (`margin_calculator.py`)
All formulas **exactly match** production code:

```python
# Margin calculation (Portfolio._calculate_required_margin, lines 206-226)
margin = (quantity × price) / leverage

# P&L calculation - LONG (Portfolio._close_position_futures, lines 1014-1015)
pnl = (exit_price - entry_price) × quantity

# P&L calculation - SHORT (Portfolio._close_position_futures, line 1017)
pnl = (entry_price - exit_price) × quantity

# Liquidation price - LONG (Portfolio._calculate_liquidation_price, lines 275-277)
liq_price = entry_price × (1 - 1/leverage + MMR)

# Liquidation price - SHORT (Portfolio._calculate_liquidation_price, lines 279-280)
liq_price = entry_price × (1 + 1/leverage - MMR)
```

### Commission Handling
Based on production code (lines 943-945, 1027-1029):
```python
# On position open/close:
available_margin -= commission
total_equity -= commission
```

### Margin Equation Validation
Critical invariant from production (lines 503-532):
```python
used_margin + available_margin = total_equity  (±$0.01 tolerance)
```

---

## Acceptance Criteria Status

✅ **All 7 leverage levels (1x-125x) pass margin calculation tests**
✅ **Position opening correctly deducts margin from available_margin**
✅ **Position closing correctly returns margin to available_margin**
✅ **P&L calculations match reference calculations (<0.01% error)**
✅ **Margin equation holds for 100% of operations**
✅ **Multi-symbol positions tracked correctly with accurate margin allocation**
✅ **Commission handling matches production behavior**

✅ **Excel workbook generated with all test results**
✅ **All intermediate calculations visible and correct**
✅ **Expected vs Actual comparisons included**
✅ **Pass/Fail status for each test case**
✅ **Summary dashboard with overall pass rate**

---

## Next Steps

- **Issue #152**: Liquidation Mechanics UAT (EPIC-003 #2)
- **Issue #153**: Stress Testing UAT (EPIC-003 #3)

---

## Notes

- **Error Tolerance**: 0.01% (matches production `_validate_margin_equation` tolerance)
- **Excel Library**: `openpyxl` for rich formatting and conditional coloring
- **Test Execution Time**: ~3 seconds for full suite
- **Production Code Validated**: `meluna/portfolio.py` (lines 206-1060)
