#!/usr/bin/env python3
"""
Manual validation script: Compare Binance data with TradingView.

Usage:
1. Fetch 1 year BTC data from Binance mainnet (this script)
2. Export same period from TradingView as CSV
3. Run comparison (this script)

Target: < 0.01% difference (EPIC-001 success criteria)

Run: python tests/benchmarks/validate_tradingview.py
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from data.binance_handler import BinanceDataHandler
from meluna.utils.binance_client import BinanceClient


def fetch_binance_data(days=365):
    """Fetch historical data from Binance mainnet."""
    print(f"📡 Fetching {days} days of BTC data from Binance mainnet...")

    client = BinanceClient({'mode': 'anonymous', 'testnet': False})
    handler = BinanceDataHandler(['BTCUSDT'], client, interval='1d')

    end_date = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

    df = handler.get_data('BTCUSDT', start_date, end_date)
    print(f"✓ Fetched {len(df)} bars from Binance")

    return df


def compare_with_tradingview(binance_df, tradingview_csv):
    """
    Compare Binance data with TradingView export.

    Args:
        binance_df: DataFrame from BinanceDataHandler
        tradingview_csv: Path to TradingView CSV export

    Returns:
        Dictionary with comparison metrics
    """
    print(f"\n📊 Comparing with TradingView data from {tradingview_csv}...")

    # Load TradingView data
    tv_df = pd.read_csv(tradingview_csv)
    tv_df['date'] = pd.to_datetime(tv_df['time'], utc=True)

    # Merge on date
    merged = binance_df.merge(tv_df, on='date', suffixes=('_binance', '_tv'))

    # Calculate price differences
    for col in ['open', 'high', 'low', 'close']:
        diff = (merged[f'{col}_binance'] - merged[f'{col}_tv']).abs()
        pct_diff = (diff / merged[f'{col}_tv']) * 100
        merged[f'{col}_diff_pct'] = pct_diff

    # Summary statistics
    results = {
        'total_bars': len(merged),
        'missing_binance': len(tv_df) - len(merged),
        'missing_tv': len(binance_df) - len(merged),
        'max_price_diff_pct': merged[
            ['open_diff_pct', 'high_diff_pct', 'low_diff_pct', 'close_diff_pct']
        ].max().max(),
        'mean_price_diff_pct': merged[
            ['open_diff_pct', 'high_diff_pct', 'low_diff_pct', 'close_diff_pct']
        ].mean().mean()
    }

    return results, merged


def main():
    """Main validation workflow."""
    print("=" * 60)
    print("TradingView Validation for EPIC-001")
    print("=" * 60)

    # Step 1: Fetch Binance data
    binance_df = fetch_binance_data(days=365)

    # Step 2: Get TradingView CSV path from user
    print("\n📝 Export BTC/USDT daily data from TradingView:")
    print("   1. Open https://www.tradingview.com/symbols/BTCUSDT/")
    print("   2. Set timeframe to 1D (daily)")
    print("   3. Click '...' → Export chart data")
    print("   4. Save CSV file")

    tradingview_csv = input("\nEnter path to TradingView CSV: ").strip()

    if not Path(tradingview_csv).exists():
        print(f"❌ File not found: {tradingview_csv}")
        return

    # Step 3: Compare
    results, merged = compare_with_tradingview(binance_df, tradingview_csv)

    # Step 4: Report results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total bars matched: {results['total_bars']}")
    print(f"Missing in Binance: {results['missing_binance']}")
    print(f"Missing in TradingView: {results['missing_tv']}")
    print(f"\nMax price difference: {results['max_price_diff_pct']:.4f}%")
    print(f"Mean price difference: {results['mean_price_diff_pct']:.4f}%")

    # Step 5: Success/Fail
    print("\n" + "=" * 60)
    if results['max_price_diff_pct'] < 0.01:
        print("✅ SUCCESS: Price match < 0.01% (EPIC-001 target met)")
    else:
        print(f"❌ FAIL: Price match {results['max_price_diff_pct']:.4f}% > 0.01% target")

    # Step 6: Continuity check
    if 365 <= len(binance_df) <= 366:
        print(f"✅ SUCCESS: 365-day continuity verified ({len(binance_df)} bars)")
    else:
        print(f"⚠ WARNING: Expected 365-366 bars, got {len(binance_df)}")

    print("=" * 60)

    # Save comparison report
    output_csv = 'tests/benchmarks/tradingview_comparison.csv'
    merged.to_csv(output_csv, index=False)
    print(f"\n📄 Detailed comparison saved to: {output_csv}")


if __name__ == '__main__':
    main()
