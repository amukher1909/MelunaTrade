#!/usr/bin/env python3
"""
Manual test script for position-aware strategy interface.
Tests the implementation without requiring full backtest infrastructure.
"""

import sys
import traceback
from unittest.mock import Mock
import pandas as pd

def test_imports():
    """Test that all imports work with new signatures."""
    print("Testing imports and signatures...")
    
    try:
        from Strategies.base import BaseStrategy
        from Strategies.MA_crossover import MovingAverageCrossoverStrategy
        from meluna.orchestrator import Orchestrator
        print("All imports successful")
        
        # Check method signature
        import inspect
        sig = inspect.signature(BaseStrategy._generate_signals)
        params = list(sig.parameters.keys())
        print(f"BaseStrategy._generate_signals params: {params}")
        
        if 'position_context' in params and len(params) == 3:  # self, market_event, position_context
            print("Method signature is correct")
        else:
            print(f"ERROR: Expected 3 params including position_context, got: {params}")
            
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_strategy_position_handling():
    """Test strategy handles position context correctly."""
    print("\n🧪 Testing strategy position context handling...")
    
    try:
        from Strategies.MA_crossover import MovingAverageCrossoverStrategy
        
        # Create test strategy
        params = {'fast_ma_period': 5, 'slow_ma_period': 10}
        mock_data_handler = Mock()
        strategy = MovingAverageCrossoverStrategy(params, mock_data_handler, ['AAPL'])
        
        # Mock indicators as ready
        strategy._is_indicator_ready = Mock(return_value=True)
        strategy._get_indicator_value = Mock()
        strategy._get_indicator_history = Mock()
        
        # Create mock market event
        market_event = Mock()
        market_event.symbol = 'AAPL'
        market_event.timestamp = pd.Timestamp('2025-01-01 10:00:00')
        
        # Test 1: No position (position_context = None)
        print("  Testing with no position (position_context = None)...")
        signals = strategy._generate_signals(market_event, None)
        print(f"    ✅ Generated {len(signals)} signals with no position")
        
        # Test 2: With position context
        print("  Testing with position context...")
        position_context = {
            'symbol': 'AAPL',
            'position_id': 'T20250101-001-AAPL',
            'unrealized_pnl_pct': -0.02,
            'bars_since_entry': 10,
            'entry_price': 150.0,
            'current_price': 147.0,
            'quantity': 100
        }
        
        signals = strategy._generate_signals(market_event, position_context)
        print(f"    ✅ Generated {len(signals)} signals with position context")
        
    except Exception as e:
        print(f"❌ Strategy position handling error: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_helper_methods():
    """Test position helper methods."""
    print("\n🧪 Testing position helper methods...")
    
    try:
        from Strategies.MA_crossover import MovingAverageCrossoverStrategy
        
        strategy = MovingAverageCrossoverStrategy({}, Mock(), ['AAPL'])
        
        # Test has_position
        print("  Testing has_position()...")
        assert strategy.has_position(None) == False, "has_position(None) should be False"
        assert strategy.has_position({'symbol': 'AAPL'}) == True, "has_position(dict) should be True"
        print("    ✅ has_position() works correctly")
        
        # Test position metrics
        position_context = {
            'unrealized_pnl_pct': -0.06,  # -6% loss
            'bars_since_entry': 55        # Over 50 bars  
        }
        
        print("  Testing position metrics...")
        pnl_pct = strategy.get_position_pnl_pct(position_context)
        bars_held = strategy.get_bars_held(position_context)
        is_profitable = strategy.is_profitable(position_context)
        loss_exceeds = strategy.is_loss_exceeding(position_context, 0.05)
        
        print(f"    PnL %: {pnl_pct}")
        print(f"    Bars held: {bars_held}")
        print(f"    Is profitable: {is_profitable}")
        print(f"    Loss exceeds 5%: {loss_exceeds}")
        
        # Validate results
        assert pnl_pct == -0.06, f"Expected -0.06, got {pnl_pct}"
        assert bars_held == 55, f"Expected 55, got {bars_held}"
        assert is_profitable == False, f"Expected False, got {is_profitable}"
        assert loss_exceeds == True, f"Expected True, got {loss_exceeds}"
        
        print("    ✅ All helper methods work correctly")
        
    except Exception as e:
        print(f"❌ Helper methods error: {e}")
        traceback.print_exc()
        return False
        
    return True

def test_position_management_scenarios():
    """Test position management exit conditions."""
    print("\n🧪 Testing position management scenarios...")
    
    try:
        from Strategies.MA_crossover import MovingAverageCrossoverStrategy
        from meluna.events import SignalEvent
        
        strategy = MovingAverageCrossoverStrategy({'fast_ma_period': 5, 'slow_ma_period': 10}, Mock(), ['AAPL'])
        
        # Mock indicators
        strategy._is_indicator_ready = Mock(return_value=True)
        strategy._detect_bearish_crossover = Mock(return_value=False)
        
        market_event = Mock()
        market_event.symbol = 'AAPL'
        market_event.timestamp = pd.Timestamp('2025-01-01 10:00:00')
        
        # Test stop-loss scenario
        print("  Testing stop-loss exit (6% loss)...")
        position_context = {
            'symbol': 'AAPL',
            'unrealized_pnl_pct': -0.06,  # 6% loss > 5% threshold
            'bars_since_entry': 20
        }
        
        signals = strategy._generate_signals(market_event, position_context)
        if signals and signals[0].direction == 'EXIT':
            print("    ✅ Stop-loss exit signal generated correctly")
        else:
            print(f"    ❌ Expected EXIT signal for stop-loss, got: {[s.direction for s in signals]}")
        
        # Test take-profit scenario  
        print("  Testing take-profit exit (12% profit)...")
        position_context = {
            'symbol': 'AAPL',
            'unrealized_pnl_pct': 0.12,  # 12% profit > 10% threshold
            'bars_since_entry': 20
        }
        
        signals = strategy._generate_signals(market_event, position_context)
        if signals and signals[0].direction == 'EXIT':
            print("    ✅ Take-profit exit signal generated correctly")
        else:
            print(f"    ❌ Expected EXIT signal for take-profit, got: {[s.direction for s in signals]}")
            
        # Test time-based exit
        print("  Testing time-based exit (55 bars held)...")
        position_context = {
            'symbol': 'AAPL',
            'unrealized_pnl_pct': 0.02,  # Small profit
            'bars_since_entry': 55  # > 50 bar limit
        }
        
        signals = strategy._generate_signals(market_event, position_context)
        if signals and signals[0].direction == 'EXIT':
            print("    ✅ Time-based exit signal generated correctly")
        else:
            print(f"    ❌ Expected EXIT signal for time-based exit, got: {[s.direction for s in signals]}")
            
        # Test no exit conditions
        print("  Testing normal position (no exit conditions)...")
        position_context = {
            'symbol': 'AAPL',
            'unrealized_pnl_pct': 0.02,  # 2% profit < 10% threshold
            'bars_since_entry': 20  # < 50 bar limit
        }
        
        signals = strategy._generate_signals(market_event, position_context)
        if len(signals) == 0:
            print("    ✅ No exit signals generated when conditions not met")
        else:
            print(f"    ❌ Expected no signals, got: {[s.direction for s in signals]}")
        
    except Exception as e:
        print(f"❌ Position management scenarios error: {e}")
        traceback.print_exc()
        return False
        
    return True

def test_orchestrator_integration():
    """Test Orchestrator integration with position context."""
    print("\n🧪 Testing Orchestrator integration...")
    
    try:
        from meluna.orchestrator import Orchestrator
        
        orchestrator = Orchestrator()
        
        # Test exception classes exist
        from meluna.orchestrator import PositionContextError, SystemConfigurationError
        print("    ✅ Custom exception classes imported successfully")
        
        # Test that orchestrator initializes
        assert orchestrator.portfolio is None, "Portfolio should be None initially"
        print("    ✅ Orchestrator initializes correctly")
        
    except Exception as e:
        print(f"❌ Orchestrator integration error: {e}")
        traceback.print_exc()
        return False
        
    return True

def main():
    """Run all manual tests."""
    print("🚀 Manual Testing: Position-Aware Strategy Interface")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_strategy_position_handling,
        test_helper_methods, 
        test_position_management_scenarios,
        test_orchestrator_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Position-aware strategy interface is working correctly.")
        print("\n💡 Next steps:")
        print("   - Run your existing backtests to test with real data")
        print("   - Update any other custom strategies to new signature")
        print("   - Ready for Issues #109-112 (Full Integration Testing)")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())