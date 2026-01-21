import logging
import yaml
from typing import List, Dict, Any, Optional
from Strategies.base import BaseStrategy
from meluna.events import SignalEvent
import meluna.technical_analysis as ta

# Get a logger specific to this strategy file
logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Position-Aware Moving Average Crossover Strategy with Configurable Risk Management.
    
    This strategy demonstrates complete position-aware trading with dedicated configuration:
    - Fully configurable via dedicated YAML file (moving_average_crossover_config.yaml)
    - Comprehensive position management with stop-loss, take-profit, and time-based exits
    - Feature toggles for each exit condition type (enable/disable for testing)
    - Streaming indicators for O(1) performance with real-time updates
    - Complete separation of strategy config from system configuration
    
    Position Management Logic:
    1. Entry Signals: Bullish MA crossover (fast MA crosses above slow MA)
    2. Exit Conditions (configurable via features section):
       - Technical: Bearish MA crossover (fast MA crosses below slow MA)
       - Stop-Loss: Configurable percentage loss threshold
       - Take-Profit: Configurable percentage profit target  
       - Time-Based: Maximum bars held limit
    
    Configuration Structure:
    - technical: MA periods for signal generation
    - position_management: Risk management parameters
    - features: Enable/disable specific exit conditions
    - metadata: Strategy versioning and documentation
    
    This serves as the reference implementation for position-aware strategy development.
    """
    def __init__(self, config_path: str, data_handler: 'DataHandler', symbol_list: List[str]):
        """
        Initialize strategy with dedicated configuration file.
        
        Args:
            config_path: Path to the strategy-specific YAML configuration file
            data_handler: DataHandler instance for market data access
            symbol_list: List of symbols to trade
        """
        self.symbol_list = symbol_list
        
        # Load strategy-specific configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Technical analysis parameters
        technical = config.get('technical', {})
        self.fast_ma_period = technical.get('fast_ma_period', 15)
        self.slow_ma_period = technical.get('slow_ma_period', 30)
        
        # Position management parameters
        pos_mgmt = config.get('position_management', {})
        self.stop_loss_pct = pos_mgmt.get('stop_loss_pct', 0.05)
        self.take_profit_pct = pos_mgmt.get('take_profit_pct', 0.10)
        self.max_bars_held = pos_mgmt.get('max_bars_held', 50)
        
        # Feature toggles
        features = config.get('features', {})
        self.enable_technical_exits = features.get('enable_technical_exits', True)
        self.enable_stop_loss = features.get('enable_stop_loss', True)
        self.enable_take_profit = features.get('enable_take_profit', True)
        self.enable_time_exits = features.get('enable_time_exits', True)
        
        # Create parameters dict for base class compatibility
        parameters = {
            'fast_ma_period': self.fast_ma_period,
            'slow_ma_period': self.slow_ma_period,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_bars_held': self.max_bars_held
        }
        
        super().__init__(parameters, data_handler)
    
    def _setup_indicators(self) -> None:
        """
        Set up streaming indicators for each symbol using the factory pattern.
        
        Creates SMA indicators for both fast and slow moving averages using
        the technical analysis factory.
        """
        for symbol in self.symbol_list:
            self.indicators[symbol] = {
                'fast_ma': ta.create('sma', period=self.fast_ma_period, input_field='close'),
                'slow_ma': ta.create('sma', period=self.slow_ma_period, input_field='close')
            }
        
        self._use_streaming_indicators = True
        self._logger.info(f"Initialized streaming indicators: SMA({self.fast_ma_period}) and SMA({self.slow_ma_period}) for symbols {self.symbol_list}")

    def _generate_signals(self, market_event: 'MarketEvent', position_context: Optional[Dict]) -> List['SignalEvent']:
        """
        Generate trading signals based on moving average crossovers AND position management.
        
        Strategy Logic:
        - If no position: Look for bullish crossover entry signals
        - If position exists: Check for exit conditions (bearish crossover, stop-loss, time-based)
        
        Uses streaming indicators to detect crossovers and generate appropriate signals.
        """
        symbol = market_event.symbol
        timestamp = market_event.timestamp
        signals = []
        
        logger.debug(f"Strategy processing event for {symbol} at {timestamp}")
        
        # Check if indicators are ready
        fast_ready = self._is_indicator_ready(symbol, 'fast_ma')
        slow_ready = self._is_indicator_ready(symbol, 'slow_ma')
        
        if not fast_ready or not slow_ready:
            logger.debug(f"Indicators not ready for {symbol} (fast_ma: {fast_ready}, slow_ma: {slow_ready}), skipping signal generation")
            return signals
        
        # Check if we have position in this symbol
        if self.has_position(position_context):
            # POSITION MANAGEMENT MODE
            logger.debug(f"Managing existing position for {symbol}")
            signals.extend(self._manage_existing_position(market_event, position_context))
        else:
            # ENTRY SIGNAL MODE  
            logger.debug(f"Looking for entry signals for {symbol}")
            signals.extend(self._find_entry_signals(market_event))
        
        return signals

    def _manage_existing_position(self, market_event: 'MarketEvent', position_context: Dict) -> List['SignalEvent']:
        """
        Handle position management with configurable exit conditions.
        
        This method implements comprehensive position-aware risk management:
        - All exit thresholds are configurable via strategy config file
        - Feature toggles allow enabling/disabling specific exit types for testing
        - Exit conditions are evaluated in priority order (stop-loss first)
        - Detailed logging for trade analysis and debugging
        
        Exit Priority Order:
        1. Stop-Loss: Prevents catastrophic losses (highest priority)
        2. Take-Profit: Locks in gains at target level
        3. Time-Based: Prevents indefinite position holding
        4. Technical: Trend reversal detection via MA crossover
        
        Args:
            market_event: Current market data event
            position_context: Current position details (entry price, bars held, etc.)
            
        Returns:
            List of EXIT signals if any exit condition is met, empty list otherwise
        """
        signals = []
        symbol = market_event.symbol
        timestamp = market_event.timestamp
        
        # Get position metrics
        pnl_pct = self.get_position_pnl_pct(position_context)
        bars_held = self.get_bars_held(position_context)
        
        # Check exit conditions based on feature toggles
        
        # 1. Stop-loss: Exit if loss exceeds configured threshold (if enabled)
        if self.enable_stop_loss and self.is_loss_exceeding(position_context, self.stop_loss_pct):
            logger.info(f"STOP-LOSS EXIT: {symbol} loss {pnl_pct:.2%} exceeds {self.stop_loss_pct:.1%} threshold")
            signals.append(SignalEvent(symbol=symbol, timestamp=timestamp, direction='EXIT'))
            return signals
        
        # 2. Take-profit: Exit if profit exceeds configured target (if enabled)
        if self.enable_take_profit and pnl_pct > self.take_profit_pct:
            logger.info(f"TAKE-PROFIT EXIT: {symbol} profit {pnl_pct:.2%} exceeds {self.take_profit_pct:.1%} target")
            signals.append(SignalEvent(symbol=symbol, timestamp=timestamp, direction='EXIT'))
            return signals
        
        # 3. Time-based: Exit if held more than configured max bars (if enabled)
        if self.enable_time_exits and bars_held > self.max_bars_held:
            logger.info(f"TIME-BASED EXIT: {symbol} held for {bars_held} bars, exceeds {self.max_bars_held}-bar limit")
            signals.append(SignalEvent(symbol=symbol, timestamp=timestamp, direction='EXIT'))
            return signals
        
        # 4. Technical exit: Check for bearish crossover (if enabled)
        if self.enable_technical_exits and self._detect_bearish_crossover(market_event):
            logger.info(f"TECHNICAL EXIT: {symbol} bearish MA crossover detected")
            signals.append(SignalEvent(symbol=symbol, timestamp=timestamp, direction='EXIT'))
            return signals
        
        return signals

    def _find_entry_signals(self, market_event: 'MarketEvent') -> List['SignalEvent']:
        """Find entry signals - existing MA crossover logic."""
        symbol = market_event.symbol
        timestamp = market_event.timestamp
        signals = []
        
        # Get current indicator values
        fast_ma_curr = self._get_indicator_value(symbol, 'fast_ma')
        slow_ma_curr = self._get_indicator_value(symbol, 'slow_ma')
        
        # Get historical values for crossover detection
        fast_ma_history = self._get_indicator_history(symbol, 'fast_ma', 2)
        slow_ma_history = self._get_indicator_history(symbol, 'slow_ma', 2)
        
        if len(fast_ma_history) < 2 or len(slow_ma_history) < 2:
            return signals
            
        fast_ma_prev = fast_ma_history[-2]
        slow_ma_prev = slow_ma_history[-2]
        
        # Calculate MA relationship and changes
        fast_change = fast_ma_curr - fast_ma_prev
        slow_change = slow_ma_curr - slow_ma_prev
        ma_gap_prev = fast_ma_prev - slow_ma_prev
        ma_gap_curr = fast_ma_curr - slow_ma_curr
        gap_change = ma_gap_curr - ma_gap_prev
        
        # Log current values and analysis
        logger.debug(f"[{timestamp}] {symbol} MA Analysis - "
                    f"Fast: {fast_ma_prev:.4f}→{fast_ma_curr:.4f} (Δ{fast_change:+.4f}) | "
                    f"Slow: {slow_ma_prev:.4f}→{slow_ma_curr:.4f} (Δ{slow_change:+.4f}) | "
                    f"Gap: {ma_gap_prev:.4f}→{ma_gap_curr:.4f} (Δ{gap_change:+.4f})")
        
        # Detect bullish crossover for entry
        bullish_crossover = fast_ma_prev <= slow_ma_prev and fast_ma_curr > slow_ma_curr
        
        if bullish_crossover:
            signal_strength = abs(gap_change)
            logger.info(f"BULLISH CROSSOVER DETECTED: {symbol} at {timestamp} - "
                       f"Fast MA ({fast_ma_curr:.4f}) crossed above Slow MA ({slow_ma_curr:.4f}) "
                       f"with strength {signal_strength:.4f}")
            signals.append(SignalEvent(symbol=symbol, timestamp=timestamp, direction='LONG'))
        
        # Log potential trend signals (when MAs are converging/diverging without crossing)
        elif abs(ma_gap_curr) < abs(ma_gap_prev) and abs(ma_gap_curr) < 0.01:  # Converging
            logger.debug(f"[{timestamp}] {symbol} MAs converging - potential crossover approaching "
                        f"(gap: {ma_gap_curr:.4f}, narrowing by {-abs(gap_change):.4f})")
        elif abs(gap_change) > 0.005:  # Significant divergence
            direction = "diverging" if abs(ma_gap_curr) > abs(ma_gap_prev) else "converging"
            trend = "bullish" if ma_gap_curr > 0 else "bearish"
            logger.debug(f"[{timestamp}] {symbol} MAs {direction} in {trend} direction "
                        f"(gap change: {gap_change:+.4f})")
        
        return signals

    def _detect_bearish_crossover(self, market_event: 'MarketEvent') -> bool:
        """Detect bearish crossover for position exit."""
        symbol = market_event.symbol
        
        # Get current indicator values
        fast_ma_curr = self._get_indicator_value(symbol, 'fast_ma')
        slow_ma_curr = self._get_indicator_value(symbol, 'slow_ma')
        
        # Get historical values for crossover detection
        fast_ma_history = self._get_indicator_history(symbol, 'fast_ma', 2)
        slow_ma_history = self._get_indicator_history(symbol, 'slow_ma', 2)
        
        if len(fast_ma_history) < 2 or len(slow_ma_history) < 2:
            return False
            
        fast_ma_prev = fast_ma_history[-2]
        slow_ma_prev = slow_ma_history[-2]
        
        # Detect bearish crossover
        return fast_ma_prev >= slow_ma_prev and fast_ma_curr < slow_ma_curr