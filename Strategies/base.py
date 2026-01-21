# meluna/strategy.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import time
import math

# We use forward-declarations for type hints to avoid circular imports,
# as the DataHandler and Event modules will be used by this module.
class DataHandler: pass
class MarketEvent: pass
class SignalEvent: pass

class BaseStrategy(ABC):
    """
    The abstract base class for all quantitative trading strategies. [cite: 1306]

    This class defines the non-negotiable contract that all strategies must
    adhere to. It acts as a pure, portfolio-unaware signal generator, 
    encapsulating its own logic and state. [cite: 1215, 1306]
    """
    def __init__(self, parameters: Dict[str, Any], data_handler: 'DataHandler'):
        """
        Initializes the strategy instance. [cite: 1307]

        Args:
            parameters (Dict[str, Any]): A dictionary of user-defined parameters 
                                        that configure the strategy's behavior. 
            data_handler (DataHandler): A reference to the system's "jailed" 
                                      DataHandler, the strategy's sole gateway 
                                      to market data. [cite: 1308, 1347]
        """
        self.parameters = parameters
        self.data_handler = data_handler
        
        # Streaming indicator support
        self.indicators: Dict[str, Dict[str, Any]] = {}
        self._use_streaming_indicators = False
        
        # Setup logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize streaming indicators if supported
        self._setup_indicators()
        
        # The strategy is responsible for initializing its own state variables here. [cite: 1309]
        # e.g., self.moving_average = None

    def on_market_data(self, market_event: 'MarketEvent', position_context: Optional[Dict]) -> List['SignalEvent']:
        """
        The core logic of the strategy. This method is called by the Orchestrator
        for each new market data event. [cite: 1310]
        
        This method automatically updates streaming indicators before calling
        the strategy-specific signal generation logic.

        Args:
            market_event (MarketEvent): An object containing the latest market data. [cite: 1312]
            position_context (Optional[Dict]): Position details for this symbol. Contains position_id, 
                                              entry_price, current P&L, etc. None if no position exists.

        Returns:
            List[SignalEvent]: A list of SignalEvent objects. The list can be empty
                               if no signals are generated. [cite: 1313, 1314]
        """
        # Update streaming indicators first if enabled
        if self._use_streaming_indicators:
            self._update_indicators(market_event)
        
        # Call strategy-specific signal generation logic
        return self._generate_signals(market_event, position_context)
    
    @abstractmethod
    def _generate_signals(self, market_event: 'MarketEvent', position_context: Optional[Dict]) -> List['SignalEvent']:
        """
        Strategy-specific signal generation logic.
        
        This method should be implemented by concrete strategies to define
        their trading logic. Indicators will already be updated when this
        method is called.
        
        Args:
            market_event (MarketEvent): An object containing the latest market data.
            position_context (Optional[Dict]): Position details for this symbol. Contains position_id, 
                                              entry_price, current P&L, etc. None if no position exists.
            
        Returns:
            List[SignalEvent]: A list of SignalEvent objects.
        """
        raise NotImplementedError("Should implement _generate_signals()")

    def _calculate_confidence_score(self, context: Dict[str, Any]) -> float:
        """
        Calculates a normalized confidence score for a generated signal.
        This method is designed to be overridden by concrete strategies. [cite: 1316]
        The base implementation returns a neutral score. [cite: 1317]

        Args:
            context (Dict[str, Any]): A dictionary containing relevant data at the
                                     time of signal generation. [cite: 1318]

        Returns:
            A float between 0.0 and 1.0 representing signal confidence. [cite: 1319]
        """
        return 0.5 # Default implementation returns a neutral score. [cite: 1320]
    
    def _setup_indicators(self) -> None:
        """
        Initialize streaming indicators for the strategy.
        
        Override this method in concrete strategies to define streaming indicators.
        This method is called during strategy initialization.
        
        Example:
            def _setup_indicators(self) -> None:
                for symbol in self.symbol_list:
                    self.indicators[symbol] = {
                        'fast_ma': ta.create('sma', period=self.fast_ma_period),
                        'slow_ma': ta.create('sma', period=self.slow_ma_period)
                    }
                self._use_streaming_indicators = True
        """
        pass  # Default implementation does nothing
    
    def _update_indicators(self, market_event: 'MarketEvent') -> None:
        """
        Update all streaming indicators for the given symbol with new market data.
        
        This method automatically converts MarketEvent data to the format
        expected by indicators and updates all indicators for the symbol.
        Provides detailed logging of update performance and values.
        
        Args:
            market_event (MarketEvent): Market data event to process
        """
        symbol = market_event.symbol
        timestamp = market_event.timestamp
        
        if symbol not in self.indicators:
            self._logger.warning(f"No indicators defined for symbol {symbol}")
            return
        
        # Convert MarketEvent to indicator input format
        data_point = self._market_event_to_dict(market_event)
        
        # Track overall update timing
        update_start_time = time.perf_counter()
        
        # Log market data being processed
        self._logger.debug(f"[{timestamp}] Processing market data for {symbol}: "
                          f"OHLCV=({market_event.open:.2f}, {market_event.high:.2f}, "
                          f"{market_event.low:.2f}, {market_event.close:.2f}, {market_event.volume})")
        
        updated_indicators = []
        indicator_timings = {}
        
        # Update all indicators for this symbol
        for indicator_name, indicator in self.indicators[symbol].items():
            try:
                # Capture indicator state before update
                was_ready_before = getattr(indicator, 'is_ready', False)
                old_value = None
                if was_ready_before:
                    try:
                        old_value = getattr(indicator, 'value', None)
                    except:
                        old_value = None
                
                # Time the individual indicator update
                indicator_start_time = time.perf_counter()
                indicator.update(data_point)
                indicator_update_time = time.perf_counter() - indicator_start_time
                
                # Capture indicator state after update
                is_ready_after = getattr(indicator, 'is_ready', False)
                new_value = None
                if is_ready_after:
                    try:
                        new_value = getattr(indicator, 'value', None)
                    except:
                        new_value = None
                
                # Store timing info
                indicator_timings[indicator_name] = indicator_update_time
                
                # Log detailed update information
                self._log_indicator_update(
                    symbol=symbol,
                    timestamp=timestamp,
                    indicator_name=indicator_name,
                    was_ready_before=was_ready_before,
                    is_ready_after=is_ready_after,
                    old_value=old_value,
                    new_value=new_value,
                    update_time_ms=indicator_update_time * 1000,
                    input_price=data_point.get('close', 'N/A')
                )
                
                updated_indicators.append(indicator_name)
                
            except Exception as e:
                self._logger.error(f"Error updating indicator {indicator_name} for {symbol}: {e}")
        
        # Log overall update summary
        total_update_time = time.perf_counter() - update_start_time
        
        if updated_indicators:
            avg_update_time = sum(indicator_timings.values()) / len(indicator_timings)
            max_update_time = max(indicator_timings.values())
            
            self._logger.info(f"[{timestamp}] Updated {len(updated_indicators)} indicators for {symbol} "
                            f"in {total_update_time*1000:.3f}ms "
                            f"(avg: {avg_update_time*1000:.3f}ms, max: {max_update_time*1000:.3f}ms) "
                            f"- Indicators: {', '.join(updated_indicators)}")
            
            # Log detailed timing breakdown at debug level
            timing_details = [f"{name}: {time_ms*1000:.3f}ms" 
                            for name, time_ms in indicator_timings.items()]
            self._logger.debug(f"[{timestamp}] Indicator timing breakdown for {symbol}: {', '.join(timing_details)}")
    
    def _log_indicator_update(self, symbol: str, timestamp, indicator_name: str, 
                            was_ready_before: bool, is_ready_after: bool,
                            old_value: Any, new_value: Any, update_time_ms: float,
                            input_price: Any) -> None:
        """
        Log detailed information about a single indicator update.
        
        Args:
            symbol (str): Symbol being processed
            timestamp: Market event timestamp
            indicator_name (str): Name of the indicator
            was_ready_before (bool): Indicator readiness before update
            is_ready_after (bool): Indicator readiness after update
            old_value (Any): Previous indicator value
            new_value (Any): New indicator value after update
            update_time_ms (float): Time taken to update in milliseconds
            input_price (Any): Input price used for calculation
        """
        # Format values for logging
        old_val_str = self._format_indicator_value(old_value)
        new_val_str = self._format_indicator_value(new_value)
        
        # Determine status change
        status_msg = ""
        if not was_ready_before and is_ready_after:
            status_msg = " [BECAME READY]"
        elif was_ready_before and not is_ready_after:
            status_msg = " [BECAME NOT READY]"
        
        # Calculate value change if both values are available
        change_msg = ""
        if (was_ready_before and is_ready_after and 
            old_value is not None and new_value is not None):
            try:
                if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    change = new_value - old_value
                    change_pct = (change / old_value * 100) if old_value != 0 else 0
                    change_msg = f" (Δ{change:+.4f}, {change_pct:+.2f}%)"
            except:
                pass
        
        # Log the update
        self._logger.debug(f"[{timestamp}] {symbol}.{indicator_name}: "
                          f"{old_val_str} → {new_val_str}{change_msg} "
                          f"[{update_time_ms:.3f}ms] input={input_price}{status_msg}")
    
    def _format_indicator_value(self, value: Any) -> str:
        """
        Format indicator value for logging display.
        
        Args:
            value (Any): Indicator value to format
            
        Returns:
            str: Formatted string representation
        """
        if value is None:
            return "None"
        elif isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            elif math.isinf(value):
                return "Inf" if value > 0 else "-Inf"
            else:
                return f"{value:.4f}"
        elif isinstance(value, dict):
            # For composite indicators that return dictionaries
            formatted_parts = []
            for k, v in value.items():
                if isinstance(v, float):
                    if math.isnan(v):
                        formatted_parts.append(f"{k}:NaN")
                    elif math.isinf(v):
                        inf_str = "Inf" if v > 0 else "-Inf"
                        formatted_parts.append(f"{k}:{inf_str}")
                    else:
                        formatted_parts.append(f"{k}:{v:.4f}")
                else:
                    formatted_parts.append(f"{k}:{v}")
            return "{" + ", ".join(formatted_parts) + "}"
        else:
            return str(value)
    
    def _market_event_to_dict(self, market_event: 'MarketEvent') -> Dict[str, Any]:
        """
        Convert MarketEvent to dictionary format expected by indicators.
        
        Args:
            market_event (MarketEvent): Market event to convert
            
        Returns:
            Dict[str, Any]: Dictionary with OHLCV fields and timestamp
        """
        return {
            'open': market_event.open,
            'high': market_event.high,
            'low': market_event.low,
            'close': market_event.close,
            'volume': market_event.volume,
            'timestamp': market_event.timestamp
        }
    
    def _get_indicator_value(self, symbol: str, indicator_name: str) -> Any:
        """
        Get the current value of a streaming indicator.
        
        Args:
            symbol (str): Symbol to get indicator for
            indicator_name (str): Name of the indicator
            
        Returns:
            Any: Current indicator value, or None if not available
        """
        if symbol in self.indicators and indicator_name in self.indicators[symbol]:
            indicator = self.indicators[symbol][indicator_name]
            return indicator.value if indicator.is_ready else None
        return None
    
    def _is_indicator_ready(self, symbol: str, indicator_name: str) -> bool:
        """
        Check if a streaming indicator is ready (has sufficient data).
        
        Args:
            symbol (str): Symbol to check
            indicator_name (str): Name of the indicator
            
        Returns:
            bool: True if indicator is ready, False otherwise
        """
        if symbol in self.indicators and indicator_name in self.indicators[symbol]:
            indicator = self.indicators[symbol][indicator_name]
            return indicator.is_ready
        return False
    
    def _get_indicator_history(self, symbol: str, indicator_name: str, n: int = 10) -> List[Any]:
        """
        Get historical values from a streaming indicator.
        
        Args:
            symbol (str): Symbol to get history for
            indicator_name (str): Name of the indicator
            n (int): Number of historical values to retrieve
            
        Returns:
            List[Any]: List of historical values, empty if not available
        """
        if symbol in self.indicators and indicator_name in self.indicators[symbol]:
            indicator = self.indicators[symbol][indicator_name]
            return indicator.get_history(n)
        return []
    
    # Position-aware helper methods for strategies
    def has_position(self, position_context: Optional[Dict]) -> bool:
        """
        Check if position context indicates an existing position.
        
        Args:
            position_context: Position context dictionary or None if no position exists
            
        Returns:
            bool: True if position exists, False otherwise
        """
        return position_context is not None
    
    def get_position_pnl_pct(self, position_context: Dict) -> float:
        """
        Get position P&L as percentage. Safe accessor with validation.
        
        Args:
            position_context: Position context dictionary
            
        Returns:
            float: P&L as percentage (e.g., -0.03 for -3%)
            
        Raises:
            ValueError: If position context is None or missing required fields
        """
        if not position_context:
            raise ValueError("Position context is None")
        
        pnl_pct = position_context.get('unrealized_pnl_pct')
        if pnl_pct is None:
            raise ValueError("Position context missing 'unrealized_pnl_pct' field")
        
        return float(pnl_pct)
    
    def get_bars_held(self, position_context: Dict) -> int:
        """
        Get number of bars since position entry.
        
        Args:
            position_context: Position context dictionary
            
        Returns:
            int: Number of bars held
            
        Raises:
            ValueError: If position context is None or missing required fields
        """
        if not position_context:
            raise ValueError("Position context is None")
        
        bars_held = position_context.get('bars_since_entry')
        if bars_held is None:
            raise ValueError("Position context missing 'bars_since_entry' field")
        
        return int(bars_held)
    
    def is_profitable(self, position_context: Dict) -> bool:
        """
        Check if position is currently profitable.
        
        Args:
            position_context: Position context dictionary
            
        Returns:
            bool: True if profitable, False otherwise
        """
        return self.get_position_pnl_pct(position_context) > 0
    
    def is_loss_exceeding(self, position_context: Dict, threshold: float) -> bool:
        """
        Check if loss exceeds threshold percentage.
        
        Args:
            position_context: Position context dictionary
            threshold: Loss threshold as positive decimal (e.g., 0.05 for 5%)
            
        Returns:
            bool: True if loss exceeds threshold, False otherwise
        """
        return self.get_position_pnl_pct(position_context) < -abs(threshold)
    
