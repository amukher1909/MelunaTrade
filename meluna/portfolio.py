from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
from meluna.events import SignalEvent, OrderEvent, FillEvent, MarketEvent, LiquidationEvent, InvalidPositionError, InvalidAdjustmentError, PositionStateError, InsufficientMarginError
#from meluna.portfolio import Position

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Single asset holding with management capabilities."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    quantity: int
    entry_price: float
    entry_timestamp: pd.Timestamp  # To know when the trade was opened

    # Unified trade ID system (replaces position_id concept)
    trade_id: str = field(default="")

    # Management fields for position control
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None

    # Position tracking metrics
    highest_price_since_entry: Optional[float] = None
    lowest_price_since_entry: Optional[float] = None
    bars_since_entry: int = 0

    # Strategy-specific management parameters
    management_params: Dict[str, Any] = field(default_factory=dict)

    # Existing calculated fields
    market_value: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Futures-specific fields (Issue #136 - EPIC_002 Phase 1)
    # Defaults preserve equity behavior (1x leverage, cash mode, no margin)
    leverage: float = 1.0  # 1x leverage = no leverage (equity equivalent)
    margin_mode: str = 'cash'  # 'cash' (equity), 'isolated', or 'cross' (future)
    initial_margin: float = 0.0  # Only used when margin_mode != 'cash'
    liquidation_price: Optional[float] = None  # Calculated in Issue #138 (EPIC_002 Phase 4)
    
    @property
    def position_id(self) -> str:
        """Trade ID for API consistency."""
        return self.trade_id

    def __repr__(self) -> str:
        """String representation with trade ID and management fields."""
        mgmt_info = []
        if self.stop_loss:
            mgmt_info.append(f"SL:{self.stop_loss:.2f}")
        if self.take_profit:
            mgmt_info.append(f"TP:{self.take_profit:.2f}")
        if self.trailing_stop_distance:
            mgmt_info.append(f"TS:{self.trailing_stop_distance:.2f}")
        
        mgmt_str = f" [{', '.join(mgmt_info)}]" if mgmt_info else ""
        
        return (f"Position({self.trade_id}: {self.direction} {self.quantity} {self.symbol} "
                f"@{self.entry_price:.2f}, PnL:{self.unrealized_pnl:.2f}{mgmt_str})")

class Portfolio:
    """Central state management for assets, cash, and positions."""

    # Valid portfolio modes (Issue #136)
    VALID_MODES = {'equity', 'futures'}

    # Maximum leverage constant (Issue #140 - EPIC_002 Phase 5)
    MAX_LEVERAGE = 125.0  # Industry standard maximum (Binance, Bybit)

    def __init__(self, initial_cash: float, mode: str = 'equity', liquidation_fee_rate: float = 0.0, default_leverage: float = 1.0):
        """Initialize portfolio with starting capital and trading mode."""
        # Validate mode (Issue #136)
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid portfolio mode '{mode}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_MODES))}"
            )

        # Validate leverage (Issue #140)
        self._validate_leverage(default_leverage)

        self.mode = mode
        self._default_leverage = default_leverage
        self.initial_cash = initial_cash
        self.current_cash = initial_cash

        # The positions dictionary holds all open Position objects, keyed by symbol.
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[Dict] = []

        # Unified ID system: daily trade counters for T{YYYYMMDD}-{NNN}-{SYMBOL} format
        self.daily_trade_counters: Dict[str, int] = {}  # {YYYYMMDD: counter}
        self.equity_curve: pd.Series = pd.Series(name='EquityCurve', dtype=float)
        self.latest_prices: Dict[str, float] = {}

        # Futures-specific fields (Issue #136 - EPIC_002 Phase 1)
        # Only initialized when mode='futures'
        if mode == 'futures':
            self.used_margin = 0.0  # Margin currently locked in open positions
            self.available_margin = initial_cash  # Margin available for new positions
            self.total_equity = initial_cash  # For margin equation validation (Issue #137)
            self.liquidated_positions: List[Dict] = []  # Track liquidation history
            self.maintenance_margin_rate = 0.004  # 0.4% (Binance default for most symbols)
            self.liquidation_fee_rate = liquidation_fee_rate  # Issue #139 - Phase 4
            logger.info(f"Portfolio initialized in FUTURES mode with margin: {initial_cash:.2f}, liquidation_fee_rate: {liquidation_fee_rate:.2%}")
        else:
            logger.info(f"Portfolio initialized in EQUITY mode with cash: {initial_cash:.2f}")

    def _generate_unified_id(self, symbol: str, timestamp: pd.Timestamp) -> str:
        """Generate unique trade ID."""
        date_key = timestamp.strftime('%Y%m%d')
        if date_key not in self.daily_trade_counters:
            self.daily_trade_counters[date_key] = 0
        self.daily_trade_counters[date_key] += 1
        return f"T{date_key}-{self.daily_trade_counters[date_key]:03d}-{symbol}"

    def _validate_leverage(self, leverage: float) -> None:
        """Validate leverage within acceptable range."""
        if not (1.0 <= leverage <= self.MAX_LEVERAGE):
            raise ValueError(
                f"Leverage {leverage}x out of range. "
                f"Must be between 1.0x and {self.MAX_LEVERAGE}x. "
                f"Received: {leverage}x"
            )

    @property
    def default_leverage(self) -> float:
        """Default leverage for new positions."""
        return self._default_leverage

    def _calculate_required_margin(self, quantity: int, price: float, leverage: float = 1.0) -> float:
        """Calculate margin required for position."""
        position_value = abs(quantity) * price
        return position_value / leverage

    def _calculate_liquidation_price(self, entry_price: float, direction: str, leverage: float) -> float:
        """Calculate liquidation price using Binance formula."""
        # TODO: Make MMR configurable per-symbol in Issue #140+ (Binance uses tiered MMR)
        mmr = self.maintenance_margin_rate  # 0.004 = 0.4% (from Issue #136)

        if direction == 'LONG':
            # LONG liquidation: price drops below entry
            liq_price = entry_price * (1 - 1/leverage + mmr)
        elif direction == 'SHORT':
            # SHORT liquidation: price rises above entry
            liq_price = entry_price * (1 + 1/leverage - mmr)
        else:
            raise ValueError(f"Invalid direction '{direction}'. Must be 'LONG' or 'SHORT'.")

        return liq_price

    def check_liquidations(self, current_prices: Dict[str, float], current_timestamp: pd.Timestamp) -> List[LiquidationEvent]:
        """Check positions for liquidation conditions."""
        liquidation_events = []

        # Only check liquidations in futures mode
        if self.mode != 'futures':
            return liquidation_events

        # Iterate through all open positions
        for symbol, position in list(self.positions.items()):
            # Skip if no liquidation price set (shouldn't happen in futures mode)
            if position.liquidation_price is None:
                continue

            # Get current price for this symbol
            current_price = current_prices.get(symbol)
            if current_price is None:
                # No price data available, skip this position
                continue

            # Check liquidation condition based on direction
            is_liquidated = False
            if position.direction == 'LONG':
                # LONG liquidates when price drops to or below liquidation price
                is_liquidated = current_price <= position.liquidation_price
            elif position.direction == 'SHORT':
                # SHORT liquidates when price rises to or above liquidation price
                is_liquidated = current_price >= position.liquidation_price

            if is_liquidated:
                # Create liquidation event for this position
                liquidation_event = self._force_liquidate_position(
                    position,
                    mark_price=current_price,
                    timestamp=current_timestamp
                )
                liquidation_events.append(liquidation_event)

                logger.warning(
                    f"LIQUIDATION TRIGGERED: {symbol} {position.direction} position "
                    f"(ID: {position.trade_id}) at mark_price=${current_price:.2f}, "
                    f"liquidation_price=${position.liquidation_price:.2f}"
                )

        return liquidation_events

    def _force_liquidate_position(
        self,
        position: Position,
        mark_price: float,
        timestamp: pd.Timestamp
    ) -> LiquidationEvent:
        """Force-close position and return liquidation event."""
        symbol = position.symbol

        # Close position at liquidation price (not mark price!)
        # This is the exchange's guaranteed behavior
        liquidation_price = position.liquidation_price

        # Calculate P&L at liquidation price
        if position.direction == 'LONG':
            # LONG: bought at entry, sell at liquidation
            pnl = (liquidation_price - position.entry_price) * position.quantity
        else:  # SHORT
            # SHORT: sold at entry, buy back at liquidation
            pnl = (position.entry_price - liquidation_price) * position.quantity

        # Calculate remaining margin before fee
        # Initial margin was locked, now we see what's left after loss
        remaining_margin_before_fee = position.initial_margin + pnl

        # Calculate liquidation fee (charged on remaining margin)
        liquidation_fee = remaining_margin_before_fee * self.liquidation_fee_rate

        # Final remaining margin after fee
        remaining_margin = remaining_margin_before_fee - liquidation_fee

        # Ensure remaining margin is not negative (account protection)
        if remaining_margin < 0:
            logger.error(
                f"CRITICAL: Negative remaining margin detected for {symbol}! "
                f"remaining={remaining_margin:.2f}, setting to 0"
            )
            remaining_margin = 0.0

        # Return remaining margin to account (update margin accounting)
        self.available_margin += remaining_margin
        self.used_margin -= position.initial_margin

        # Update total equity (loses the initial_margin - remaining_margin)
        loss = position.initial_margin - remaining_margin
        self.total_equity -= loss

        # Create liquidation event
        liquidation_event = LiquidationEvent(
            symbol=symbol,
            position_id=position.trade_id,
            liquidation_price=liquidation_price,
            mark_price=mark_price,
            pnl=pnl,
            remaining_margin=remaining_margin,
            liquidation_fee=liquidation_fee,
            timestamp=timestamp
        )

        # Log liquidation to trade history
        liquidation_record = {
            'trade_id': position.trade_id,
            'symbol': symbol,
            'direction': position.direction,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': liquidation_price,
            'entry_timestamp': position.entry_timestamp,
            'exit_timestamp': timestamp,
            'pnl': pnl,
            'bars_held': position.bars_since_entry,
            'exit_reason': 'LIQUIDATION',
            'leverage': position.leverage,
            'initial_margin': position.initial_margin,
            'liquidation_price': liquidation_price,
            'mark_price_at_liquidation': mark_price,
            'liquidation_fee': liquidation_fee,
            'remaining_margin': remaining_margin
        }
        self.trade_log.append(liquidation_record)
        self.liquidated_positions.append(liquidation_record)

        # Remove position from active positions
        del self.positions[symbol]

        # Validate margin equation after liquidation
        self._validate_margin_equation()

        # Log detailed liquidation info
        logger.error(
            f"POSITION LIQUIDATED: {symbol} {position.direction} | "
            f"ID: {position.trade_id} | "
            f"Entry: ${position.entry_price:.2f} | "
            f"Liq Price: ${liquidation_price:.2f} | "
            f"Mark Price: ${mark_price:.2f} | "
            f"P&L: ${pnl:.2f} | "
            f"Fee: ${liquidation_fee:.2f} | "
            f"Remaining: ${remaining_margin:.2f} | "
            f"Leverage: {position.leverage}x"
        )

        return liquidation_event

    def _validate_margin_available(self, required_margin: float, symbol: str) -> None:
        """Check if sufficient margin available."""
        if self.mode == 'futures' and required_margin > self.available_margin:
            raise InsufficientMarginError(required_margin, self.available_margin, symbol)

    def _validate_margin_equation(self) -> None:
        """Validate margin equation integrity."""
        if self.mode == 'futures':
            calculated_total = self.used_margin + self.available_margin
            tolerance = 0.01  # Allow $0.01 floating point error

            if abs(calculated_total - self.total_equity) > tolerance:
                raise AssertionError(
                    f"MARGIN EQUATION VIOLATED! "
                    f"used_margin ({self.used_margin:.2f}) + "
                    f"available_margin ({self.available_margin:.2f}) = "
                    f"{calculated_total:.2f}, but total_equity = {self.total_equity:.2f}. "
                    f"Difference: {abs(calculated_total - self.total_equity):.4f}"
                )

    def on_signal(self, signal: SignalEvent) -> List[OrderEvent]:
        """Convert signal to order events."""
        logger.info(f"Portfolio received signal: {signal.direction} on {signal.symbol}")
        orders = []
        fixed_trade_quantity = 10

        if signal.direction == 'LONG' and signal.symbol not in self.positions:
            # Futures mode: check margin before generating order
            if self.mode == 'futures':
                estimated_price = self.latest_prices.get(signal.symbol, 0.0)
                if estimated_price == 0.0:
                    logger.warning(
                        f"Cannot estimate margin requirement for {signal.symbol}: "
                        f"no price data available. Skipping order generation."
                    )
                    return []

                required_margin = self._calculate_required_margin(
                    fixed_trade_quantity,
                    estimated_price,
                    leverage=1.0
                )

                try:
                    self._validate_margin_available(required_margin, signal.symbol)
                except InsufficientMarginError as e:
                    logger.warning(f"LONG signal rejected - {e}")
                    return []

            order = OrderEvent(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                quantity=fixed_trade_quantity,
                direction='BUY'
            )
            orders.append(order)
            logger.info(f"Generated BUY order for {fixed_trade_quantity} shares of {signal.symbol}")

        elif signal.direction == 'SHORT' and signal.symbol not in self.positions:
            # Futures mode: check margin before generating order
            if self.mode == 'futures':
                estimated_price = self.latest_prices.get(signal.symbol, 0.0)
                if estimated_price == 0.0:
                    logger.warning(
                        f"Cannot estimate margin requirement for {signal.symbol}: "
                        f"no price data available. Skipping order generation."
                    )
                    return []

                required_margin = self._calculate_required_margin(
                    fixed_trade_quantity,
                    estimated_price,
                    leverage=1.0
                )

                try:
                    self._validate_margin_available(required_margin, signal.symbol)
                except InsufficientMarginError as e:
                    logger.warning(f"SHORT signal rejected - {e}")
                    return []

            order = OrderEvent(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                quantity=fixed_trade_quantity,
                direction='SELL'
            )
            orders.append(order)
            logger.info(f"Generated SELL order for SHORT position: {fixed_trade_quantity} shares of {signal.symbol}")
            
        elif signal.direction == 'EXIT' and signal.symbol in self.positions:
            position = self.positions[signal.symbol]
            current_quantity = position.quantity
            if current_quantity > 0:
                # Direction-aware exit: LONG exits with SELL, SHORT exits with BUY
                exit_direction = 'SELL' if position.direction == 'LONG' else 'BUY'
                order = OrderEvent(
                    symbol=signal.symbol,
                    timestamp=signal.timestamp,
                    quantity=current_quantity,
                    direction=exit_direction
                )
                orders.append(order)
                logger.info(f"Generated {exit_direction} order to close {position.direction} position: {current_quantity} shares of {signal.symbol}")
                
        elif signal.direction == 'ADJUST':
            # Handle position adjustment operations
            try:
                self._handle_adjust_signal(signal)
                logger.info(f"Position adjustment applied for {signal.position_id or signal.symbol}: {signal.management_params}")
            except (InvalidPositionError, InvalidAdjustmentError, PositionStateError) as e:
                logger.error(f"Position adjustment failed: {e}")

        return orders

    def _handle_adjust_signal(self, signal: SignalEvent) -> None:
        """Handle position adjustment signal."""
        # Find position by position_id or symbol
        position = None
        if signal.position_id:
            # Search by position_id (trade_id)
            for pos in self.positions.values():
                if pos.trade_id == signal.position_id:
                    position = pos
                    break
            if not position:
                available_positions = [pos.trade_id for pos in self.positions.values()]
                raise InvalidPositionError(signal.position_id, available_positions)
        else:
            # Fall back to symbol-based lookup
            if signal.symbol in self.positions:
                position = self.positions[signal.symbol]
            else:
                available_symbols = list(self.positions.keys())
                raise InvalidPositionError(signal.symbol, available_symbols)
        
        # Validate and apply adjustment parameters
        for param_name, param_value in signal.management_params.items():
            if param_name == 'new_stop_loss':
                if not isinstance(param_value, (int, float)):
                    raise InvalidAdjustmentError(param_name, param_value, "must be a number")
                if param_value <= 0:
                    raise InvalidAdjustmentError(param_name, param_value, "must be positive")
                
                # Contextual validation for position direction
                if position.direction == 'LONG' and param_value >= position.entry_price:
                    raise InvalidAdjustmentError(param_name, param_value, 
                        f"stop loss ({param_value}) must be below entry price ({position.entry_price}) for LONG position")
                elif position.direction == 'SHORT' and param_value <= position.entry_price:
                    raise InvalidAdjustmentError(param_name, param_value, 
                        f"stop loss ({param_value}) must be above entry price ({position.entry_price}) for SHORT position")
                
                position.stop_loss = float(param_value)
                logger.debug(f"Updated stop_loss for {position.trade_id}: {param_value}")
                
            elif param_name == 'new_take_profit':
                if not isinstance(param_value, (int, float)):
                    raise InvalidAdjustmentError(param_name, param_value, "must be a number")
                if param_value <= 0:
                    raise InvalidAdjustmentError(param_name, param_value, "must be positive")
                
                # Contextual validation for position direction  
                if position.direction == 'LONG' and param_value <= position.entry_price:
                    raise InvalidAdjustmentError(param_name, param_value, 
                        f"take profit ({param_value}) must be above entry price ({position.entry_price}) for LONG position")
                elif position.direction == 'SHORT' and param_value >= position.entry_price:
                    raise InvalidAdjustmentError(param_name, param_value, 
                        f"take profit ({param_value}) must be below entry price ({position.entry_price}) for SHORT position")
                
                position.take_profit = float(param_value)
                logger.debug(f"Updated take_profit for {position.trade_id}: {param_value}")
                
            elif param_name == 'trailing_distance':
                if not isinstance(param_value, (int, float)) or param_value <= 0:
                    raise InvalidAdjustmentError(param_name, param_value, "must be positive number")
                position.trailing_stop_distance = float(param_value)
                logger.debug(f"Updated trailing_stop_distance for {position.trade_id}: {param_value}")
                
            elif param_name == 'partial_exit_ratio':
                if not isinstance(param_value, (int, float)) or not (0 < param_value <= 1):
                    raise InvalidAdjustmentError(param_name, param_value, "must be between 0 and 1")
                # Note: Partial exits would require order generation, not implemented in this phase
                logger.info(f"Partial exit ratio {param_value} noted for {position.trade_id} (not yet implemented)")
                
            elif param_name == 'adjustment_reason':
                # Store reason in position management params for analysis
                position.management_params['last_adjustment_reason'] = param_value
                position.management_params['last_adjustment_timestamp'] = signal.timestamp
                logger.debug(f"Adjustment reason logged for {position.trade_id}: {param_value}")
                
            else:
                # Allow custom parameters to be stored in position management_params
                position.management_params[param_name] = param_value
                logger.debug(f"Custom adjustment parameter for {position.trade_id}: {param_name}={param_value}")

    # --- Fill Event Handling ---
    def on_fill(self, fill_event: FillEvent):
        """Update portfolio state on fill."""
        if self.mode == 'equity':
            self._handle_equity_fill(fill_event)
        else:  # futures mode
            self._handle_futures_fill(fill_event)
            # CRITICAL: Validate margin equation after every futures operation
            self._validate_margin_equation()

    def _handle_equity_fill(self, fill_event: FillEvent):
        """Handle fill in equity mode."""
        trade_value = fill_event.fill_price * fill_event.quantity

        if fill_event.direction == 'BUY':
            if fill_event.symbol in self.positions:
                # Closing SHORT position
                closed_position = self.positions[fill_event.symbol]

                # Direction-aware P&L (should be SHORT)
                if closed_position.direction == 'LONG':
                    pnl = (fill_event.fill_price - closed_position.entry_price) * closed_position.quantity
                else:  # SHORT
                    pnl = (closed_position.entry_price - fill_event.fill_price) * closed_position.quantity
                pnl -= fill_event.commission

                # Create trade record
                trade_record = {
                    'trade_id': closed_position.trade_id,
                    'symbol': fill_event.symbol,
                    'direction': closed_position.direction,
                    'entry_timestamp': closed_position.entry_timestamp,
                    'exit_timestamp': fill_event.timestamp,
                    'entry_price': closed_position.entry_price,
                    'exit_price': fill_event.fill_price,
                    'quantity': closed_position.quantity,
                    'pnl': pnl,
                    'bars_held': closed_position.bars_since_entry
                }
                self.trade_log.append(trade_record)
                logger.info(f"Closed trade logged: {trade_record}")

                # SHORT close: pay cash
                self.current_cash -= (trade_value + fill_event.commission)

                del self.positions[fill_event.symbol]
                logger.info(f"Portfolio state updated on BUY (SHORT CLOSE): Position {fill_event.symbol} removed. Cash at {self.current_cash:.2f}")
            else:
                # Opening LONG position
                trade_id = self._generate_unified_id(fill_event.symbol, fill_event.timestamp)

                # Determine margin mode based on portfolio mode (Issue #136)
                margin_mode = 'isolated' if self.mode == 'futures' else 'cash'
                leverage = 1.0  # Default to 1x leverage (Issue #137 will add configurable leverage)

                new_position = Position(
                    symbol=fill_event.symbol,
                    direction='LONG',
                    quantity=fill_event.quantity,
                    entry_price=fill_event.fill_price,
                    entry_timestamp=fill_event.timestamp,
                    trade_id=trade_id,
                    highest_price_since_entry=fill_event.fill_price,
                    lowest_price_since_entry=fill_event.fill_price,
                    market_value=trade_value,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    # Futures-specific fields (Issue #136)
                    leverage=leverage,
                    margin_mode=margin_mode,
                    initial_margin=0.0,  # Will be calculated in Issue #137
                    liquidation_price=None  # Will be calculated in Issue #138
                )
                self.positions[fill_event.symbol] = new_position
                # LONG open: pay cash (margin accounting in Issue #137)
                self.current_cash -= (trade_value + fill_event.commission)
                logger.info(f"Portfolio state updated on BUY (LONG OPEN): Position {fill_event.symbol} added. Cash at {self.current_cash:.2f}")

        elif fill_event.direction == 'SELL':
            if fill_event.symbol in self.positions:
                # 1. Get the position to close
                closed_position = self.positions[fill_event.symbol]

                # 2. Calculate the Profit and Loss (P&L) for the trade (direction-aware)
                if closed_position.direction == 'LONG':
                    pnl = (fill_event.fill_price - closed_position.entry_price) * closed_position.quantity
                else:  # SHORT
                    pnl = (closed_position.entry_price - fill_event.fill_price) * closed_position.quantity
                pnl -= fill_event.commission
                
                # 3. Create the trade record using the unified trade ID from position
                trade_record = {
                    'trade_id': closed_position.trade_id,  # Use unified ID from position
                    'symbol': fill_event.symbol,
                    'direction': closed_position.direction,
                    'entry_timestamp': closed_position.entry_timestamp,
                    'exit_timestamp': fill_event.timestamp,
                    'entry_price': closed_position.entry_price,
                    'exit_price': fill_event.fill_price,
                    'quantity': closed_position.quantity,
                    'pnl': pnl,
                    'bars_held': closed_position.bars_since_entry
                }
                self.trade_log.append(trade_record)
                logger.info(f"Closed trade logged: {trade_record}")
                
                # Hopefully Increase cash
                self.current_cash += (trade_value - fill_event.commission)

                del self.positions[fill_event.symbol]
                logger.info(f"Portfolio state updated on SELL: Position {fill_event.symbol} removed. Cash at {self.current_cash:.2f}")
            else:
                # Opening SHORT position
                trade_id = self._generate_unified_id(fill_event.symbol, fill_event.timestamp)

                # Determine margin mode based on portfolio mode (Issue #136)
                margin_mode = 'isolated' if self.mode == 'futures' else 'cash'
                leverage = 1.0  # Default to 1x leverage (Issue #137 will add configurable leverage)

                new_position = Position(
                    symbol=fill_event.symbol,
                    direction='SHORT',
                    quantity=fill_event.quantity,
                    entry_price=fill_event.fill_price,
                    entry_timestamp=fill_event.timestamp,
                    trade_id=trade_id,
                    highest_price_since_entry=fill_event.fill_price,
                    lowest_price_since_entry=fill_event.fill_price,
                    market_value=trade_value,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    # Futures-specific fields (Issue #136)
                    leverage=leverage,
                    margin_mode=margin_mode,
                    initial_margin=0.0,  # Will be calculated in Issue #137
                    liquidation_price=None  # Will be calculated in Issue #138
                )
                self.positions[fill_event.symbol] = new_position
                # SHORT position: receive cash when opening
                self.current_cash += (trade_value - fill_event.commission)
                logger.info(f"Portfolio state updated on SELL (SHORT OPEN): Position {fill_event.symbol} added. Cash at {self.current_cash:.2f}")

    def _handle_futures_fill(self, fill_event: FillEvent):
        """Handle fill in futures mode."""
        if fill_event.direction == 'BUY':
            if fill_event.symbol in self.positions:
                # Closing SHORT position
                self._close_position_futures(fill_event, 'SHORT')
            else:
                # Opening LONG position
                self._open_position_futures(fill_event, 'LONG')

        elif fill_event.direction == 'SELL':
            if fill_event.symbol in self.positions:
                # Closing LONG position
                self._close_position_futures(fill_event, 'LONG')
            else:
                # Opening SHORT position
                self._open_position_futures(fill_event, 'SHORT')

    def _open_position_futures(self, fill_event: FillEvent, direction: str):
        """Open position in futures mode."""
        leverage = self.default_leverage  # Configurable leverage (Issue #140)

        # Calculate margin required
        required_margin = self._calculate_required_margin(
            fill_event.quantity,
            fill_event.fill_price,
            leverage
        )

        # Validate sufficient margin (raises InsufficientMarginError if not)
        self._validate_margin_available(required_margin, fill_event.symbol)

        # Deduct margin
        self.used_margin += required_margin
        self.available_margin -= required_margin

        # Deduct commission (conservative: reduces available margin and total equity)
        self.available_margin -= fill_event.commission
        self.total_equity -= fill_event.commission

        # Calculate liquidation price (Issue #138)
        liquidation_price = self._calculate_liquidation_price(
            fill_event.fill_price,
            direction,
            leverage
        )

        # Create position with margin tracking
        trade_id = self._generate_unified_id(fill_event.symbol, fill_event.timestamp)
        trade_value = fill_event.fill_price * fill_event.quantity

        new_position = Position(
            symbol=fill_event.symbol,
            direction=direction,
            quantity=fill_event.quantity,
            entry_price=fill_event.fill_price,
            entry_timestamp=fill_event.timestamp,
            trade_id=trade_id,
            highest_price_since_entry=fill_event.fill_price,
            lowest_price_since_entry=fill_event.fill_price,
            market_value=trade_value,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            # Futures fields
            leverage=leverage,
            margin_mode='isolated',
            initial_margin=required_margin,
            liquidation_price=liquidation_price
        )
        self.positions[fill_event.symbol] = new_position

        logger.info(
            f"FUTURES: Opened {direction} position {fill_event.symbol} | "
            f"Margin deducted: ${required_margin:.2f} | "
            f"Commission: ${fill_event.commission:.2f} | "
            f"Liquidation price: ${liquidation_price:.2f} | "
            f"Available margin: ${self.available_margin:.2f}"
        )

    def _close_position_futures(self, fill_event: FillEvent, expected_direction: str):
        """Close position in futures mode."""
        closed_position = self.positions[fill_event.symbol]

        # Validate direction matches
        assert closed_position.direction == expected_direction, (
            f"Position direction mismatch: expected {expected_direction}, "
            f"got {closed_position.direction}"
        )

        # Calculate direction-aware P&L
        if closed_position.direction == 'LONG':
            pnl = (fill_event.fill_price - closed_position.entry_price) * closed_position.quantity
        else:  # SHORT
            pnl = (closed_position.entry_price - fill_event.fill_price) * closed_position.quantity

        # Return margin
        self.used_margin -= closed_position.initial_margin
        self.available_margin += closed_position.initial_margin

        # Apply P&L (can be positive or negative)
        self.total_equity += pnl
        self.available_margin += pnl

        # Deduct commission (conservative)
        self.total_equity -= fill_event.commission
        self.available_margin -= fill_event.commission

        # Log trade
        trade_record = {
            'trade_id': closed_position.trade_id,
            'symbol': fill_event.symbol,
            'direction': closed_position.direction,
            'entry_timestamp': closed_position.entry_timestamp,
            'exit_timestamp': fill_event.timestamp,
            'entry_price': closed_position.entry_price,
            'exit_price': fill_event.fill_price,
            'quantity': closed_position.quantity,
            'pnl': pnl,
            'commission': fill_event.commission,  # Only this close commission
            'leverage': closed_position.leverage,
            'initial_margin': closed_position.initial_margin,
            'margin_mode': closed_position.margin_mode,
            'liquidation_price': closed_position.liquidation_price,  # Issue #138
            'bars_held': closed_position.bars_since_entry
        }
        self.trade_log.append(trade_record)

        # Remove position
        del self.positions[fill_event.symbol]

        logger.info(
            f"FUTURES: Closed {closed_position.direction} position {fill_event.symbol} | "
            f"P&L: ${pnl:.2f} | "
            f"Commission: ${fill_event.commission:.2f} | "
            f"Margin returned: ${closed_position.initial_margin:.2f} | "
            f"Available margin: ${self.available_margin:.2f}"
        )

    def get_position_context(self, symbol: str) -> Optional[Dict]:
        """Get position context for symbol."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        current_price = self.latest_prices.get(symbol, position.entry_price)
        
        # Calculate percentage P&L
        invested_capital = position.entry_price * abs(position.quantity)
        unrealized_pnl_pct = position.unrealized_pnl / invested_capital if invested_capital > 0 else 0.0
        
        return {
            'symbol': symbol,
            'position_id': position.position_id,
            'entry_price': position.entry_price,
            'current_price': current_price,
            'quantity': position.quantity,
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'bars_since_entry': position.bars_since_entry,
            'current_stop_loss': position.stop_loss,
            'management_params': position.management_params.copy()
        }

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve."""
        return self.equity_curve.resample('D').last().dropna()


    def get_trade_log_df(self):
        """Get trade log as DataFrame."""
        print(self.trade_log)
        return pd.DataFrame(self.trade_log)
    
    
    def on_market_data(self, market_event):
        """Update mark-to-market values."""
        if self.equity_curve.empty:
            self.equity_curve.loc[market_event.timestamp] = self.initial_cash
        # First, update our dictionary with the latest price for the incoming symbol.
        self.latest_prices[market_event.symbol] = market_event.close
        
        # Now, calculate the total value of all positions held.
        total_positions_value = 0.0
        for symbol, position in self.positions.items():
            # Use the last known price for each symbol to get its current market value.
            latest_price = self.latest_prices.get(symbol, position.entry_price)
            position.market_value = latest_price * position.quantity

            # Direction-aware unrealized P&L
            if position.direction == 'LONG':
                position.unrealized_pnl = (latest_price - position.entry_price) * position.quantity
            else:  # SHORT
                position.unrealized_pnl = (position.entry_price - latest_price) * position.quantity
            
            # Update position tracking metrics only for the symbol receiving market data
            if symbol == market_event.symbol:
                position.bars_since_entry += 1
                if position.highest_price_since_entry is None or latest_price > position.highest_price_since_entry:
                    position.highest_price_since_entry = latest_price
                if position.lowest_price_since_entry is None or latest_price < position.lowest_price_since_entry:
                    position.lowest_price_since_entry = latest_price
                
            total_positions_value += position.market_value

        # The true equity is our cash plus the value of all our assets.
        total_equity = self.current_cash + total_positions_value
        self.equity_curve.loc[market_event.timestamp] = total_equity
