# meluna/events.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Dict, Any, Optional, List
from enum import IntEnum
from decimal import Decimal

# --- Position Management Exceptions -----------------------------------
class InvalidPositionError(Exception):
    """Position not found in portfolio."""
    
    def __init__(self, position_id: str, available_positions: List[str] = None):
        if available_positions:
            available_str = ", ".join(available_positions)
            message = f"Position ID '{position_id}' not found. Available positions: {available_str}"
        else:
            message = f"Position ID '{position_id}' not found"
        super().__init__(message)
        self.position_id = position_id
        self.available_positions = available_positions or []


class InvalidAdjustmentError(Exception):
    """Invalid position adjustment parameters."""
    
    def __init__(self, parameter_name: str, value: Any, reason: str):
        message = f"Invalid adjustment parameter '{parameter_name}': {value} ({reason})"
        super().__init__(message)
        self.parameter_name = parameter_name
        self.value = value
        self.reason = reason


class PositionStateError(Exception):
    """Position in wrong state for requested operation."""

    def __init__(self, position_id: str, current_state: str, required_state: str):
        message = f"Position '{position_id}' in state '{current_state}', requires '{required_state}' for adjustment"
        super().__init__(message)
        self.position_id = position_id
        self.current_state = current_state
        self.required_state = required_state


class InsufficientMarginError(Exception):
    """Insufficient margin for futures position."""

    def __init__(self, required_margin: float, available_margin: float, symbol: str):
        shortfall = required_margin - available_margin
        message = (
            f"Insufficient margin for {symbol}: "
            f"Required ${required_margin:.2f}, "
            f"Available ${available_margin:.2f}, "
            f"Shortfall ${shortfall:.2f}"
        )
        super().__init__(message)
        self.required_margin = required_margin
        self.available_margin = available_margin
        self.symbol = symbol
        self.shortfall = shortfall

# --- Event Priority System ----------------------------------------
class EventPriority(IntEnum):
    """Event processing priority levels."""
    MARKET = 1       # Price updates must happen first
    LIQUIDATION = 2  # Forced closures (involuntary, instant)
    ORDER = 3        # Strategy orders (voluntary)
    FILL = 4         # Order executions
    SIGNAL = 5       # Strategy signals

# --- Base Event --------------------------------------------------
class Event:
    """Base event class with priority support."""
    @property
    def priority(self) -> int:
        """Event priority level."""
        return EventPriority.SIGNAL
# --- Market Events -----------------------------------------------
@dataclass
class MarketEvent(Event):
    """Market data bar for a symbol."""
    # Required fields come first
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    # Default field comes last
    event_type: str = "MARKET"

    @property
    def priority(self) -> int:
        """Highest priority."""
        return EventPriority.MARKET

# --- Signal Events -----------------------------------------------
@dataclass
class SignalEvent(Event):
    """Trading signal from strategy."""
    # Required fields come first
    symbol: str
    timestamp: datetime
    direction: Literal['LONG', 'SHORT', 'EXIT', 'ADJUST']
    # Optional fields for position management
    position_id: Optional[str] = None  # Links to Position.trade_id
    # Default fields come last
    confidence_score: float = 1.0
    management_params: Dict[str, Any] = field(default_factory=dict)
    event_type: str = "SIGNAL"

    @property
    def priority(self) -> int:
        """Lowest priority."""
        return EventPriority.SIGNAL

    def __repr__(self) -> str:
        """String representation."""
        position_str = f", pos_id={self.position_id}" if self.position_id else ""
        return (f"SignalEvent({self.symbol}, {self.direction}, "
                f"ts={self.timestamp.strftime('%Y-%m-%d %H:%M')}, "
                f"conf={self.confidence_score}{position_str})")

# --- Order Events ------------------------------------------------
@dataclass
class OrderEvent(Event):
    """Order for execution system."""
    # Required fields come first
    symbol: str
    timestamp: datetime
    quantity: int
    direction: Literal['BUY', 'SELL']
    # Default field comes last
    order_type: Literal['MKT', 'LMT'] = 'MKT'
    event_type: str = "ORDER"

    @property
    def priority(self) -> int:
        """Process after liquidations."""
        return EventPriority.ORDER

# --- Fill Events -------------------------------------------------
@dataclass
class FillEvent(Event):
    """Order fill from execution system."""
    # Required fields come first
    symbol: str
    timestamp: datetime
    quantity: int
    direction: Literal['BUY', 'SELL']
    fill_price: float
    # Default field comes last
    commission: float = 0.0
    event_type: str = "FILL"

    @property
    def priority(self) -> int:
        """Process after orders."""
        return EventPriority.FILL

# --- Liquidation Events ------------------------------------------
@dataclass
class LiquidationEvent(Event):
    """Forced position closure due to margin breach."""
    # Required fields come first
    symbol: str
    position_id: str
    liquidation_price: float
    mark_price: float
    pnl: float
    remaining_margin: float
    liquidation_fee: float
    timestamp: datetime
    # Default field comes last
    event_type: str = "LIQUIDATION"

    @property
    def priority(self) -> int:
        """High priority, process before orders."""
        return EventPriority.LIQUIDATION

    def __repr__(self) -> str:
        """String representation."""
        return (f"LiquidationEvent({self.symbol}, pos_id={self.position_id}, "
                f"liq_price={self.liquidation_price:.2f}, "
                f"mark={self.mark_price:.2f}, "
                f"PnL={self.pnl:.2f}, "
                f"remaining=${self.remaining_margin:.2f}, "
                f"fee=${self.liquidation_fee:.2f})")

