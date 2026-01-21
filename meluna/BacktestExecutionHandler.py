import logging
from meluna.events import OrderEvent, FillEvent, MarketEvent

logger = logging.getLogger(__name__)

class BacktestExecutionHandler:
    """Simulates order fills for backtesting."""
    def __init__(self, commission_bps: float, slippage_bps: float):
        """Initialize with commission and slippage rates."""
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        logger.info("BacktestExecutionHandler initialized.")

    def simulate_fill(self, order: OrderEvent, next_bar: MarketEvent) -> FillEvent:
        """Simulate order fill using next bar data."""
        fill_price = next_bar.open
        
        # --- Calculate Slippage ---
        # Slippage makes the price worse for us
        if order.direction == 'BUY':
            slippage_cost = fill_price * self.slippage_rate
            fill_price += slippage_cost
        elif order.direction == 'SELL':
            slippage_cost = fill_price * self.slippage_rate
            fill_price -= slippage_cost

        # --- Calculate Commission ---
        trade_value = fill_price * order.quantity
        commission = trade_value * self.commission_rate

        logger.info(f"Simulating fill for {order.direction} {order.quantity} of {order.symbol} at {fill_price:.2f}")

        # Create the FillEvent
        fill_event = FillEvent(
            symbol=order.symbol,
            timestamp=next_bar.timestamp, # Fill occurs at the time of the next bar
            quantity=order.quantity,
            direction=order.direction,
            fill_price=fill_price,
            commission=commission
        )
        
        return fill_event