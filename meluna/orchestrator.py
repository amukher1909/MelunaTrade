import queue
import time
from meluna.events import Event, LiquidationEvent
import logging

logger = logging.getLogger(__name__)


class PositionContextError(Exception):
    """Position context retrieval failed."""
    pass


class SystemConfigurationError(Exception):
    """System components not configured."""
    pass


class InvalidPositionContextError(Exception):
    """Invalid position context structure."""
    pass

class Orchestrator:
    """Main event loop managing event queue and processing."""
    def __init__(self):
        """Initialize orchestrator and event sequencer."""
        self.event_queue = queue.PriorityQueue()
        self.running = True
        self.strategies = []
        self.portfolio = None
        self.execution_handler = None
        self.pending_orders = {}  # To track pending orders

        # The orchestrator is now responsible for sequencing all events
        self.sequence_counter = 0

    def register_strategy(self, strategy):
        """Add strategy to orchestrator."""
        self.strategies.append(strategy)

    def register_portfolio(self, portfolio):
        """Link portfolio to orchestrator."""
        self.portfolio = portfolio

    def register_execution_handler(self, handler):
        self.execution_handler = handler

    def put_event(self, event: Event, priority: int = None):
        """Add event to queue with priority and sequence number."""
        # Use event's intrinsic priority if not overridden
        if priority is None:
            priority = event.priority

        self.sequence_counter += 1
        self.event_queue.put((event.timestamp, priority, self.sequence_counter, event))

    def run(self):
        """Run main event loop."""
        print("Starting main event loop...")
        while self.running:
            try:
                # Unpack the 4-element tuple
                timestamp, priority, seq, event = self.event_queue.get(block=True, timeout=1)
            except queue.Empty:
                print("Event queue is empty. Backtest finished.")
                self.running = False
                continue
            
            # --- EVENT DISPATCH LOGIC ---
            if event.event_type == 'MARKET':
                # First, execute any pending orders using this new market data
                if self.portfolio:
                    self.portfolio.on_market_data(event)
                if event.symbol in self.pending_orders:
                    order_to_execute = self.pending_orders.pop(event.symbol)
                    fill_event = self.execution_handler.simulate_fill(order_to_execute, event)
                    self.put_event(fill_event)  # Use event's intrinsic priority

                # CRITICAL: Check for liquidations immediately after market data update
                # Liquidations must trigger before any new orders are placed
                if self.portfolio and hasattr(self.portfolio, 'check_liquidations'):
                    # Build current prices dict from latest_prices in portfolio
                    current_prices = self.portfolio.latest_prices.copy()

                    # Check for liquidations
                    liquidation_events = self.portfolio.check_liquidations(
                        current_prices=current_prices,
                        current_timestamp=event.timestamp
                    )

                    # Add liquidation events to queue (will process before orders due to priority)
                    for liq_event in liquidation_events:
                        self.put_event(liq_event)  # Uses LIQUIDATION priority (2)
                        logger.warning(
                            f"Liquidation event queued: {liq_event.symbol} "
                            f"position_id={liq_event.position_id}"
                        )

                # Get position context for this symbol (returns None if no position exists)
                symbol = event.symbol

                if not self.portfolio:
                    raise SystemConfigurationError("Portfolio not initialized in Orchestrator")

                try:
                    position_context = self.portfolio.get_position_context(symbol)
                except Exception as e:
                    raise PositionContextError(f"Failed to retrieve position context for {symbol}: {e}")

                # Then, send the market event to strategies with position context
                for strategy in self.strategies:
                    try:
                        signals = strategy.on_market_data(event, position_context)
                        for signal in signals:
                            # Use event's intrinsic priority (signals have lowest priority)
                            self.put_event(signal)
                    except Exception as e:
                        # Isolate strategy failures - continue with other strategies
                        print(f"Strategy {strategy.__class__.__name__} failed processing {symbol}: {e}")
                        continue

            elif event.event_type == 'LIQUIDATION':
                # Liquidation events are already processed by _force_liquidate_position
                # in Portfolio.check_liquidations(). Just log the event here.
                logger.error(
                    f"LIQUIDATION PROCESSED: {event.symbol} position {event.position_id} "
                    f"liquidated at ${event.liquidation_price:.2f} "
                    f"(mark: ${event.mark_price:.2f}, P&L: ${event.pnl:.2f})"
                )
                # Note: Position is already closed and margin returned by Portfolio

            elif event.event_type == 'SIGNAL':
                if self.portfolio:
                    orders = self.portfolio.on_signal(event)
                    for order in orders:
                        self.put_event(order)  # Use event's intrinsic priority

            elif event.event_type == 'ORDER':
                # Hold the order until the next market event for that symbol
                print(f"  Order Captured: {event.direction} on {event.symbol}. Holding for T+1 execution.")
                self.pending_orders[event.symbol] = event

            elif event.event_type == 'FILL':
                # Send the fill event to the portfolio to update state
                if self.portfolio:
                    self.portfolio.on_fill(event)
            
        print("Main event loop finished.")