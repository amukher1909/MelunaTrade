# main.py
import pandas as pd
import logging
import os
import shutil  # Import the shutil library for file copying
from pathlib import Path
from datetime import datetime  # Import the datetime library

# Import custom modules
from meluna.core import ConfigLoader, setup_logging, get_next_version_path
from data.factory import DataHandlerFactory
from meluna.events import MarketEvent
from meluna.orchestrator import Orchestrator
from meluna.portfolio import Portfolio
from meluna.BacktestExecutionHandler import BacktestExecutionHandler
from Strategies.MA_crossover import MovingAverageCrossoverStrategy



def main():
    print("Starting Project Meluna Backtest...")

    # --- 1. System Initialization ---
    # CORRECTED PATH: Use a robust relative path, not a hardcoded absolute path
    project_root = Path(__file__).resolve().parent
    config_path = project_root / 'config.yml'
    config_loader = ConfigLoader(config_path=str(config_path))

    config = config_loader.get_all()
    
    # Setup logging
    log_config = config_loader.get('logging')
    os.makedirs(project_root / os.path.dirname(log_config['handlers']['file']['filename']), exist_ok=True)
    setup_logging(log_config)

    print("\n--- New Backtest Run ---")
    # Skip interactive input for testing
    backtest_notes = "Testing multi-symbol data loading fix"

    # --- Automated Versioning and Setup ---
    backtest_name = config['backtest_settings'].get('backtest_name', 'unnamed_backtest')
    base_results_dir = project_root / 'results'
    run_path = get_next_version_path(base_results_dir, backtest_name)

    logging.info(f"--- Starting Backtest Run: {backtest_name} | Version: {run_path.name} ---")

    # Save notes and config for reproducibility
    notes_path = run_path / 'notes.txt'
    with open(notes_path, 'w') as f:
        f.write(backtest_notes)
    shutil.copy(config_path, run_path / 'config.yml')
    logging.info(f"Saved notes and config to: {run_path}")

    # Initialize main components
    orchestrator = Orchestrator()
    data_handler = DataHandlerFactory(config)

    # --- Initialize and register the Portfolio ---
    logging.info("Initializing Portfolio...")
    initial_cash = config_loader.get('backtest_settings')['initial_aum']

    # Load strategy config to extract leverage setting (Issue #140)
    from configloader import ConfigLoader as StrategyConfigLoader
    strategy_config = StrategyConfigLoader(str(project_root / 'Strategies/moving_average_crossover_config.yaml'))
    default_leverage = strategy_config.get('position_management', {}).get('leverage', 1.0)

    portfolio = Portfolio(
        initial_cash=initial_cash,
        mode='futures',  # Trading perpetual futures
        default_leverage=default_leverage
    )
    orchestrator.register_portfolio(portfolio)

    # --- Initialize and register the Execution Handler ---
    exec_settings = config_loader.get('execution_settings')
    execution_handler = BacktestExecutionHandler(
        commission_bps=exec_settings.get('commission_bps', 0),
        slippage_bps=exec_settings.get('slippage_bps', 0)
    )
    orchestrator.register_execution_handler(execution_handler)
    
    # --- Initialize Strategy with Dedicated Configuration ---
    strategy_config_path = project_root / 'Strategies/moving_average_crossover_config.yaml'
    symbol_list = config_loader.get('data_settings')['symbol_list']
    strategy = MovingAverageCrossoverStrategy(
        config_path=str(strategy_config_path),
        data_handler=data_handler,
        symbol_list=symbol_list
    )
    orchestrator.register_strategy(strategy)

    # --- 2. Data Loading and Event Generation ---
    logging.info("Loading historical data...")
    backtest_settings = config_loader.get('backtest_settings')
    
    all_symbol_data = {} 
    failed_symbols = []
    successful_symbols = []
    
    for symbol in symbol_list:
        logging.info(f"Attempting to load data for {symbol}...")
        try:
            df = data_handler.get_data(
                symbol=symbol,
                start_date=backtest_settings.get('start_date'),
                end_date=backtest_settings.get('end_date'),
                force_redownload=config.get('force_redownload_data', False)
            )

            if df is None or df.empty:
                logging.warning(f"No data returned for {symbol}, it will be skipped.")
                failed_symbols.append(symbol)
                continue
            
            # Store the successfully loaded dataframe in our dictionary.
            all_symbol_data[symbol] = df
            successful_symbols.append(symbol)
            logging.info(f"Successfully loaded {len(df)} data points for {symbol}.")
            
        except Exception as e:
            logging.error(f"Failed to load data for {symbol}: {e}")
            failed_symbols.append(symbol)
            continue
    
    # Report data loading summary
    logging.info(f"Data loading summary: {len(successful_symbols)}/{len(symbol_list)} symbols loaded successfully")
    if successful_symbols:
        logging.info(f"Successfully loaded symbols: {', '.join(successful_symbols)}")
    if failed_symbols:
        logging.warning(f"Failed to load symbols: {', '.join(failed_symbols)}")
    
    # Validate minimum data requirements
    if not all_symbol_data:
        logging.error("FATAL: No symbols loaded successfully. Cannot proceed with backtest.")
        return
    elif len(all_symbol_data) < len(symbol_list):
        logging.warning(f"Proceeding with partial data: {len(all_symbol_data)} out of {len(symbol_list)} symbols loaded.")
    
    # All data is cached and strategy indicators are initialized via streaming approach   
    
    logging.info("Generating market events...")
    
    for symbol,df in all_symbol_data.items():

        if df is None or df.empty:
            logging.warning(f"No data for {symbol} to generate events, skipping.")
            continue
            
        for index, row in df.iterrows():
            event = MarketEvent(
                symbol=symbol, timestamp=row['date'], open=row['open'], high=row['high'],
                low=row['low'], close=row['close'], volume=row['volume']
            )
            # Let the orchestrator handle the sequencing
            orchestrator.put_event(event, priority=5)

    logging.info(f"Successfully loaded {len(orchestrator.event_queue.queue)} market events onto the queue.")

    # --- Run the Backtest ---
    logging.info("Starting simulation...")
    orchestrator.run()
    logging.info("Backtest simulation finished.")

    # --- Performance Analysis and Reporting ---
    logging.info("Generating performance report...")
    
    trade_log_df = portfolio.get_trade_log_df()
    equity_curve = portfolio.get_equity_curve()

    # Save equity curve
    if not equity_curve.empty:
        equity_curve_path = run_path / 'equity_curve.parquet'
        equity_curve_df = equity_curve.to_frame(name="Equity")
        equity_curve_df.reset_index(inplace=True)
        equity_curve_df = equity_curve_df.rename(columns={'index': 'date'})
        equity_curve_df.to_parquet(equity_curve_path)

        equity_curve_csv_path = run_path / 'equity_curve.csv'
        equity_curve_df.to_csv(equity_curve_csv_path)
        logging.info(f"Equity curve saved to: {equity_curve_csv_path}")

    # Generate comprehensive analysis using BacktestMetrics
    from meluna.analysis.metrics.BacktestMetrics import BacktestMetrics
    analyzer = BacktestMetrics(equity_curve, trade_log_df)
    
    # Calculate and display metrics
    all_metrics = analyzer.calculate_all_metrics()
    analyzer.display_metrics()
    
    # Save all trade reports using the analyzer
    analyzer.save_all_reports(run_path)

    # Prompt for user comments about the backtest
    print("\n" + "="*60)
    print("BACKTEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    user_comment = input("\nWould you like to add any comments about this backtest? (Press Enter to skip): ").strip()
    
    # Save user comment if provided
    if user_comment:
        comment_file = run_path / 'user_comments.txt'
        with open(comment_file, 'w', encoding='utf-8') as f:
            f.write("USER COMMENTS\n")
            f.write("="*50 + "\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Backtest Run: {run_path.name}\n")
            f.write("-"*50 + "\n")
            f.write(f"{user_comment}\n")
        print(f"Comments saved to: {comment_file}")
    
    logging.info(f"Analysis for run '{run_path.name}' complete.")

if __name__ == "__main__":
    main()