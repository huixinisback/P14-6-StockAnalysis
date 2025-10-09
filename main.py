# run: python main.py
from core.data import get_numeric_close
from core.indicators import sma_sliding_window, daily_simple_returns, max_profit_multiple_transactions
from core.runs import find_up_down_runs
from core.plot import (
    plot_price_sma_and_runs, 
    plot_multiple_stocks_comparison,
    create_interactive_plot
)

import pandas as pd # using pandas for data manipulation (DataFrame, Series, etc.)
from typing import Dict


def print_analysis_results(ticker: str, period: str, interval: str, close_prices: pd.Series, indicators_dict: Dict):
    """Print comprehensive analysis results to console."""
    print("\n" + "="*60)
    print(" ANALYSIS RESULTS")
    print("="*60)
    print(f"Stock: {ticker}")
    print(f"Period: {period} | Interval: {interval}")
    print(f"Total Data Points: {len(close_prices)}")
    print(f"Latest Price: ${close_prices.iloc[-1]:.2f}")
    
    # SMA Analysis - use the window that was used for signals
    sma_window_used = indicators_dict.get("SMA_Window_Used", 20)
    sma_key = f"SMA_{sma_window_used}"
    sma_data = indicators_dict.get(sma_key)
    if sma_data is not None:
        sma_values = sma_data.dropna().values
        if len(sma_values) > 0:
            print(f"\n📊 Simple Moving Average (Window={sma_window_used}):")
            first_5 = sma_values[:min(5, len(sma_values))]
            print(f"   First {len(first_5)} SMA values: {first_5}")
            print(f"   Latest SMA: ${sma_values[-1]:.2f}")
    
    # Daily Returns
    daily_returns = close_prices.pct_change().dropna()
    if len(daily_returns) > 0:
        print(f"\n📉 Daily Returns:")
        first_5_returns = daily_returns.values[:min(5, len(daily_returns))]
        print(f"   First {len(first_5_returns)} returns: {first_5_returns}")
        avg_return = daily_returns.mean()
        print(f"   Average return: {avg_return:.4f} ({avg_return*100:.2f}%)")
    
    # Runs Analysis
    print(f"\n🎯 Runs Analysis:")
    runs_up = []
    runs_down = []
    current_run_length = 1
    current_direction = None
    
    for i in range(1, len(close_prices)):
        if close_prices.iloc[i] > close_prices.iloc[i-1]:
            direction = "up"
        elif close_prices.iloc[i] < close_prices.iloc[i-1]:
            direction = "down"
        else:
            continue
        
        if direction == current_direction:
            current_run_length += 1
        else:
            if current_direction == "up":
                runs_up.append(current_run_length)
            elif current_direction == "down":
                runs_down.append(current_run_length)
            current_direction = direction
            current_run_length = 1
    
    # Don't forget last run
    if current_direction == "up":
        runs_up.append(current_run_length)
    elif current_direction == "down":
        runs_down.append(current_run_length)
    
    total_up_days = sum(runs_up) if runs_up else 0
    total_down_days = sum(runs_down) if runs_down else 0
    longest_up = max(runs_up) if runs_up else 0
    longest_down = max(runs_down) if runs_down else 0
    
    print(f"   Upward Runs: {len(runs_up)} runs ({total_up_days} total days)")
    print(f"   Downward Runs: {len(runs_down)} runs ({total_down_days} total days)")
    print(f"   Longest Upward Streak: {longest_up} consecutive days")
    print(f"   Longest Downward Streak: {longest_down} consecutive days")
    
    # Max Profit
    max_profit = indicators_dict.get("Max_Profit", 0.0)
    print(f"\n💰 Max Profit Potential (multiple transactions): ${max_profit:.2f}")
    
    # Buy/Sell Signals
    signals_dict = indicators_dict.get("Buy_Sell_Signals", {})
    buy_indices = signals_dict.get('buy_indices', [])
    sell_indices = signals_dict.get('sell_indices', [])
    buy_prices = signals_dict.get('buy_prices', [])
    sell_prices = signals_dict.get('sell_prices', [])
    
    print(f"\n🔔 Buy/Sell Signals:")
    print(f"   Buy Signals: {len(buy_indices)}")
    print(f"   Sell Signals: {len(sell_indices)}")
    
    if len(buy_indices) > 0 and len(buy_prices) > 0:
        first_buy_idx = buy_indices[0]
        first_buy_price = buy_prices[0]
        print(f"   First Buy: ${first_buy_price:.2f} on day {first_buy_idx}")
    
    if len(sell_indices) > 0 and len(sell_prices) > 0:
        first_sell_idx = sell_indices[0]
        first_sell_price = sell_prices[0]
        print(f"   First Sell: ${first_sell_price:.2f} on day {first_sell_idx}")
    
    print("="*60 + "\n")


# -------- User Input --------
print("\n" + "="*60)
print("STOCK ANALYSIS CONFIGURATION")
print("="*60)

TICKER = input("Enter stock ticker symbol (e.g., AAPL) [default: AAPL]: ").strip().upper() or "AAPL"
PERIOD = input("Enter period (1y, 2y, 3y, 5y) [default: 3y]: ").strip() or "3y"
INTERVAL = input("Enter interval (1d, 1wk, 1mo) [default: 1d]: ").strip() or "1d"

# SMA Window with validation
sma_input = input("Enter SMA window size (1-200) [default: 5]: ").strip()
try:
    SMA_WINDOW = int(sma_input) if sma_input else 5
    if SMA_WINDOW < 1 or SMA_WINDOW > 200:
        print(f"Invalid SMA window {SMA_WINDOW}, using default 5")
        SMA_WINDOW = 5
except ValueError:
    print(f"Invalid SMA window input, using default 5")
    SMA_WINDOW = 5

print(f"\nConfiguration: {TICKER} | {PERIOD} | {INTERVAL} | SMA Window: {SMA_WINDOW}")
print("="*60 + "\n")

def run_basic_analysis(ticker=None, period=None, interval=None, sma_window=None):
    """Run basic analysis with original functionality."""
    # Use provided parameters or defaults
    ticker = ticker or TICKER
    period = period or PERIOD
    interval = interval or INTERVAL
    sma_window = sma_window or SMA_WINDOW
    
    # load data from yfinance
    close = get_numeric_close(ticker, period, interval)
    data = pd.DataFrame({"Close": close})

    # get simple moving average , key indicator as part of requirements
    sma_series = sma_sliding_window(data["Close"], sma_window)
    data[sma_series.name] = sma_series
    data["Daily_Return"] = daily_simple_returns(data["Close"])

    # 3) Runs + summary
    runs, summary = find_up_down_runs(data["Close"])

    # 4) Strategy: max profit
    profit = max_profit_multiple_transactions(data["Close"])

    # 5) Print summary
    print(f"\n=== {ticker} | {period} | {interval} ===")
    print(f"SMA window: {sma_window}")
    print("\nUp/Down Runs Summary:")
    print(f"  Up   -> num_runs: {summary['up']['num_runs']}, "
          f"total_days: {summary['up']['total_days_in_runs']}, "
          f"longest_streak: {summary['up']['longest_streak']}")
    print(f"  Down -> num_runs: {summary['down']['num_runs']}, "
          f"total_days: {summary['down']['total_days_in_runs']}, "
          f"longest_streak: {summary['down']['longest_streak']}")
    print(f"\nMax Profit (multiple transactions): {profit:.2f}")

    # 6) Plot
    plot_price_sma_and_runs(
        data,
        sma_series.name,
        runs,
        title=f"{ticker} Close vs. {sma_series.name} (shaded up/down runs)", 
        inName = ticker
    )
    
    return data, runs


def run_multiple_stocks():
    """Run analysis for multiple stocks specified by user."""
    print("=== Multiple Stock Analysis ===")
    print("Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL)")
    
    stocks_input = input("Enter stock symbols: ").strip()
    if not stocks_input:
        print("No stocks entered. Using default: AAPL")
        stocks = ["AAPL"]
    else:
        stocks = [s.strip().upper() for s in stocks_input.split(",")]
    
    print(f"Analyzing {len(stocks)} stocks: {', '.join(stocks)}")
    
    # Collect data for all stocks
    stock_data_list = []
    
    for i, stock in enumerate(stocks):
        try:
            print(f"\n--- Processing {stock} ({i+1}/{len(stocks)}) ---")
            
            # Get data
            close = get_numeric_close(stock, PERIOD, INTERVAL)
            data = pd.DataFrame({"Close": close})
            
            # Calculate indicators
            sma_series = sma_sliding_window(data["Close"], SMA_WINDOW)
            data[sma_series.name] = sma_series
            data["Daily_Return"] = daily_simple_returns(data["Close"])
            
            # Calculate runs and profit
            runs, summary = find_up_down_runs(data["Close"])
            profit = max_profit_multiple_transactions(data["Close"])
            
            # Store data for comparison plot
            stock_data_list.append({
                'ticker': stock,
                'data': data,
                'sma_col': sma_series.name,
                'runs': runs,
                'summary': summary,
                'profit': profit
            })
            
            # Print summary
            print(f"Up runs: {summary['up']['num_runs']}, Down runs: {summary['down']['num_runs']}")
            print(f"Max profit: {profit:.2f}")
            
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            continue
    
    # Create comparison plot
    if stock_data_list:
        print(f"\n--- Creating comparison plot for {len(stock_data_list)} stocks ---")
        plot_multiple_stocks_comparison(
            stock_data_list, 
            title=f"Stock Comparison: {', '.join([s['ticker'] for s in stock_data_list])}"
        )
    
    print(f"\nCompleted analysis for {len(stock_data_list)} stocks.")


def main():
    """Main function - all interactions through matplotlib interface."""
    import sys
    
    print("="*80)
    print("STOCK ANALYSIS SYSTEM - MATPLOTLIB INTERFACE")
    print("="*80)
    print()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['multi', 'multiple', 'compare']:
            print("Running multiple stock analysis...")
            run_multiple_stocks()
        elif arg in ['interactive', 'plot', 'controls']:
            print("Opening interactive plot with controls...")
            create_interactive_plot(TICKER, PERIOD, INTERVAL, SMA_WINDOW, print_results_func=print_analysis_results)
        else:
            # Treat as single stock ticker
            print(f"Opening interactive plot for {arg.upper()}...")
            create_interactive_plot(arg.upper(), PERIOD, INTERVAL, SMA_WINDOW, print_results_func=print_analysis_results)
    else:
        # Default to interactive plot
        print("Opening interactive plot with all controls...")
        print("Use the controls at the bottom to:")
        print("• Select Add/Remove mode with radio buttons")
        print("• Toggle individual indicators with checkboxes")
        print("• Add/remove stocks with the text input and submit")
        print("• Figure auto-resizes to fit all content")
        print()
        create_interactive_plot(TICKER, PERIOD, INTERVAL, SMA_WINDOW, print_results_func=print_analysis_results)

if __name__ == "__main__":
    main()
