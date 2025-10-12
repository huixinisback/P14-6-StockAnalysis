# run: python main.py
from core.data import get_numeric_close
from core.indicators import sma_sliding_window, daily_simple_returns, max_profit_multiple_transactions
from core.runs import find_up_down_runs
from core.plot import create_interactive_plot

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
    
    # SMA window used for signals
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
    
    # Append final run
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
    
    # Get price dates for transaction display
    price_dates = close_prices.index
    
    # Max Profit
    max_profit = indicators_dict.get("Max_Profit", 0.0)
    max_profit_transactions = indicators_dict.get("Max_Profit_Transactions", [])
    
    print(f"\n💰 Max Profit Potential (multiple transactions): ${max_profit:.2f}")
    print(f"   Total Transactions: {len(max_profit_transactions)}")
    
    if len(max_profit_transactions) > 0:
        # First 5 transactions
        print(f"\n   First {min(5, len(max_profit_transactions))} Transactions:")
        print(f"   {'#':<4} {'Buy Date':<12} {'Buy Price':<12} {'Sell Date':<12} {'Sell Price':<12} {'Profit':<12}")
        print(f"   {'-'*68}")
        for i in range(min(5, len(max_profit_transactions))):
            txn = max_profit_transactions[i]
            buy_idx = txn['buy_index']
            sell_idx = txn['sell_index']
            buy_date = close_prices.index[buy_idx].strftime('%Y-%m-%d') if buy_idx < len(close_prices) else 'N/A'
            sell_date = close_prices.index[sell_idx].strftime('%Y-%m-%d') if sell_idx < len(close_prices) else 'N/A'
            print(f"   {i+1:<4} {buy_date:<12} ${txn['buy_price']:<11.2f} {sell_date:<12} ${txn['sell_price']:<11.2f} ${txn['profit']:<11.2f}")
        
        # Last 5 transactions (if more than 5)
        if len(max_profit_transactions) > 5:
            print(f"\n   Last {min(5, len(max_profit_transactions))} Transactions:")
            print(f"   {'#':<4} {'Buy Date':<12} {'Buy Price':<12} {'Sell Date':<12} {'Sell Price':<12} {'Profit':<12}")
            print(f"   {'-'*68}")
            for i in range(max(0, len(max_profit_transactions)-5), len(max_profit_transactions)):
                txn = max_profit_transactions[i]
                buy_idx = txn['buy_index']
                sell_idx = txn['sell_index']
                buy_date = close_prices.index[buy_idx].strftime('%Y-%m-%d') if buy_idx < len(close_prices) else 'N/A'
                sell_date = close_prices.index[sell_idx].strftime('%Y-%m-%d') if sell_idx < len(close_prices) else 'N/A'
                print(f"   {i+1:<4} {buy_date:<12} ${txn['buy_price']:<11.2f} {sell_date:<12} ${txn['sell_price']:<11.2f} ${txn['profit']:<11.2f}")
    
    # Buy/Sell Signals
    signals_dict = indicators_dict.get("Buy_Sell_Signals", {})
    buy_indices = signals_dict.get('buy_indices', [])
    sell_indices = signals_dict.get('sell_indices', [])
    buy_prices = signals_dict.get('buy_prices', [])
    sell_prices = signals_dict.get('sell_prices', [])
    
    # Get signal statistics
    signal_total_profit = indicators_dict.get("Signal_Total_Profit", 0.0)
    signal_win_factor = indicators_dict.get("Signal_Win_Factor", 0.0)
    winning_trades = indicators_dict.get("Signal_Winning_Trades", 0)
    losing_trades = indicators_dict.get("Signal_Losing_Trades", 0)
    
    print(f"\n🔔 Buy/Sell Signals:")
    print(f"   Total Buy Signals: {len(buy_indices)}")
    print(f"   Total Sell Signals: {len(sell_indices)}")
    print(f"   Total Profit (from completed trades): ${signal_total_profit:.2f}")
    print(f"   Win Factor: {signal_win_factor:.2f} ({winning_trades}W / {losing_trades}L)")
    
    num_buys = len(buy_indices)
    
    if num_buys > 0:
        # First 5 transactions
        print(f"\n   First {min(5, num_buys)} Transactions:")
        print(f"   {'#':<4} {'Buy Date':<12} {'Buy Price':<12} {'Sell Date':<12} {'Sell Price':<12} {'Profit':<12}")
        print(f"   {'-'*68}")
        for i in range(min(5, num_buys)):
            buy_idx = buy_indices[i]
            buy_date = close_prices.index[buy_idx].strftime('%Y-%m-%d') if buy_idx < len(close_prices) else 'N/A'
            
            if i < len(sell_indices):
                sell_idx = sell_indices[i]
                sell_date = close_prices.index[sell_idx].strftime('%Y-%m-%d') if sell_idx < len(close_prices) else 'N/A'
                sell_price_val = sell_prices[i]
                profit = sell_price_val - buy_prices[i]
                print(f"   {i+1:<4} {buy_date:<12} ${buy_prices[i]:<11.2f} {sell_date:<12} ${sell_price_val:<11.2f} ${profit:<11.2f}")
            else:
                print(f"   {i+1:<4} {buy_date:<12} ${buy_prices[i]:<11.2f} {'-':<12} {'-':<12} {'(ongoing)':<12}")
        
        # Last 5 transactions (if more than 5)
        if num_buys > 5:
            print(f"\n   Last {min(5, num_buys)} Transactions:")
            print(f"   {'#':<4} {'Buy Date':<12} {'Buy Price':<12} {'Sell Date':<12} {'Sell Price':<12} {'Profit':<12}")
            print(f"   {'-'*68}")
            for i in range(max(0, num_buys-5), num_buys):
                buy_idx = buy_indices[i]
                buy_date = close_prices.index[buy_idx].strftime('%Y-%m-%d') if buy_idx < len(close_prices) else 'N/A'
                
                if i < len(sell_indices):
                    sell_idx = sell_indices[i]
                    sell_date = close_prices.index[sell_idx].strftime('%Y-%m-%d') if sell_idx < len(close_prices) else 'N/A'
                    sell_price_val = sell_prices[i]
                    profit = sell_price_val - buy_prices[i]
                    print(f"   {i+1:<4} {buy_date:<12} ${buy_prices[i]:<11.2f} {sell_date:<12} ${sell_price_val:<11.2f} ${profit:<11.2f}")
                else:
                    print(f"   {i+1:<4} {buy_date:<12} ${buy_prices[i]:<11.2f} {'-':<12} {'-':<12} {'(ongoing)':<12}")
    
    print("="*60 + "\n")


# User Configuration
print("\n" + "="*60)
print("STOCK ANALYSIS CONFIGURATION")
print("="*60)

TICKER = input("Enter stock ticker symbol (e.g., AAPL) [default: AAPL]: ").strip().upper() or "AAPL"
PERIOD = input("Enter period (1y, 2y, 3y, 5y) [default: 3y]: ").strip() or "3y"
INTERVAL = input("Enter interval (1d, 1wk, 1mo) [default: 1d]: ").strip() or "1d"

# SMA window with validation
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


def main():
    """Main function - all interactions through matplotlib interface."""
    print("="*80)
    print("STOCK ANALYSIS SYSTEM - MATPLOTLIB INTERFACE")
    print("="*80)
    print()
    
    # Always run interactive plot (ignore any command-line arguments)
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
