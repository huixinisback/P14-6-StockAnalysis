#!/usr/bin/env python3
"""
Performance timing tests for core functions.
Configuration: AAPL, 3y, 1d, SMA window 5.
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import get_numeric_close
from core.indicators import (
    sma_sliding_window,
    exponential_moving_average,
    relative_strength_index,
    bollinger_bands,
    daily_simple_returns,
    compute_buy_sell_signals,
    max_profit_multiple_transactions
)
from core.runs import find_up_down_runs


def time_function(func, *args, **kwargs):
    """Time a function execution and return result + elapsed time."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    return result, elapsed_ms


def run_performance_tests():
    """Run all performance timing tests."""
    # Configuration
    ticker = "AAPL"
    period = "3y"
    interval = "1d"
    sma_window = 5
    
    print("\n" + "="*70)
    print("PERFORMANCE TIMING TESTS")
    print("="*70)
    print(f"Configuration: {ticker} | {period} | {interval} | SMA-{sma_window}")
    print("="*70)
    
    # Download data
    print("\nDownloading data...")
    prices, download_time = time_function(get_numeric_close, ticker, period, interval)
    print(f"✓ {len(prices)} data points | {download_time:.2f} ms")
    
    # SMA tests
    print("\n" + "-"*70)
    print("INDICATORS.PY - sma_sliding_window()")
    print("-"*70)
    for window in [sma_window, 20]:
        result, elapsed_ms = time_function(sma_sliding_window, prices, window)
        print(f"SMA-{window}: {elapsed_ms:.3f} ms")
    
    # EMA tests
    print("\n" + "-"*70)
    print("INDICATORS.PY - exponential_moving_average()")
    print("-"*70)
    for period_val in [12, 26]:
        result, elapsed_ms = time_function(exponential_moving_average, prices, period=period_val)
        print(f"EMA-{period_val}: {elapsed_ms:.3f} ms")
    
    # RSI test
    print("\n" + "-"*70)
    print("INDICATORS.PY - relative_strength_index()")
    print("-"*70)
    result, elapsed_ms = time_function(relative_strength_index, prices, 14)
    print(f"RSI-14: {elapsed_ms:.3f} ms")
    
    # Bollinger Bands test
    print("\n" + "-"*70)
    print("INDICATORS.PY - bollinger_bands()")
    print("-"*70)
    result, elapsed_ms = time_function(bollinger_bands, prices, 20, 2.0)
    print(f"Bollinger Bands (20, 2σ): {elapsed_ms:.3f} ms")
    
    # Daily Returns test
    print("\n" + "-"*70)
    print("INDICATORS.PY - daily_simple_returns()")
    print("-"*70)
    result, elapsed_ms = time_function(daily_simple_returns, prices)
    print(f"Daily Returns: {elapsed_ms:.3f} ms")
    
    # Buy/Sell Signals test
    print("\n" + "-"*70)
    print("INDICATORS.PY - compute_buy_sell_signals()")
    print("-"*70)
    sma = sma_sliding_window(prices, sma_window)
    result, elapsed_ms = time_function(compute_buy_sell_signals, prices.values, sma.values)
    buy_count = len(result['buy_indices'])
    sell_count = len(result['sell_indices'])
    print(f"Buy/Sell Signals: {elapsed_ms:.3f} ms | {buy_count} buys, {sell_count} sells")
    
    # Max Profit test
    print("\n" + "-"*70)
    print("INDICATORS.PY - max_profit_multiple_transactions()")
    print("-"*70)
    result, elapsed_ms = time_function(max_profit_multiple_transactions, prices, return_transactions=True)
    profit, transactions = result
    print(f"Max Profit: {elapsed_ms:.3f} ms | ${profit:.2f} from {len(transactions)} trades")
    
    # Runs Analysis test
    print("\n" + "-"*70)
    print("RUNS.PY - find_up_down_runs()")
    print("-"*70)
    result, elapsed_ms = time_function(find_up_down_runs, prices)
    runs, summary = result
    print(f"Runs Analysis: {elapsed_ms:.3f} ms | {summary['up']['num_runs']} up, {summary['down']['num_runs']} down")
    
    # Full suite test
    print("\n" + "-"*70)
    print("COMBINED - Full Indicator Suite")
    print("-"*70)
    
    start_all = time.perf_counter()
    
    # Calculate all indicators
    sma_5 = sma_sliding_window(prices, 5)
    sma_10 = sma_sliding_window(prices, 10)
    sma_20 = sma_sliding_window(prices, 20)
    sma_50 = sma_sliding_window(prices, 50)
    ema_12 = exponential_moving_average(prices, period=12)
    ema_26 = exponential_moving_average(prices, period=26)
    rsi_14 = relative_strength_index(prices, 14)
    bb_middle, bb_upper, bb_lower = bollinger_bands(prices, 20, 2.0)
    signals = compute_buy_sell_signals(prices.values, sma_5.values)
    max_profit_val, transactions = max_profit_multiple_transactions(prices, return_transactions=True)
    runs, runs_summary = find_up_down_runs(prices)
    
    total_time = (time.perf_counter() - start_all) * 1000
    
    indicator_count = 13
    print(f"All {indicator_count} indicators: {total_time:.3f} ms")
    print(f"  Average per indicator: {total_time/indicator_count:.3f} ms")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Dataset: {len(prices)} points ({ticker}, {period}, {interval})")
    print(f"Data download: {download_time:.2f} ms")
    print(f"All computations: {total_time:.2f} ms")
    print(f"Computation: {total_time/(download_time+total_time)*100:.1f}% of total time")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_performance_tests()
