"""
Indicator calculation and plotting functions.
Handles SMA, EMA, RSI, and Bollinger Bands indicators.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .indicators import (
    sma_sliding_window,
    exponential_moving_average,
    relative_strength_index,
    bollinger_bands,
    compute_buy_sell_signals,
    max_profit_multiple_transactions,
)
from .runs import find_up_down_runs


def calculate_all_indicators(close_prices: pd.Series, ticker: str, sma_window: int = 5) -> Dict:
    """
    Calculate all technical indicators for a stock.
    
    Features:
        - SMAs: 5, 10, 20, 50 (+ custom window if specified)
        - EMAs: 12, 26
        - RSI: 14 period
        - Bollinger Bands: 20 period, 2σ
        - Buy/Sell signals: SMA crossover
        - Max profit: greedy algorithm
        - Runs analysis: up/down streaks
    
    Args:
        close_prices: Closing prices with datetime index
        ticker: Stock symbol (e.g., 'AAPL')
        sma_window: SMA period for signals (default 5)
    
    Returns:
        Dict: All indicators, stored in plot.indicators[ticker]
            Keys: SMA_*, EMA_*, RSI_14, BB_*, Buy_Sell_Signals,
                  Price_Index, SMA_Window_Used, Max_Profit, Runs, Runs_Summary
    
    Effects:
        Updates plot.indicators global dict
    """
    from . import plot
    
    if ticker not in plot.indicators:
        plot.indicators[ticker] = {}
    
    # Standard SMAs: 5, 10, 20, 50
    plot.indicators[ticker]["SMA_5"] = sma_sliding_window(close_prices, 5)
    plot.indicators[ticker]["SMA_10"] = sma_sliding_window(close_prices, 10)
    plot.indicators[ticker]["SMA_20"] = sma_sliding_window(close_prices, 20)
    plot.indicators[ticker]["SMA_50"] = sma_sliding_window(close_prices, 50)
    
    # Add custom SMA window if non-standard
    if sma_window not in [5, 10, 20, 50]:
        plot.indicators[ticker][f"SMA_{sma_window}"] = sma_sliding_window(close_prices, sma_window)
    plot.indicators[ticker]["EMA_12"] = exponential_moving_average(close_prices, period=12)
    plot.indicators[ticker]["EMA_26"] = exponential_moving_average(close_prices, period=26)
    plot.indicators[ticker]["RSI_14"] = relative_strength_index(close_prices, 14)
    
    bb_middle, bb_upper, bb_lower = bollinger_bands(close_prices, 20, 2.0)
    plot.indicators[ticker]["BB_Upper"] = bb_upper
    plot.indicators[ticker]["BB_Middle"] = bb_middle
    plot.indicators[ticker]["BB_Lower"] = bb_lower
    
    # Buy/Sell signals from SMA crossover
    sma_key = f"SMA_{sma_window}"
    if sma_key not in plot.indicators[ticker]:
        plot.indicators[ticker][sma_key] = sma_sliding_window(close_prices, sma_window)
    
    sma_for_signals = plot.indicators[ticker][sma_key]
    prices_array = close_prices.values
    sma_array = sma_for_signals.values
    signals_dict = compute_buy_sell_signals(prices_array, sma_array)
    
    # Calculate signal statistics
    buy_prices_list = signals_dict['buy_prices']
    sell_prices_list = signals_dict['sell_prices']
    num_completed = min(len(buy_prices_list), len(sell_prices_list))
    
    total_profit = 0.0
    winning_trades = 0
    losing_trades = 0
    
    for i in range(num_completed):
        profit = sell_prices_list[i] - buy_prices_list[i]
        total_profit += profit
        if profit > 0:
            winning_trades += 1
        elif profit < 0:
            losing_trades += 1
    
    # Win factor: ratio of winning to losing trades
    win_factor = winning_trades / losing_trades if losing_trades > 0 else (winning_trades if winning_trades > 0 else 0)
    
    # Store signals with metadata and statistics
    plot.indicators[ticker]["Buy_Sell_Signals"] = signals_dict
    plot.indicators[ticker]["Price_Index"] = close_prices.index
    plot.indicators[ticker]["SMA_Window_Used"] = sma_window
    plot.indicators[ticker]["Signal_Total_Profit"] = total_profit
    plot.indicators[ticker]["Signal_Win_Factor"] = win_factor
    plot.indicators[ticker]["Signal_Winning_Trades"] = winning_trades
    plot.indicators[ticker]["Signal_Losing_Trades"] = losing_trades
    
    # Max profit (greedy algorithm) with transaction details
    max_profit, max_profit_transactions = max_profit_multiple_transactions(close_prices, return_transactions=True)
    plot.indicators[ticker]["Max_Profit"] = max_profit
    plot.indicators[ticker]["Max_Profit_Transactions"] = max_profit_transactions
    
    # Runs analysis
    runs, runs_summary = find_up_down_runs(close_prices)
    plot.indicators[ticker]["Runs"] = runs
    plot.indicators[ticker]["Runs_Summary"] = runs_summary
    
    return plot.indicators[ticker]


def remove_indicator_for_all_stocks(indicator_key):
    """
    Remove indicator lines from plot across all stocks.
    
    Features:
        - Finds and removes matching matplotlib lines
        - Special handling for RSI (removes secondary axis)
        - Refreshes canvas after removal
    
    Args:
        indicator_key: Indicator ID (e.g., 'SMA_5', 'RSI_14', 'BB_Upper')
    
    Returns:
        None
    
    Effects:
        Removes lines, refreshes canvas (RSI: removes secondary y-axis)
    """
    from . import plot
    
    # RSI: remove entire secondary axis
    if indicator_key == "RSI_14" and "RSI_Axis" in plot.lines:
        rsi_axis = plot.lines["RSI_Axis"]
        
        # Clear all RSI lines first
        rsi_keys = [k for k in list(plot.lines.keys()) if k.startswith("RSI_")]
        for key in rsi_keys:
            if key != "RSI_Axis":
                obj = plot.lines.get(key)
                if obj and hasattr(obj, 'remove'):
                    try:
                        obj.remove()
                    except:
                        pass
                if key in plot.lines:
                    del plot.lines[key]
        
        # Remove secondary axis
        try:
            rsi_axis.remove()
        except:
            pass
        if "RSI_Axis" in plot.lines:
            del plot.lines["RSI_Axis"]
    
    # Standard indicator removal
    keys_to_remove = [key for key in list(plot.lines.keys()) if indicator_key in key]
    for key in keys_to_remove:
        obj = plot.lines.get(key)
        if obj is None:
            continue
        try:
            if isinstance(obj, list):  # Fill collections
                for item in obj:
                    item.remove()
            else:
                obj.remove()
        except Exception:
            pass
        
        if key in plot.lines:
            del plot.lines[key]
    
    plot.update_plot_properties()
    if plot.fig is not None:
        plot.fig.canvas.draw_idle()


def plot_sma_for_all_stocks(period):
    """
    Plot SMA for all active stocks.
    
    Features:
        - Plots SMA for each stock in current date range
        - Auto-assigns colors (tab10 cycle)
        - Labels: "{ticker} SMA {period}"
        - Removes existing SMA_{period} lines first
    
    Args:
        period: SMA period (5, 10, 20, or 50)
    
    Returns:
        None
    
    Effects:
        Adds lines to plot.lines, updates legend, refreshes canvas
    """
    from . import plot
    
    # Clear existing SMA lines for this period
    keys_to_remove = [k for k in plot.lines.keys() if k.startswith(f"SMA_{period}_")]
    for key in keys_to_remove:
        plot.lines[key].remove()
        del plot.lines[key]
    
    color_cycle = plt.cm.tab10.colors
    idx = 0
    
    # Plot SMA for each active stock
    for stock_key in [k for k in plot.lines if k.endswith("_Close")]:
        stock_name = stock_key.replace("_Close", "")
        
        if stock_name not in plot.indicators or f"SMA_{period}" not in plot.indicators[stock_name]:
            continue
        
        data = plot.indicators[stock_name][f"SMA_{period}"]
        
        # Filter to visible date range
        if plot.current_data is not None and len(plot.current_data) > 0:
            data = data.loc[data.index.isin(plot.current_data.index)]
        
        if len(data) == 0:
            continue
        
        color = color_cycle[idx % len(color_cycle)]
        idx += 1
        line, = plot.ax.plot(data.index, data.values, label=f"SMA {period} {stock_name}", 
                       linewidth=1.5, alpha=0.8, linestyle='--', color=color)
        plot.lines[f"SMA_{period}_{stock_name}"] = line
    
    plot.update_plot_properties()
    if plot.fig is not None:
        plot.fig.canvas.draw_idle()


def plot_ema_for_all_stocks(period):
    """
    Plot EMA for all active stocks.
    
    Features:
        - Plots EMA for each stock in current date range
        - Auto-assigns colors (Set2 cycle)
        - Labels: "{ticker} EMA {period}"
        - Removes existing EMA_{period} lines first
    
    Args:
        period: EMA period (12 or 26)
    
    Returns:
        None
    
    Effects:
        Adds lines to plot.lines, updates legend, refreshes canvas
    """
    from . import plot
    
    # Clear existing EMA lines for this period
    keys_to_remove = [k for k in plot.lines.keys() if k.startswith(f"EMA_{period}_")]
    for key in keys_to_remove:
        plot.lines[key].remove()
        del plot.lines[key]
    
    color_cycle = plt.cm.Set2.colors
    idx = 0
    
    # Plot EMA for each active stock
    for stock_key in [k for k in plot.lines if k.endswith("_Close")]:
        stock_name = stock_key.replace("_Close", "")
        
        if stock_name not in plot.indicators or f"EMA_{period}" not in plot.indicators[stock_name]:
            continue
        
        data = plot.indicators[stock_name][f"EMA_{period}"]
        
        # Filter to visible date range
        if plot.current_data is not None and len(plot.current_data) > 0:
            data = data.loc[data.index.isin(plot.current_data.index)]
        
        if len(data) == 0:
            continue
        
        color = color_cycle[idx % len(color_cycle)]
        idx += 1
        line, = plot.ax.plot(data.index, data.values, label=f"EMA {period} {stock_name}", 
                       linewidth=1.5, alpha=0.8, color=color)
        plot.lines[f"EMA_{period}_{stock_name}"] = line
    
    plot.update_plot_properties()
    if plot.fig is not None:
        plot.fig.canvas.draw_idle()


def plot_rsi_for_all_stocks():
    """
    Plot RSI on secondary y-axis (right side).
    
    Features:
        - Creates secondary axis if not exists
        - RSI range 0-100 with reference lines at 30/70
        - Multi-color lines for multiple stocks
        - Removes existing RSI lines first
    
    Args:
        None - reads plot.indicators
    
    Returns:
        None
    
    Effects:
        Creates secondary y-axis (twinx), adds lines, updates legend
    """
    from . import plot
    
    # Clear existing RSI lines
    keys_to_remove = [k for k in plot.lines.keys() if k.startswith("RSI_14_")]
    for key in keys_to_remove:
        plot.lines[key].remove()
        del plot.lines[key]
    
    # Create secondary y-axis for RSI if needed
    if "RSI_Axis" not in plot.lines:
        ax_rsi = plot.ax.twinx()
        ax_rsi.set_ylabel("RSI", color='red')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.tick_params(axis='y', labelcolor='red')
        ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        plot.lines["RSI_Axis"] = ax_rsi
    
    ax_rsi = plot.lines["RSI_Axis"]
    color_cycle = ['red', 'orange', 'purple', 'brown', 'pink']
    idx = 0
    
    # Plot RSI for each active stock
    for stock_key in [k for k in plot.lines if k.endswith("_Close")]:
        stock_name = stock_key.replace("_Close", "")
        
        if stock_name not in plot.indicators or "RSI_14" not in plot.indicators[stock_name]:
            continue
        
        data = plot.indicators[stock_name]["RSI_14"]
        
        # Filter to visible date range
        if plot.current_data is not None and len(plot.current_data) > 0:
            data = data.loc[data.index.isin(plot.current_data.index)]
        
        if len(data) == 0:
            continue
        
        color = color_cycle[idx % len(color_cycle)]
        idx += 1
        line, = ax_rsi.plot(data.index, data.values, label=f"RSI 14 {stock_name}", 
                           linewidth=1.5, color=color, alpha=0.8)
        plot.lines[f"RSI_14_{stock_name}"] = line
    
    plot.update_plot_properties()
    if plot.fig is not None:
        plot.fig.canvas.draw_idle()


def plot_bollinger_for_all_stocks():
    """
    Plot Bollinger Bands (3 lines + fill) for all stocks.
    
    Features:
        - Plots upper, middle (SMA), lower bands
        - Fills area between upper/lower (semi-transparent)
        - Auto-assigns colors (tab10 cycle)
        - Removes existing BB lines first
    
    Args:
        None - reads plot.indicators
    
    Returns:
        None
    
    Effects:
        Adds BB lines/fills to plot.lines, updates legend
    """
    from . import plot
    
    # Clear existing Bollinger Band lines
    keys_to_remove = [k for k in plot.lines.keys() if k.startswith("BB_")]
    for key in keys_to_remove:
        try:
            if isinstance(plot.lines[key], list):
                for item in plot.lines[key]:
                    item.remove()
            else:
                plot.lines[key].remove()
            del plot.lines[key]
        except Exception:
            pass
    
    color_cycle = plt.cm.tab10.colors
    idx = 0
    
    stock_keys = [k for k in plot.lines if k.endswith("_Close")]
    
    for stock_key in stock_keys:
        stock_name = stock_key.replace("_Close", "")
        
        if stock_name not in plot.indicators:
            continue
        
        has_upper = "BB_Upper" in plot.indicators[stock_name]
        has_middle = "BB_Middle" in plot.indicators[stock_name]
        has_lower = "BB_Lower" in plot.indicators[stock_name]
        
        if not (has_upper and has_middle and has_lower):
            continue
        
        upper = plot.indicators[stock_name]["BB_Upper"]
        middle = plot.indicators[stock_name]["BB_Middle"]
        lower = plot.indicators[stock_name]["BB_Lower"]
        
        if len(upper) == 0 or len(middle) == 0 or len(lower) == 0:
            continue
        
        color = color_cycle[idx % len(color_cycle)]
        idx += 1
        
        try:
            lu, = plot.ax.plot(upper.index, upper.values, label=f"BB Upper {stock_name}", 
                         linewidth=1, alpha=0.7, color='blue')
            lm, = plot.ax.plot(middle.index, middle.values, label=f"BB Middle {stock_name}", 
                         linewidth=1, alpha=0.7, color='gray', linestyle='--')
            ll, = plot.ax.plot(lower.index, lower.values, label=f"BB Lower {stock_name}", 
                         linewidth=1, alpha=0.7, color='blue')
            fill = plot.ax.fill_between(upper.index, upper.values, lower.values, 
                                   alpha=0.1, color=color)
            
            plot.lines[f"BB_Upper_{stock_name}"] = lu
            plot.lines[f"BB_Middle_{stock_name}"] = lm
            plot.lines[f"BB_Lower_{stock_name}"] = ll
            plot.lines[f"BB_Fill_{stock_name}"] = [fill]
        except Exception:
            pass
    plot.update_plot_properties()
    if plot.fig is not None:
        plot.fig.canvas.draw_idle()

