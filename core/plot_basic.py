"""
Basic static plotting functions for stock analysis.
Used by run_basic_analysis and run_multiple_stocks in main.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict


def plot_price_sma_and_runs(df: pd.DataFrame, sma_col: str, runs: List[Dict], title: str, inName: str):
    """
    Static plot: price + SMA + shaded runs.
    
    Features:
        - Plots closing price and SMA overlay
        - Shades upward runs (green), downward runs (coral)
        - Opens in new matplotlib window
    
    Args:
        df: Data with 'Close' column, datetime index
        sma_col: SMA column name (e.g., 'SMA_5')
        runs: List of dicts with 'start', 'end', 'direction', 'length'
        title: Plot title
        inName: Stock ticker for legend
    
    Returns:
        None - displays via plt.show()
    """
    fig_local, ax_local = plt.subplots(figsize=(10, 4))
    ax_local.plot(df.index, df["Close"], label=inName, linewidth=1.5)
    ax_local.plot(df.index, df[sma_col], label=f"{sma_col}", linewidth=1.0, alpha=0.8)
    if runs:
        for r in runs:
            color = "lightgreen" if r.get("direction") == "up" else "lightcoral"
            ax_local.axvspan(r.get("start"), r.get("end"), alpha=0.15, color=color)
    ax_local.set_title(title)
    ax_local.set_xlabel("Date")
    ax_local.set_ylabel("Price")
    ax_local.grid(True, linestyle="--", alpha=0.3)
    ax_local.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_multiple_stocks_comparison(stock_data_list: List[Dict], title: str = "Stock Comparison"):
    """
    Compare multiple stocks on single plot.
    
    Features:
        - Overlays all stocks on same axes
        - Solid lines for price, dashed for SMA
        - Auto-cycles through tab10 colors
        - Legend positioned outside plot
    
    Args:
        stock_data_list: List of dicts with:
            - 'ticker': Stock symbol
            - 'data': DataFrame with 'Close'
            - 'sma_col': SMA column name
        title: Plot title (default "Stock Comparison")
    
    Returns:
        None - displays via plt.show(), skips if empty list
    """
    if not stock_data_list:
        return
    fig_local, ax_local = plt.subplots(figsize=(14, 8))
    color_cycle = plt.cm.tab10.colors
    for i, stock_info in enumerate(stock_data_list):
        ticker = stock_info.get('ticker', f'Stock {i+1}')
        data = stock_info['data']
        sma_col = stock_info['sma_col']
        color = color_cycle[i % len(color_cycle)]
        ax_local.plot(data.index, data["Close"], label=ticker, linewidth=1.8, color=color)
        ax_local.plot(data.index, data[sma_col], label=f"{sma_col} {ticker}", linewidth=1.0, linestyle='--', color=color, alpha=0.8)
    
    # Set properties once after all stocks are plotted
    ax_local.set_title(title, fontsize=16, fontweight='bold')
    ax_local.set_xlabel("Date")
    ax_local.set_ylabel("Price")
    ax_local.grid(True, linestyle="--", alpha=0.3)
    ax_local.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


