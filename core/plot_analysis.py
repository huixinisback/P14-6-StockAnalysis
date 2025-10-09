"""
Analysis visualization functions.
Handles runs visualization and buy/sell signal plotting.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_runs_for_all_stocks():
    """
    Open popup windows showing color-coded runs for all stocks.
    
    Features:
        - Separate popup per stock
        - Colors: Green (up), Red (down), Blue (longest up), Brown (longest down)
        - Extracts data from main plot
        - Delegates to plot_runs_popup()
    
    Args:
        None - reads plot.lines and plot.indicators
    
    Returns:
        None
    
    Effects:
        - Opens matplotlib window(s)
    """
    from . import plot
    
    # Plot runs for each stock in a separate window
    for stock_key in [k for k in plot.lines if k.endswith("_Close")]:
        stock_name = stock_key.replace("_Close", "")
        
        # Check if this stock has runs calculated
        if (stock_name not in plot.indicators or 
            "Runs" not in plot.indicators[stock_name] or
            "Runs_Summary" not in plot.indicators[stock_name]):
            continue
        
        runs = plot.indicators[stock_name]["Runs"]
        runs_summary = plot.indicators[stock_name]["Runs_Summary"]
        
        # Get the close price data
        close_line = plot.lines.get(f"{stock_name}_Close")
        if close_line is None:
            continue
        
        # Get price data
        xdata = close_line.get_xdata()
        ydata = close_line.get_ydata()
        
        if len(xdata) == 0 or len(ydata) == 0:
            continue
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Date': xdata,
            'Close': ydata
        })
        
        # Calculate returns for run detection
        returns = np.diff(ydata)
        returns = np.insert(returns, 0, np.nan)  # Add NaN at start
        
        # Create runs info dict
        runs_info = {
            'max_up_streak': runs_summary["up"]["longest_streak"],
            'max_down_streak': runs_summary["down"]["longest_streak"]
        }
        
        # Plot in new window
        plot_runs_popup(plot_data, returns, runs_info, stock_name)


def plot_runs_popup(data: pd.DataFrame, returns: np.ndarray, runs_info: dict, ticker: str):
    """
    Plot price with color-coded runs in new window.
    
    Features:
        - Colors segments: Green (up), Red (down)
        - Highlights longest streaks: Blue (up), Brown (down)
        - Flat prices continue previous direction
        - Title shows run statistics
    
    Args:
        data: DataFrame with 'Date' and 'Close' columns
        returns: Daily returns array
        runs_info: Dict with 'max_up_streak', 'max_down_streak'
        ticker: Stock symbol
    
    Returns:
        None - displays via plt.show()
    """
    fig_runs = plt.figure(figsize=(12, 6))

    if len(returns) <= 1 or len(data) == 0:
        plt.title(f'Runs of {ticker} (No Data)')
        plt.show()
        return

    colors = {'up': 'green', 'down': 'red', 'neutral': 'gray'}

    # Track segments to plot and identify longest streaks
    segments = []  # List of (start_idx, end_idx, direction, streak_length)
    current_dir = None
    start_idx = 0
    current_streak = 0

    # Start from index 1 (index 0 has NaN return)
    for i in range(1, len(returns)):
        r = returns[i]

        # Determine direction
        if r > 0:
            new_dir = 'up'
        elif r < 0:
            new_dir = 'down'
        else:
            # Zero return - continue with current direction
            new_dir = current_dir if current_dir is not None else 'neutral'

        # Check if direction changed
        if new_dir != current_dir and current_dir is not None:
            # Save the previous segment
            end_idx = i
            segments.append((start_idx, end_idx, current_dir, current_streak))
            # Start new segment from current position
            start_idx = i
            current_streak = 1 if new_dir in ['up', 'down'] else 0
        else:
            if new_dir in ['up', 'down']:
                current_streak += 1

        current_dir = new_dir

    # Save the final segment
    if current_dir is not None and start_idx < len(data):
        segments.append((start_idx, len(data), current_dir, current_streak))

    # Get longest streaks
    max_up_streak = runs_info.get('max_up_streak', 0)
    max_down_streak = runs_info.get('max_down_streak', 0)

    # Track which segment types we've plotted for legend
    plotted_types = set()
    
    # Plot segments with special colors for longest streaks
    for start, end, direction, streak in segments:
        color = colors[direction]
        linewidth = 2
        alpha = 0.7
        label = None

        # Highlight longest streaks
        if direction == 'up' and streak == max_up_streak and max_up_streak > 0:
            color = 'blue'
            linewidth = 3
            alpha = 0.9
            if 'longest_up' not in plotted_types:
                label = f'Longest Up Streak ({streak}d)'
                plotted_types.add('longest_up')
        elif direction == 'down' and streak == max_down_streak and max_down_streak > 0:
            color = 'gold'
            linewidth = 3
            alpha = 0.9
            if 'longest_down' not in plotted_types:
                label = f'Longest Down Streak ({streak}d)'
                plotted_types.add('longest_down')
        elif direction == 'up':
            if 'up' not in plotted_types:
                label = 'Upward Run'
                plotted_types.add('up')
        elif direction == 'down':
            if 'down' not in plotted_types:
                label = 'Downward Run'
                plotted_types.add('down')

        plt.plot(
            data['Date'][start:end+1],
            data['Close'][start:end+1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label
        )

    plt.title(f'Runs of {ticker}', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_buysell_signals_for_all_stocks():
    """
    Plot SMA crossover buy/sell signals on main plot.
    
    Features:
        - Green ^ triangles for buy (price crosses above SMA)
        - Red v triangles for sell (price crosses below SMA)
        - Filters to current date range
        - Removes existing signals first
    
    Args:
        None - reads plot.indicators and plot.ax
    
    Returns:
        None
    
    Effects:
        Adds scatter plots to plot.lines, updates legend, refreshes canvas
    """
    from . import plot
    
    # Ensure we're using the interactive plot's axis
    if plot.ax is None:
        print("Error: No interactive plot axis available")
        return
    
    # Remove any existing signal scatters
    keys_to_remove = [k for k in plot.lines.keys() if k.startswith("Signal_")]
    for key in keys_to_remove:
        if key in plot.lines:
            try:
                plot.lines[key].remove()
            except:
                pass
            del plot.lines[key]
    
    # Plot signals for each stock on the interactive plot's axis
    for stock_key in [k for k in plot.lines if k.endswith("_Close")]:
        stock_name = stock_key.replace("_Close", "")
        
        # Check if this stock has signals calculated
        if (stock_name not in plot.indicators or 
            "Buy_Sell_Signals" not in plot.indicators[stock_name] or
            "Price_Index" not in plot.indicators[stock_name]):
            continue
        
        signals_dict = plot.indicators[stock_name]["Buy_Sell_Signals"]
        price_index = plot.indicators[stock_name]["Price_Index"]
        
        # Get the close price line for this stock
        close_line = plot.lines.get(f"{stock_name}_Close")
        if close_line is None:
            continue
        
        # Get x (dates) data from the line
        xdata = close_line.get_xdata()
        
        if len(xdata) == 0:
            continue
        
        # Plot buy signals using indices and prices from signals_dict
        buy_indices = signals_dict['buy_indices']
        buy_prices = signals_dict['buy_prices']
        
        if len(buy_indices) > 0:
            buy_dates = []
            for idx in buy_indices:
                if idx < len(xdata):
                    buy_dates.append(xdata[idx])
            
            if buy_dates and len(buy_dates) == len(buy_prices):
                # ALWAYS plot on the interactive plot's axis (ax)
                scatter_buy = plot.ax.scatter(buy_dates, buy_prices, 
                                        marker='^', color='green', s=100, 
                                        label=f'Buy {stock_name}', zorder=5)
                plot.lines[f"Signal_Buy_{stock_name}"] = scatter_buy
        
        # Plot sell signals using indices and prices from signals_dict
        sell_indices = signals_dict['sell_indices']
        sell_prices = signals_dict['sell_prices']
        
        if len(sell_indices) > 0:
            sell_dates = []
            for idx in sell_indices:
                if idx < len(xdata):
                    sell_dates.append(xdata[idx])
            
            if sell_dates and len(sell_dates) == len(sell_prices):
                # ALWAYS plot on the interactive plot's axis (ax)
                scatter_sell = plot.ax.scatter(sell_dates, sell_prices, 
                                         marker='v', color='red', s=100, 
                                         label=f'Sell {stock_name}', zorder=5)
                plot.lines[f"Signal_Sell_{stock_name}"] = scatter_sell
    
    plot.update_plot_properties()
    if plot.fig is not None:
        plot.fig.canvas.draw_idle()

