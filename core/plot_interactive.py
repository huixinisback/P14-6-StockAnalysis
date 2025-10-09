"""
Interactive plotting system with controls.
Handles the main interactive plot window with add/remove stock controls and indicator checkboxes.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons, CheckButtons
import pandas as pd
from typing import Optional
from .data import get_numeric_close
from .plot_indicators import calculate_all_indicators, remove_indicator_for_all_stocks
from .plot_indicators import plot_sma_for_all_stocks, plot_ema_for_all_stocks
from .plot_indicators import plot_rsi_for_all_stocks, plot_bollinger_for_all_stocks
from .plot_analysis import plot_runs_for_all_stocks, plot_buysell_signals_for_all_stocks


def create_controls():
    """
    Create interactive UI controls for plot.
    
    Features:
        - Radio buttons: Add/Remove Stock mode
        - Text input: Ticker entry
        - 3 checkbox columns: Moving Averages, Momentum/Volatility, Signals
    
    Returns:
        None - stores controls in plot.controls dict
    """
    from . import plot
    
    plot.controls = {}
    
    # Radio buttons - bottom left
    ax_radio = plt.axes([0.12, 0.15, 0.15, 0.04])
    plot.controls['mode_radio'] = RadioButtons(ax_radio, ['Add Stock', 'Remove Stock'])
    plot.controls['mode_radio'].on_clicked(on_mode_change)
    
    # Text box - below radio buttons
    ax_text = plt.axes([0.12, 0.12, 0.12, 0.03])
    plot.controls['stock_input'] = TextBox(ax_text, 'Ticker: ', initial='')
    plot.controls['stock_input'].on_submit(handle_stock_input)
    
    # Checkboxes - organized by indicator type in columns
    
    # Column 1: Moving Averages
    moving_avg_labels = ["SMA 5", "SMA 10", "SMA 20", "EMA 12", "EMA 26"]
    ax_check1 = plt.axes([0.30, 0.01, 0.22, 0.2])
    ax_check1.set_title("Moving Averages", fontsize=7, fontweight='bold')
    plot.controls['checkboxes_ma'] = CheckButtons(ax_check1, moving_avg_labels)
    plot.controls['checkboxes_ma'].on_clicked(on_checkbox_change)
    for label in plot.controls['checkboxes_ma'].labels:
        label.set_fontsize(8)
    
    # Column 2: Momentum & Volatility
    momentum_labels = ["RSI 14", "Bollinger Bands"]
    ax_check2 = plt.axes([0.53, 0.01, 0.22, 0.2])
    ax_check2.set_title("Momentum/Volatility", fontsize=7, fontweight='bold')
    plot.controls['checkboxes_momentum'] = CheckButtons(ax_check2, momentum_labels)
    plot.controls['checkboxes_momentum'].on_clicked(on_checkbox_change)
    for label in plot.controls['checkboxes_momentum'].labels:
        label.set_fontsize(8)
    
    # Column 3: Signals & Runs
    trend_labels = ["Buy/Sell Signals", "Show Runs"]
    ax_check3 = plt.axes([0.76, 0.01, 0.22, 0.2])
    ax_check3.set_title("Signals", fontsize=7, fontweight='bold')
    plot.controls['checkboxes_trend'] = CheckButtons(ax_check3, trend_labels)
    plot.controls['checkboxes_trend'].on_clicked(on_checkbox_change)
    for label in plot.controls['checkboxes_trend'].labels:
        label.set_fontsize(8)


def plot_initial_data():
    """
    Plot initial stock (delegates to add_stock).
    
    Features:
        - Clears axes and resets lines dict
        - Calls add_stock for consistency
    """
    from . import plot
    
    # Ensure a clean axes for initial draw
    plot.ax.clear()
    plot.lines = {}
    # Reuse the same flow as adding a stock so indicators/legend/layout are consistent
    if plot.current_ticker:
        add_stock(plot.current_ticker)


def update_plot_properties():
    """
    Refresh plot title, labels, legend, and grid.
    
    Features:
        - Title shows ticker, period, interval, price, max profit
        - Updates legend with all active lines
        - Applies grid styling
    
    Called after adding/removing stocks/indicators
    """
    from . import plot
    
    # Build title with max profit and SMA window info if available
    title = f"Interactive Stock Analysis"
    
    plot.ax.set_title(title, fontsize=14, fontweight='bold')
    plot.ax.set_xlabel("Date", fontsize=12)
    plot.ax.set_ylabel("Price", fontsize=12)
    
    # Combine legends from main axis and RSI axis (if it exists)
    handles, labels = plot.ax.get_legend_handles_labels()
    
    # Add RSI axis legend if it exists
    if "RSI_Axis" in plot.lines:
        ax_rsi = plot.lines["RSI_Axis"]
        rsi_handles, rsi_labels = ax_rsi.get_legend_handles_labels()
        handles.extend(rsi_handles)
        labels.extend(rsi_labels)
    
    plot.ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
    plot.ax.grid(True, alpha=0.3)


def on_mode_change(label):
    """
    Handle Add/Remove mode switch (radio button callback).
    
    Args:
        label: 'Add Stock' or 'Remove Stock'
    
    Changes text input label accordingly
    """
    from . import plot
    
    if label == 'Add Stock':
        plot.controls['stock_input'].label.set_text('Add Ticker: ')
    else:
        plot.controls['stock_input'].label.set_text('Remove Ticker: ')


def handle_stock_input(text):
    """
    Process ticker submission (text box callback).
    
    Features:
        - Uppercases and strips input
        - Routes to add_stock() or remove_stock() based on mode
    
    Args:
        text: User-entered ticker symbol
    """
    from . import plot
    
    ticker = text.strip().upper()
    if not ticker:
        print("Please enter a valid stock symbol.")
        return
    mode = plot.controls['mode_radio'].value_selected
    if mode == 'Add Stock':
        add_stock(ticker)
    else:
        remove_stock(ticker)
    plot.controls['stock_input'].set_val('')


def add_stock(ticker):
    """
    Add stock to plot and calculate all indicators.
    
    Features:
        - Downloads 3y daily data
        - Calculates all indicators (SMA, EMA, RSI, BB, signals, runs, profit)
        - Plots price line
        - Re-applies active checkbox indicators
        - Prints max profit and signal count
        - Auto-resizes and scales
    
    Args:
        ticker: Stock symbol to add
    """
    from . import plot
    
    if ticker in [key.replace("_Close", "") for key in plot.lines.keys() if key.endswith("_Close")]:
        print(f"Stock {ticker} already plotted.")
        return
    try:
        close = get_numeric_close(ticker, "3y", "1d")
        data = pd.DataFrame({"Close": close})
        
        # Calculate indicators for this stock (use SMA_20 by default for additional stocks)
        calculate_all_indicators(data["Close"], ticker, sma_window=20)
        
        # Print max profit for this stock
        if ticker in plot.indicators and "Max_Profit" in plot.indicators[ticker]:
            max_profit = plot.indicators[ticker]["Max_Profit"]
            print(f"Max Profit for {ticker}: ${max_profit:.2f}")
            
            # Print buy/sell signal count
            signals_dict = plot.indicators[ticker].get("Buy_Sell_Signals", {})
            buy_count = len(signals_dict.get('buy_indices', []))
            sell_count = len(signals_dict.get('sell_indices', []))
            sma_window_used = plot.indicators[ticker].get("SMA_Window_Used", 20)
            print(f"Buy/Sell Signals (SMA {sma_window_used}): {buy_count} buys, {sell_count} sells")
        
        # Plot the stock price
        line, = plot.ax.plot(data.index, data["Close"], label=ticker, linewidth=2, alpha=0.8)
        plot.lines[f"{ticker}_Close"] = line
        
        # Re-plot any active indicators for all stocks
        # Collect states from all checkbox columns
        all_labels = []
        all_states = []
        
        for widget_key in ['checkboxes_ma', 'checkboxes_momentum', 'checkboxes_trend']:
            if widget_key in plot.controls:
                widget = plot.controls[widget_key]
                labels = [t.get_text() for t in widget.labels]
                states = widget.get_status()
                all_labels.extend(labels)
                all_states.extend(states)
        
        for i, is_checked in enumerate(all_states):
            if is_checked:
                label = all_labels[i]
                # Trigger the same plotting logic as checkbox change
                if label.startswith("SMA "):
                    period = label.split()[1]
                    plot_sma_for_all_stocks(period)
                elif label.startswith("EMA "):
                    period = label.split()[1]
                    plot_ema_for_all_stocks(period)
                elif label == "RSI 14":
                    plot_rsi_for_all_stocks()
                elif label == "Bollinger Bands":
                    plot_bollinger_for_all_stocks()
                elif label == "Buy/Sell Signals":
                    plot_buysell_signals_for_all_stocks()
                elif label == "Show Runs":
                    plot_runs_for_all_stocks()
        
        print(f"Added {ticker} to plot")
        update_plot_properties()
        auto_resize_figure()
    except Exception as e:
        print(f"Error adding stock {ticker}: {e}")


def remove_stock(ticker):
    """
    Remove stock and all its indicators from plot.
    
    Features:
        - Removes all lines matching ticker (price, SMAs, EMAs, RSI, BB, signals)
        - Clears from plot.lines and plot.indicators dicts
        - Updates title and legend
        - Auto-resizes figure
    
    Args:
        ticker: Stock symbol to remove
    """
    from . import plot
    
    keys_to_remove = [key for key in plot.lines.keys() if ticker in key]
    if not keys_to_remove:
        print(f"Stock {ticker} not found in plot.")
        return
    
    for key in keys_to_remove:
        obj = plot.lines[key]
        if key == "RSI_Axis":
            # Don't remove RSI axis, other stocks might still use it
            pass
        elif isinstance(obj, list):  # Fill areas
            for item in obj:
                item.remove()
            del plot.lines[key]
        elif hasattr(obj, 'remove'):
            obj.remove()
            del plot.lines[key]
        elif hasattr(obj, '__iter__'):  # Bar collection
            for bar in obj:
                bar.remove()
            del plot.lines[key]
    
    # Remove indicators for this stock
    if ticker in plot.indicators:
        del plot.indicators[ticker]
    
    print(f"Removed {ticker} from plot")
    update_plot_properties()
    auto_resize_figure()


def on_checkbox_change(label):
    """
    Toggle indicators on/off (checkbox callback).
    
    Features:
        - Plots indicator if checked, removes if unchecked
        - Handles: SMA, EMA, RSI, Bollinger Bands, Buy/Sell Signals, Show Runs
        - Show Runs opens popup (no removal logic)
    
    Args:
        label: Checkbox text (e.g., 'SMA 5', 'RSI 14')
    """
    from . import plot
    
    # Find which column contains this label and get its status
    is_checked = False
    found_widget = None
    
    for widget_key in ['checkboxes_ma', 'checkboxes_momentum', 'checkboxes_trend']:
        if widget_key in plot.controls:
            widget = plot.controls[widget_key]
            labels = [t.get_text() for t in widget.labels]
            if label in labels:
                checkbox_states = widget.get_status()
                is_checked = checkbox_states[labels.index(label)]
                found_widget = widget_key
                break
    
    if label.startswith("SMA "):
        period = label.split()[1]
        if is_checked:
            plot_sma_for_all_stocks(period)
        else:
            remove_indicator_for_all_stocks(f"SMA_{period}")
    elif label.startswith("EMA "):
        period = label.split()[1]
        if is_checked:
            plot_ema_for_all_stocks(period)
        else:
            remove_indicator_for_all_stocks(f"EMA_{period}")
    elif label == "RSI 14":
        if is_checked:
            plot_rsi_for_all_stocks()
        else:
            remove_indicator_for_all_stocks("RSI_14")
    elif label == "Bollinger Bands":
        if is_checked:
            plot_bollinger_for_all_stocks()
        else:
            remove_indicator_for_all_stocks("BB_")
    elif label == "Buy/Sell Signals":
        if is_checked:
            plot_buysell_signals_for_all_stocks()
        else:
            remove_indicator_for_all_stocks("Signal_")
    elif label == "Show Runs":
        if is_checked:
            plot_runs_for_all_stocks()
        # No need to remove anything when unchecked - runs are in separate window
    plot.fig.canvas.draw_idle()


def auto_scale_y_axis():
    """
    Auto-scale y-axis to fit all price data (5% padding).
    
    Features:
        - Finds min/max from all *_Close lines
        - Adds 5% padding above/below
    """
    from . import plot
    
    if not plot.lines:
        return
    y_data = []
    for key, line in plot.lines.items():
        if key.endswith("_Close") and hasattr(line, 'get_ydata'):
            y_data.extend(line.get_ydata())
    if y_data:
        y_min = min(y_data)
        y_max = max(y_data)
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        plot.ax.set_ylim(y_min - padding, y_max + padding)


def auto_resize_figure():
    """
    Dynamically resize figure based on stock/indicator count.
    
    Features:
        - Base height + extra per stock/indicator
        - Prevents overcrowding
        - Calls draw_idle() to refresh
    """
    from . import plot
    
    stock_count = len([k for k in plot.lines if k.endswith("_Close")])
    
    # Count active indicators across all checkbox columns
    active_indicators = 0
    for widget_key in ['checkboxes_ma', 'checkboxes_momentum', 'checkboxes_trend']:
        if widget_key in plot.controls:
            checkbox_states = plot.controls[widget_key].get_status()
            active_indicators += sum(checkbox_states) if checkbox_states is not None else 0
    base = 8
    height = base + max(0, stock_count - 1) * 1.5 + max(0, active_indicators - 3) * 0.5
    if plot.fig is not None:
        plot.fig.set_size_inches(12, height)
        plt.subplots_adjust(bottom=0.15, left=0.08, right=0.95, top=0.95, hspace=0.3)
    auto_scale_y_axis()
    if plot.fig is not None:
        plot.fig.canvas.draw_idle()


def create_interactive_plot(ticker: str = "AAPL", period: str = "3y", interval: str = "1d", 
                           sma_window: int = 5, print_results_func=None):
    """
    Main entry point - create interactive stock analysis plot.
    
    Features:
        - Downloads initial stock data
        - Calculates all indicators
        - Creates figure with controls (radio, text, checkboxes)
        - Auto-resizes based on content
        - Optional console output via print_results_func
    
    Args:
        ticker: Initial stock symbol (default 'AAPL')
        period: Time period (default '3y')
        interval: Data frequency (default '1d')
        sma_window: SMA for signals (default 5)
        print_results_func: Optional callback(ticker, period, interval, close, indicators)
    
    Returns:
        None - displays via plt.show()
    """
    from . import plot
    
    close_prices = get_numeric_close(ticker, period, interval)
    plot.current_data = pd.DataFrame({"Close": close_prices})
    plot.current_ticker = ticker
    
    # Calculate indicators for the initial stock with ticker parameter
    calculate_all_indicators(close_prices, ticker, sma_window)
    
    # Print comprehensive analysis results if function provided
    if print_results_func is not None:
        print_results_func(ticker, period, interval, close_prices, plot.indicators[ticker])
    
    plot.fig = plt.figure(figsize=(12, 8))
    plot.ax = plt.subplot2grid((5, 7), (0, 0), colspan=6, rowspan=4)
    create_controls()
    plot_initial_data()
    plt.subplots_adjust(bottom=0.15, left=0.08, right=0.95, top=0.95, hspace=0.3)
    auto_resize_figure()
    plt.show()

