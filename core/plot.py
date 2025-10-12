"""
Main plotting module - orchestrates all plotting functionality.
Maintains global state and re-exports functions from specialized modules.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional

# ===================== Global State =====================
# Shared across all plotting modules
lines = {}  # all lines plotted
indicators = {}  # calculated indicators per stock: indicators[ticker][indicator_name] = data
controls: Dict[str, object] = {}
fig = None
ax = None
current_data: Optional[pd.DataFrame] = None
current_ticker: Optional[str] = None
current_period: Optional[str] = None  # Period for data download (e.g., '3y')
current_interval: Optional[str] = None  # Interval for data download (e.g., '1d')
current_sma_window: int = 5  # SMA window for signals

# ===================== Import from Submodules =====================

# Basic static plots
from .plot_basic import (
    plot_price_sma_and_runs
)

# Indicator functions
from .plot_indicators import (
    calculate_all_indicators,
    remove_indicator_for_all_stocks,
    plot_sma_for_all_stocks,
    plot_ema_for_all_stocks,
    plot_rsi_for_all_stocks,
    plot_bollinger_for_all_stocks
)

# Analysis visualizations
from .plot_analysis import (
    plot_runs_for_all_stocks,
    plot_runs_popup,
    plot_buysell_signals_for_all_stocks
)

# Interactive plot core
from .plot_interactive import (
    create_interactive_plot,
    create_controls,
    plot_initial_data,
    update_plot_properties,
    on_mode_change,
    handle_stock_input,
    add_stock,
    remove_stock,
    on_checkbox_change,
    auto_scale_y_axis,
    auto_resize_figure
)

# ===================== Re-export All Functions =====================
# This allows other modules to import from core.plot as before

__all__ = [
    # Global state
    'lines',
    'indicators',
    'controls',
    'fig',
    'ax',
    'current_data',
    'current_ticker',
    'current_period',
    'current_interval',
    'current_sma_window',
    
    # Basic plots
    'plot_price_sma_and_runs',
    
    # Indicator functions
    'calculate_all_indicators',
    'remove_indicator_for_all_stocks',
    'plot_sma_for_all_stocks',
    'plot_ema_for_all_stocks',
    'plot_rsi_for_all_stocks',
    'plot_bollinger_for_all_stocks',
    
    # Analysis
    'plot_runs_for_all_stocks',
    'plot_runs_popup',
    'plot_buysell_signals_for_all_stocks',
    
    # Interactive plot
    'create_interactive_plot',
    'create_controls',
    'plot_initial_data',
    'update_plot_properties',
    'on_mode_change',
    'handle_stock_input',
    'add_stock',
    'remove_stock',
    'on_checkbox_change',
    'auto_scale_y_axis',
    'auto_resize_figure',
]
