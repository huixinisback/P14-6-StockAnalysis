from collections import deque
import numpy as np
import pandas as pd
from typing import Union, List


def sma_sliding_window(values: Union[pd.Series, List, np.ndarray], k: int) -> pd.Series:
    """
    Calculate Simple Moving Average using sliding window (O(n) algorithm).
    
    Features:
        - Maintains running sum for efficiency
        - First (k-1) values are NaN
        - Named output: 'SMA_{k}'
    
    Args:
        values: Price data (Series, List, or ndarray)
        k: Window size (must be > 0)
    
    Returns:
        pd.Series: SMA values, NaN for first (k-1) positions
    """
    # Convert to numpy array for efficient processing
    if isinstance(values, pd.Series):
        data = values.values
        index = values.index
    else:
        data = np.array(values)
        index = range(len(data))
    
    n = len(data)
    if n == 0:
        return pd.Series([], name=f"SMA_{k}")
    
    # Initialize output array
    sma_values = np.full(n, np.nan, dtype=float)
    
    # Handle edge case
    if k <= 0 or k > n:
        return pd.Series(sma_values, index=index, name=f"SMA_{k}")
    
    # Calculate first valid SMA
    window_sum = np.sum(data[:k])
    if not np.isnan(window_sum):
        sma_values[k-1] = window_sum / k
    
    # Sliding window: O(n) time complexity
    for i in range(k, n):
        # Remove oldest value, add newest value
        window_sum = window_sum - data[i-k] + data[i]
        if not np.isnan(window_sum):
            sma_values[i] = window_sum / k
    
    return pd.Series(sma_values, index=index, name=f"SMA_{k}")


def daily_simple_returns(close: Union[pd.Series, List, np.ndarray]) -> pd.Series:
    """
    Calculate daily simple returns: (P_t - P_{t-1}) / P_{t-1}
    
    Features:
        - Returns as decimals (0.05 = 5% gain)
        - First value is NaN
        - Handles zero and NaN values
    
    Args:
        close: Closing prices (Series, List, or ndarray)
    
    Returns:
        pd.Series: Named 'Daily_Return', O(n) time
    """
    # Convert to numpy array
    if isinstance(close, pd.Series):
        data = close.values
        index = close.index
    else:
        data = np.array(close)
        index = range(len(data))
    
    n = len(data)
    if n <= 1:
        return pd.Series([np.nan] * n, index=index, name="Daily_Return")
    
    # Calculate returns: O(n) time complexity
    returns = np.full(n, np.nan, dtype=float)
    
    for i in range(1, n):
        if not np.isnan(data[i]) and not np.isnan(data[i-1]) and data[i-1] != 0:
            returns[i] = (data[i] - data[i-1]) / data[i-1]
    
    return pd.Series(returns, index=index, name="Daily_Return")


def exponential_moving_average(values: Union[pd.Series, List, np.ndarray], 
                              alpha: float = None, period: int = None) -> pd.Series:
    """
    Calculate Exponential Moving Average (weighted toward recent prices).
    
    Features:
        - Formula: EMA_t = alpha * Price_t + (1-alpha) * EMA_{t-1}
        - Specify either alpha (0 < alpha <= 1) or period
        - Period converts to alpha = 2/(period+1)
    
    Args:
        values: Price data (Series, List, or ndarray)
        alpha: Smoothing factor (optional)
        period: Lookback period (optional)
    
    Returns:
        pd.Series: Named 'EMA_{period}', O(n) time
    
    Raises:
        ValueError: Neither alpha nor period specified
    """
    # Convert to numpy array
    if isinstance(values, pd.Series):
        data = values.values
        index = values.index
    else:
        data = np.array(values)
        index = range(len(data))
    
    n = len(data)
    if n == 0:
        return pd.Series([], name="EMA")
    
    # Calculate alpha if period is provided
    if alpha is None:
        if period is None or period <= 0:
            raise ValueError("Either alpha or period must be specified")
        alpha = 2.0 / (period + 1)
    elif not (0 < alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    # Initialize EMA array
    ema_values = np.full(n, np.nan, dtype=float)
    
    # Find first valid value
    first_valid_idx = None
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx is None:
        return pd.Series(ema_values, index=index, name="EMA")
    
    # Initialize EMA with first valid value
    ema_values[first_valid_idx] = data[first_valid_idx]
    
    # Calculate EMA: O(n) time complexity
    for i in range(first_valid_idx + 1, n):
        if not np.isnan(data[i]):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
        else:
            ema_values[i] = ema_values[i-1]  # Forward-fill on NaN
    
    return pd.Series(ema_values, index=index, name=f"EMA_{period or alpha}")


def relative_strength_index(values: Union[pd.Series, List, np.ndarray], 
                           period: int = 14) -> pd.Series:
    """
    Calculate RSI momentum oscillator (0-100 range).
    
    Features:
        - RSI > 70 = overbought, < 30 = oversold
        - Uses Wilder's smoothing
        - Formula: RSI = 100 - (100 / (1 + avg_gain/avg_loss))
    
    Args:
        values: Price data (Series, List, or ndarray)
        period: Lookback period (default 14)
    
    Returns:
        pd.Series: Named 'RSI_{period}', first 'period' values NaN, O(n) time
    """
    # Convert to numpy array
    if isinstance(values, pd.Series):
        data = values.values
        index = values.index
    else:
        data = np.array(values)
        index = range(len(data))
    
    n = len(data)
    if n <= period:
        return pd.Series([np.nan] * n, index=index, name=f"RSI_{period}")
    
    # Calculate price changes
    price_changes = np.diff(data)
    
    # Separate gains and losses
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    
    # Initialize RSI array
    rsi_values = np.full(n, np.nan, dtype=float)
    
    # Calculate initial average gain and loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        rsi_values[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100 - (100 / (1 + rs))
    
    # Calculate RSI using Wilder's smoothing: O(n) time complexity
    for i in range(period + 1, n):
        change_idx = i - 1  # Map to price_changes array index
        
        # Update average gain and loss using Wilder's smoothing
        avg_gain = ((avg_gain * (period - 1)) + gains[change_idx]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[change_idx]) / period
        
        if avg_loss == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi_values, index=index, name=f"RSI_{period}")


def bollinger_bands(values: Union[pd.Series, List, np.ndarray], 
                   period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands (volatility channels).
    
    Features:
        - Middle: SMA
        - Upper: SMA + (std_dev × σ)
        - Lower: SMA - (std_dev × σ)
        - Bands widen in high volatility, narrow in low
    
    Args:
        values: Price data (Series, List, or ndarray)
        period: MA period (default 20)
        std_dev: Std dev multiplier (default 2.0)
    
    Returns:
        tuple: (middle, upper, lower) as pd.Series, first (period-1) NaN
    
    Algorithm: O(n) sliding window for both SMA and std deviation
    """
    # Convert to numpy array
    if isinstance(values, pd.Series):
        data = values.values
        index = values.index
    else:
        data = np.array(values)
        index = range(len(data))
    
    n = len(data)
    if n < period:
        nan_series = pd.Series([np.nan] * n, index=index)
        return nan_series, nan_series, nan_series
    
    # Calculate middle band (SMA), preserve datetime index if Series input
    if isinstance(values, pd.Series):
        middle_band = sma_sliding_window(values, period)
    else:
        middle_band = sma_sliding_window(data, period)
        middle_band.index = index
    
    # Initialize bands
    upper_band = np.full(n, np.nan, dtype=float)
    lower_band = np.full(n, np.nan, dtype=float)
    
    # Sliding window variance: Var = E[X²] - E[X]²
    # Maintain running sums for O(n) calculation
    
    # Initialize first window
    window_sum = np.sum(data[:period])
    window_sum_sq = np.sum(data[:period] ** 2)
    
    if not np.isnan(window_sum):
        mean = window_sum / period
        variance = (window_sum_sq / period) - (mean ** 2)
        std = np.sqrt(max(0, variance))  # Clamp to avoid negative from float error
        middle_val = middle_band.iloc[period - 1]
        if not np.isnan(middle_val):
            upper_band[period - 1] = middle_val + (std_dev * std)
            lower_band[period - 1] = middle_val - (std_dev * std)
    
    # Slide window: remove oldest, add newest
    for i in range(period, n):
        old_val = data[i - period]
        new_val = data[i]
        
        window_sum = window_sum - old_val + new_val
        window_sum_sq = window_sum_sq - (old_val ** 2) + (new_val ** 2)
        
        if not np.isnan(window_sum):
            mean = window_sum / period
            variance = (window_sum_sq / period) - (mean ** 2)
            std = np.sqrt(max(0, variance))
            middle_val = middle_band.iloc[i]
            if not np.isnan(middle_val):
                upper_band[i] = middle_val + (std_dev * std)
                lower_band[i] = middle_val - (std_dev * std)
    
    upper_series = pd.Series(upper_band, index=index, name=f"BB_Upper_{period}")
    lower_series = pd.Series(lower_band, index=index, name=f"BB_Lower_{period}")
    
    return middle_band, upper_series, lower_series


def compute_buy_sell_signals(prices: np.ndarray, sma: np.ndarray) -> dict:
    """
    Generate buy/sell signals from SMA crossovers with position tracking.
    
    Features:
        - BUY: Price crosses above SMA (only when not holding)
        - SELL: Price crosses below SMA (only when holding)
        - Enforces BUY → SELL → BUY → SELL sequence
        - Skips NaN values
        - Single pass O(n) algorithm
    
    Args:
        prices: Price array (same length as sma)
        sma: SMA values (may contain NaN)
    
    Returns:
        dict: 'buy_indices', 'sell_indices', 'buy_prices', 'sell_prices'
              Empty lists if no signals or invalid input
              Arrays are always paired: len(buy) == len(sell) or len(buy) == len(sell)+1
    
    Example:
        prices=[10,12,11,13,9], sma=[nan,nan,11.5,12,11]
        → buy at i=3, sell at i=4
    """
    if len(prices) < 2 or len(sma) < 2 or len(prices) != len(sma):
        return {'buy_indices': [], 'sell_indices': [], 'buy_prices': [], 'sell_prices': []}

    buy_indices = []
    sell_indices = []
    buy_prices = []
    sell_prices = []

    # Find first valid SMA index
    start_idx = None
    for i in range(len(sma)):
        if not np.isnan(sma[i]):
            start_idx = i
            break

    if start_idx is None or start_idx >= len(prices) - 1:
        return {'buy_indices': [], 'sell_indices': [], 'buy_prices': [], 'sell_prices': []}

    # Position tracking: 0 = no position (flat), 1 = holding (long)
    position = 0
    
    # Check for crossovers starting from first valid SMA + 1
    for i in range(start_idx + 1, len(prices)):
        if np.isnan(sma[i]) or np.isnan(sma[i-1]):
            continue

        # Buy signal: price crosses above SMA (only if not already holding)
        if position == 0 and prices[i] > sma[i] and prices[i-1] <= sma[i-1]:
            buy_indices.append(i)
            buy_prices.append(float(prices[i]))
            position = 1  # Now holding

        # Sell signal: price crosses below SMA (only if holding)
        elif position == 1 and prices[i] < sma[i] and prices[i-1] >= sma[i-1]:
            sell_indices.append(i)
            sell_prices.append(float(prices[i]))
            position = 0  # Now flat

    return {
        'buy_indices': buy_indices,
        'sell_indices': sell_indices,
        'buy_prices': buy_prices,
        'sell_prices': sell_prices
    }


def max_profit_multiple_transactions(close: pd.Series, return_transactions: bool = False):
    """
    Calculate max profit from unlimited buy/sell transactions (greedy).
    
    Features:
        - Captures all upward price movements
        - Sums all positive day-to-day differences
        - Optionally returns individual transactions
        - O(n) time, O(1) space
    
    Args:
        close: Closing prices
        return_transactions: If True, returns (profit, transactions_list)
    
    Returns:
        float: Max profit (0.0 if <2 prices or decreasing)
        OR tuple: (profit, transactions) if return_transactions=True
            transactions is list of dicts with:
                'buy_index', 'sell_index', 'buy_price', 'sell_price', 'profit'
    
    Example:
        [7,1,5,3,6,4] → (5-1) + (6-3) = 7
    """
    if len(close) < 2:
        if return_transactions:
            return 0.0, []
        return 0.0
    
    profit = 0.0
    transactions = []
    
    # Track buy position
    buy_idx = None
    buy_price = None
    
    for i in range(1, len(close)):
        price_going_up = close.iloc[i] > close.iloc[i-1]
        
        if price_going_up:
            # Start a buy if not already holding
            if buy_idx is None:
                buy_idx = i - 1
                buy_price = close.iloc[i-1]
        else:
            # Price going down - sell if holding
            if buy_idx is not None:
                sell_idx = i - 1
                sell_price = close.iloc[i-1]
                trade_profit = sell_price - buy_price
                profit += trade_profit
                
                if return_transactions:
                    transactions.append({
                        'buy_index': buy_idx,
                        'sell_index': sell_idx,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'profit': trade_profit
                    })
                
                buy_idx = None
                buy_price = None
    
    # If still holding at end, sell at last price
    if buy_idx is not None:
        sell_idx = len(close) - 1
        sell_price = close.iloc[-1]
        trade_profit = sell_price - buy_price
        profit += trade_profit
        
        if return_transactions:
            transactions.append({
                'buy_index': buy_idx,
                'sell_index': sell_idx,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'profit': trade_profit
            })
    
    if return_transactions:
        return profit, transactions
    return profit
