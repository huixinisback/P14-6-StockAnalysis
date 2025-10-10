import pandas as pd # using pandas for data manipulation
import yfinance as yf  # using yahoo finance API

def get_numeric_close(ticker: str, period: str, interval: str) -> pd.Series:
    """
    Download stock closing prices from Yahoo Finance.
    
    Features:
        - Fetches historical data via yfinance
        - Handles single/multi-ticker formats
        - Returns numeric series with datetime index
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        period: Time period ('1y', '2y', '3y', '5y', 'max')
        interval: Data frequency ('1d', '1wk', '1mo')
    
    Returns:
        pd.Series: Closing prices, sorted by date
        
    Raises:
        ValueError: No data returned or Close column missing
    """
    df = yf.download(
        tickers=ticker,          # pass a string for single ticker
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        actions=False
    )

    if df is None or df.empty:
        raise ValueError("No data returned from yfinance.")

    # Standard single-ticker columns
    if "Close" in df.columns and not isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]

    # Multi-index format (multiple tickers)
    elif isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)]
        else:
            close_cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == "Close"]
            if not close_cols:
                raise ValueError(f"'Close' level not found in multi-index columns: {df.columns}")
            close = df[close_cols[0]]

    # Fallback: Series or Adj Close
    elif isinstance(df, pd.Series):
        close = df
    else:
        if "Adj Close" in df.columns:
            close = df["Adj Close"]
        else:
            raise ValueError(f"Could not locate a 'Close' column. Columns: {df.columns}")

    # Normalize data
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = pd.to_numeric(close, errors="coerce")

    if close.isna().all():
        raise ValueError("Close series is all NaN after coercion.")

    return close
