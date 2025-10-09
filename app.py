# app.py
# Streamlit "mini-TradingView" using your core modules

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.data import get_numeric_close
from core.indicators import sma_sliding_window, daily_simple_returns, max_profit_multiple_transactions
from core.runs import find_up_down_runs

# -----------------------------
# Sidebar controls
# -----------------------------
st.set_page_config(page_title="Stock Trend Analysis", layout="wide")

st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL", help="Any yfinance-supported symbol")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
sma_window = st.sidebar.number_input("SMA window", min_value=2, max_value=200, value=5, step=1)

show_adjacent_stats = st.sidebar.checkbox("Show extra stats (returns)", value=True)
show_runs = st.sidebar.checkbox("Highlight up/down runs", value=True)
use_area_fills = st.sidebar.checkbox("Fill up/down runs (transparent)", value=True)
show_returns_hist = st.sidebar.checkbox("Show returns histogram", value=False)

st.sidebar.caption("Tip: zoom with drag, reset via double-click; use the range slider at bottom.")

# -----------------------------
# Data fetch + compute (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    close = get_numeric_close(ticker, period, interval)
    df = pd.DataFrame({"Close": close})
    return df

try:
    df = load_data(ticker, period, interval)
except Exception as e:
    st.error(f"Failed to load data for {ticker}: {e}")
    st.stop()

# Indicators / analytics
sma = sma_sliding_window(df["Close"], int(sma_window))
df[sma.name] = sma
df["Daily_Return"] = daily_simple_returns(df["Close"])
runs, summary = find_up_down_runs(df["Close"])
profit = max_profit_multiple_transactions(df["Close"])

# -----------------------------
# Header + KPIs
# -----------------------------
st.title("📈 Stock Market Trend Analysis")
st.subheader(f"{ticker} • Period: {period} • Interval: {interval}")

kpi_cols = st.columns(3)
kpi_cols[0].metric("SMA window", value=int(sma_window))
kpi_cols[1].metric("Max profit (multi-trades)", value=f"{profit:.2f}")
kpi_cols[2].metric("Data points", value=len(df))

# -----------------------------
# Plotly chart (interactive)
# -----------------------------
fig = go.Figure()

# Price
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="Close",
        hovertemplate="Date: %{x}<br>Close: %{y}<extra></extra>",
    )
)

# SMA
fig.add_trace(
    go.Scatter(
        x=df.index, y=df[sma.name], mode="lines", name=sma.name,
        hovertemplate="Date: %{x}<br>" + sma.name + ": %{y}<extra></extra>",
    )
)

# Shaded runs
if show_runs and runs:
    for r in runs:
        # color by direction
        color = "rgba(0, 200, 0, 0.12)" if r["direction"] == "up" else "rgba(255, 0, 0, 0.12)"
        line_color = "rgba(0,180,0,0.4)" if r["direction"] == "up" else "rgba(200,0,0,0.4)"
        # add translucent vertical rectangles
        if use_area_fills:
            fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=color, opacity=0.15, line_width=0)
        else:
            fig.add_vrect(x0=r["start"], x1=r["end"], line_color=line_color, fillcolor=None)

# Layout polish
fig.update_layout(
    height=600,
    margin=dict(l=10, r=10, t=40, b=0),
    xaxis=dict(
        rangeslider=dict(visible=True),
        showspikes=True, spikemode="across", spikesnap="cursor", showline=True
    ),
    yaxis=dict(showspikes=True, spikemode="across", showline=True),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Stats + tables
# -----------------------------
st.markdown("### Up/Down Runs Summary")
c1, c2 = st.columns(2)
with c1:
    st.json(summary["up"], expanded=False)
with c2:
    st.json(summary["down"], expanded=False)

# Runs table
if runs:
    runs_df = pd.DataFrame(runs)
    runs_df = runs_df[["start", "end", "direction", "length"]]
    runs_df = runs_df.sort_values("start").reset_index(drop=True)
    st.dataframe(runs_df, use_container_width=True)
    # Download
    csv = runs_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download runs as CSV", data=csv, file_name=f"{ticker}_runs.csv", mime="text/csv")

# Returns summary + histogram
if show_adjacent_stats:
    st.markdown("### Returns Summary")
    rets = df["Daily_Return"].dropna()
    if not rets.empty:
        st.write(
            pd.DataFrame({
                "Mean": [rets.mean()],
                "Std": [rets.std()],
                "Min": [rets.min()],
                "Max": [rets.max()],
                "Sharpe (naïve, daily)": [rets.mean() / (rets.std() + 1e-12)],
            }).T.rename(columns={0: "Value"})
        )

if show_returns_hist:
    rets = df["Daily_Return"].dropna()
    if not rets.empty:
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=rets, nbinsx=50, name="Daily returns"))
        hist.update_layout(
            title="Daily Returns Histogram",
            bargap=0.05,
            xaxis_title="Return",
            yaxis_title="Frequency",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(hist, use_container_width=True)

# Data preview + download
st.markdown("### Data Preview")
st.dataframe(df.tail(250), use_container_width=True)
csv_all = df.to_csv(index=True).encode("utf-8")
st.download_button("Download full dataset (CSV)", data=csv_all, file_name=f"{ticker}_{period}_{interval}.csv", mime="text/csv")

st.caption("Powered by yfinance • Built with Streamlit + Plotly • Uses your modular core for analytics.")
# Run: streamlit run app.py