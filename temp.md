# How to Run

## Setup

```bash
pip install -r requirements.txt
```

---

## Main Application

### Interactive Plot (Default)
```bash
python main.py
```
- Prompts for: ticker, period, interval, SMA window
- Opens matplotlib window with controls
- Add/remove stocks, toggle indicators

### Interactive Plot (Specific Ticker)
```bash
python main.py MSFT
python main.py GOOGL
```

### Multiple Stock Comparison
```bash
python main.py multi
```
- Prompts for comma-separated tickers
- Creates comparison plot

---

## Streamlit Web App

```bash
streamlit run app.py
```
- Opens browser interface at `http://localhost:8501`
- Interactive controls in sidebar
- Plotly charts with zoom/pan

---

## Testing

### Run All Tests
```bash
cd tests
python run_tests.py all
```

### Run Specific Indicator Tests
```bash
python run_tests.py sma
python run_tests.py ema
python run_tests.py rsi
python run_tests.py bollinger
python run_tests.py signals
python run_tests.py profit
python run_tests.py daily
```

### Individual Test File
```bash
python tests/test_sma.py run
python tests/test_signals.py run
```

---

## Command Line Options

| Command | Action |
|---------|--------|
| `python main.py` | Interactive plot (prompts for config) |
| `python main.py AAPL` | Interactive plot for AAPL |
| `python main.py multi` | Multiple stock comparison |
| `streamlit run app.py` | Web dashboard |
| `python tests/run_tests.py all` | Run all tests |

---

## Configuration

Interactive mode prompts for:
- **Ticker:** Stock symbol (e.g., AAPL, MSFT, GOOGL)
- **Period:** 1y, 2y, 3y, 5y
- **Interval:** 1d (daily), 1wk (weekly), 1mo (monthly)
- **SMA Window:** 1-200 (default: 5)

---

## Interactive Plot Controls

### Radio Buttons
- **Add Stock** - Add ticker to plot
- **Remove Stock** - Remove ticker from plot

### Checkboxes (Moving Averages)
- SMA 5, SMA 10, SMA 20
- EMA 12, EMA 26

### Checkboxes (Momentum/Volatility)
- RSI 14 (secondary y-axis, 0-100 range)
- Bollinger Bands (upper, middle, lower + fill)

### Checkboxes (Signals)
- Buy/Sell Signals (SMA crossover triangles)
- Show Runs (popup window with colored streaks)


