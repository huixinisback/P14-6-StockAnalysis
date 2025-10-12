# Technical Indicators Test Suite

Simple test cases for technical indicators with command-line control.

## Structure

- `test_sma.py` - Simple Moving Average tests
- `test_daily_returns.py` - Daily Simple Returns tests
- `test_ema.py` - Exponential Moving Average tests
- `test_rsi.py` - Relative Strength Index tests
- `test_bollinger.py` - Bollinger Bands tests
- `test_macd.py` - MACD tests
- `run_tests.py` - Command-line test runner

## Usage

### Run All Tests
```bash
python run_tests.py all
```

### Run Specific Indicator
```bash
python run_tests.py sma
python run_tests.py ema
python run_tests.py rsi
```

### Run Individual Test File
```bash
python test_sma.py run
python test_ema.py run
```

## Indicators

- **sma** - Simple Moving Average
- **daily** - Daily Simple Returns
- **ema** - Exponential Moving Average
- **rsi** - Relative Strength Index
- **bollinger** - Bollinger Bands
- **macd** - MACD

## Examples

```bash
# Test SMA
python run_tests.py sma

# Test all indicators
python run_tests.py all

# Run individual test file
python test_sma.py run
```

## Test Cases Per Indicator

Each indicator has types of 3 test cases:

1. **Basic Test** - Tests core functionality with known values
2. **Edge Cases** - Tests boundary conditions and error handling
3. **API Comparison** - Compares results with pandas implementations
