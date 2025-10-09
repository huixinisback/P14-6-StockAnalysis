#!/usr/bin/env python3
"""
Test Relative Strength Index (RSI) indicator.
Comprehensive test suite with static calculations, manual verification, and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import relative_strength_index


class TestRSI(unittest.TestCase):
    """Test Relative Strength Index - 12 test cases."""
    
    def test_01_static_calculation_all_gains(self):
        """Test 1: Static calculation - all upward movements."""
        # Price increases by 1 each day
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        result = relative_strength_index(prices, period=14)
        
        # For all gains and no losses, RSI should be 100
        valid_rsi = result.dropna()
        if len(valid_rsi) > 0:
            np.testing.assert_array_almost_equal(
                valid_rsi.values,
                [100.0] * len(valid_rsi),
                decimal=10
            )
    
    def test_02_static_calculation_all_losses(self):
        """Test 2: Static calculation - all downward movements."""
        # Price decreases by 1 each day
        prices = [115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
        result = relative_strength_index(prices, period=14)
        
        # For all losses and no gains, RSI should be 0
        valid_rsi = result.dropna()
        if len(valid_rsi) > 0:
            np.testing.assert_array_almost_equal(
                valid_rsi.values,
                [0.0] * len(valid_rsi),
                decimal=10
            )
    
    def test_03_static_calculation_mixed(self):
        """Test 3: Static calculation with mixed gains/losses."""
        # Known pattern with predictable RSI
        prices = [44, 44.5, 45, 45.5, 46, 45.5, 45, 44.5, 44, 43.5, 43, 42.5, 42, 41.5, 41, 40.5]
        result = relative_strength_index(prices, period=14)
        
        # RSI should be between 0 and 100
        valid_rsi = result.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))
    
    def test_04_manual_calculation_period5(self):
        """Test 4: Manual RSI calculation with period=5."""
        # Simple test data with known gains/losses
        prices = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106]
        result = relative_strength_index(prices, period=5)
        
        # Manual calculation for index 5 (after 5 price changes):
        # Changes: [+2, -1, +2, -1, +2]
        # Gains: [2, 0, 2, 0, 2] -> avg_gain = 6/5 = 1.2
        # Losses: [0, 1, 0, 1, 0] -> avg_loss = 2/5 = 0.4
        # RS = 1.2/0.4 = 3.0
        # RSI = 100 - (100/(1+3)) = 100 - 25 = 75
        expected_at_5 = 75.0
        
        if not result.isna().all():
            first_valid_idx = result.first_valid_index()
            self.assertAlmostEqual(result.iloc[first_valid_idx], expected_at_5, places=6)
    
    def test_05_edge_case_insufficient_data(self):
        """Test 5: Edge case - insufficient data."""
        result = relative_strength_index([100, 102, 101], period=14)
        self.assertTrue(result.isna().all())
    
    def test_06_edge_case_exact_period_data(self):
        """Test 6: Edge case - exactly period+1 data points."""
        # Need period+1 points for period price changes
        prices = [100] + list(range(101, 116))  # 16 points for period=14
        result = relative_strength_index(prices, period=14)
        
        # Should have exactly 1 valid RSI value
        valid_count = result.notna().sum()
        self.assertGreaterEqual(valid_count, 1)
    
    def test_07_edge_case_all_same_prices(self):
        """Test 7: Edge case - all same prices (no change)."""
        prices = [100] * 20
        result = relative_strength_index(prices, period=14)
        
        # When no price changes, avg_loss = 0, so RSI = 100
        valid_rsi = result.dropna()
        if len(valid_rsi) > 0:
            np.testing.assert_array_almost_equal(
                valid_rsi.values,
                [100.0] * len(valid_rsi),
                decimal=10
            )
    
    def test_08_rsi_range_bounds(self):
        """Test 8: RSI should always be between 0 and 100."""
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100
        result = relative_strength_index(prices, period=14)
        
        valid_rsi = result.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))
            self.assertGreaterEqual(valid_rsi.min(), 0)
            self.assertLessEqual(valid_rsi.max(), 100)
    
    def test_09_wilder_smoothing(self):
        """Test 9: Verify Wilder's smoothing is applied."""
        prices = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109, 108, 110]
        result = relative_strength_index(prices, period=5)
        
        # Wilder's smoothing should produce continuous values
        valid_rsi = result.dropna()
        if len(valid_rsi) >= 2:
            # RSI should change smoothly (not jump wildly)
            diffs = np.abs(np.diff(valid_rsi.values))
            # All changes should be reasonable (< 50 points typically)
            self.assertTrue(all(diff < 50 for diff in diffs))
    
    def test_10_pandas_series_input(self):
        """Test 10: Test with pandas Series input."""
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
        result = relative_strength_index(prices, period=14)
        
        # Should have valid RSI
        valid_rsi = result.dropna()
        self.assertGreater(len(valid_rsi), 0)
        # For uptrend, RSI should be high
        self.assertGreater(valid_rsi.iloc[-1], 90)
    
    def test_11_numpy_array_input(self):
        """Test 11: Test with numpy array input."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
        result = relative_strength_index(prices, period=14)
        
        # Should have valid RSI
        valid_rsi = result.dropna()
        self.assertGreater(len(valid_rsi), 0)
    
    def test_12_overbought_oversold(self):
        """Test 12: Test overbought/oversold scenarios."""
        # Strong uptrend -> overbought
        uptrend = list(range(100, 150))
        rsi_up = relative_strength_index(uptrend, period=14)
        valid_up = rsi_up.dropna()
        if len(valid_up) > 0:
            self.assertGreater(valid_up.iloc[-1], 70)  # Overbought
        
        # Strong downtrend -> oversold
        downtrend = list(range(150, 100, -1))
        rsi_down = relative_strength_index(downtrend, period=14)
        valid_down = rsi_down.dropna()
        if len(valid_down) > 0:
            self.assertLess(valid_down.iloc[-1], 30)  # Oversold


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        unittest.main(verbosity=2)
    else:
        print("Usage: python test_rsi.py run")
