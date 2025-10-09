#!/usr/bin/env python3
"""
Test Exponential Moving Average (EMA) indicator.
Comprehensive test suite with static calculations, pandas comparison, and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import exponential_moving_average


class TestEMA(unittest.TestCase):
    """Test Exponential Moving Average - 12 test cases."""
    
    def test_01_static_calculation_period3(self):
        """Test 1: Static calculation with period=3 (alpha=0.5)."""
        prices = [100, 102, 101, 103, 105]
        result = exponential_moving_average(prices, period=3)
        
        # Manual calculation with alpha = 2/(3+1) = 0.5:
        # EMA[0] = 100 (first value)
        # EMA[1] = 0.5*102 + 0.5*100 = 51 + 50 = 101.0
        # EMA[2] = 0.5*101 + 0.5*101 = 50.5 + 50.5 = 101.0
        # EMA[3] = 0.5*103 + 0.5*101 = 51.5 + 50.5 = 102.0
        # EMA[4] = 0.5*105 + 0.5*102 = 52.5 + 51 = 103.5
        expected = [100.0, 101.0, 101.0, 102.0, 103.5]
        
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_02_static_calculation_alpha(self):
        """Test 2: Static calculation with explicit alpha=0.5."""
        prices = [100, 102, 101, 103]
        result = exponential_moving_average(prices, alpha=0.5)
        
        # Should be same as period=3
        expected = [100.0, 101.0, 101.0, 102.0]
        
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_03_static_calculation_alpha025(self):
        """Test 3: Static calculation with alpha=0.25."""
        prices = [100, 104, 108]
        result = exponential_moving_average(prices, alpha=0.25)
        
        # Manual calculation with alpha = 0.25:
        # EMA[0] = 100
        # EMA[1] = 0.25*104 + 0.75*100 = 26 + 75 = 101.0
        # EMA[2] = 0.25*108 + 0.75*101 = 27 + 75.75 = 102.75
        expected = [100.0, 101.0, 102.75]
        
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_04_pandas_comparison(self):
        """Test 4: Compare with pandas ewm."""
        np.random.seed(42)  # Deterministic
        prices = np.random.randn(100) * 10 + 100
        df = pd.DataFrame({'Close': prices})
        
        our_ema = exponential_moving_average(prices, period=12)
        pandas_ema = df['Close'].ewm(span=12, adjust=False).mean()
        
        np.testing.assert_array_almost_equal(
            our_ema.values, 
            pandas_ema.values, 
            decimal=8
        )
    
    def test_05_edge_case_empty(self):
        """Test 5: Edge case - empty array."""
        result = exponential_moving_average([], period=5)
        self.assertEqual(len(result), 0)
    
    def test_06_edge_case_single_value(self):
        """Test 6: Edge case - single value."""
        result = exponential_moving_average([100], period=5)
        self.assertEqual(result.iloc[0], 100.0)
    
    def test_07_edge_case_two_values(self):
        """Test 7: Edge case - two values."""
        prices = [100, 110]
        result = exponential_moving_average(prices, period=5)
        # alpha = 2/(5+1) = 0.333...
        # EMA[0] = 100
        # EMA[1] = 0.333*110 + 0.667*100 = 36.67 + 66.7 = 103.33
        expected = [100.0, 103.333333]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=5)
    
    def test_08_edge_case_no_alpha_no_period(self):
        """Test 8: Edge case - neither alpha nor period specified."""
        with self.assertRaises(ValueError):
            exponential_moving_average([100, 102], alpha=None, period=None)
    
    def test_09_edge_case_invalid_alpha(self):
        """Test 9: Edge case - invalid alpha value."""
        with self.assertRaises(ValueError):
            exponential_moving_average([100, 102], alpha=1.5)  # > 1
        
        with self.assertRaises(ValueError):
            exponential_moving_average([100, 102], alpha=0)  # <= 0
    
    def test_10_pandas_series_input(self):
        """Test 10: Test with pandas Series input."""
        prices = pd.Series([100, 102, 101, 103, 105])
        result = exponential_moving_average(prices, period=3)
        expected = [100.0, 101.0, 101.0, 102.0, 103.5]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_11_numpy_array_input(self):
        """Test 11: Test with numpy array input."""
        prices = np.array([100, 102, 101, 103, 105])
        result = exponential_moving_average(prices, period=3)
        expected = [100.0, 101.0, 101.0, 102.0, 103.5]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_12_period_conversion(self):
        """Test 12: Verify period correctly converts to alpha."""
        prices = [100, 110]
        # For period=5, alpha should be 2/(5+1) = 1/3
        result_period = exponential_moving_average(prices, period=5)
        result_alpha = exponential_moving_average(prices, alpha=1/3)
        
        np.testing.assert_array_almost_equal(
            result_period.values,
            result_alpha.values,
            decimal=10
        )


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        unittest.main(verbosity=2)
    else:
        print("Usage: python test_ema.py run")
