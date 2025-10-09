#!/usr/bin/env python3
"""
Test Daily Simple Returns indicator.
Comprehensive test suite with static calculations, pandas comparison, and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import daily_simple_returns


class TestDailyReturns(unittest.TestCase):
    """Test Daily Simple Returns - 12 test cases."""
    
    def test_01_static_calculation_basic(self):
        """Test 1: Static calculation with known values."""
        prices = [100, 102, 101, 103]
        result = daily_simple_returns(prices)
        
        # Manual calculation:
        # Return[0] = NaN (no previous price)
        # Return[1] = (102-100)/100 = 2/100 = 0.02 (2%)
        # Return[2] = (101-102)/102 = -1/102 = -0.00980392
        # Return[3] = (103-101)/101 = 2/101 = 0.01980198
        expected = [np.nan, 0.02, -0.00980392, 0.01980198]
        
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_02_static_calculation_percentage(self):
        """Test 2: Static calculation verifying percentages."""
        prices = [100, 110, 99, 105.6]
        result = daily_simple_returns(prices)
        
        # Manual calculation:
        # Return[1] = (110-100)/100 = 10/100 = 0.10 (10% gain)
        # Return[2] = (99-110)/110 = -11/110 = -0.10 (10% loss)
        # Return[3] = (105.6-99)/99 = 6.6/99 = 0.066666... (6.67% gain)
        expected = [np.nan, 0.10, -0.10, 0.0666666667]
        
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_03_pandas_comparison(self):
        """Test 3: Compare with pandas pct_change."""
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100
        df = pd.DataFrame({'Close': prices})
        
        our_returns = daily_simple_returns(prices)
        pandas_returns = df['Close'].pct_change()
        
        np.testing.assert_array_almost_equal(
            our_returns.values, 
            pandas_returns.values, 
            decimal=10
        )
    
    def test_04_edge_case_empty(self):
        """Test 4: Edge case - empty array."""
        result = daily_simple_returns([])
        self.assertEqual(len(result), 0)
    
    def test_05_edge_case_single_value(self):
        """Test 5: Edge case - single value."""
        result = daily_simple_returns([100])
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result.iloc[0]))
    
    def test_06_edge_case_two_values(self):
        """Test 6: Edge case - two values."""
        result = daily_simple_returns([100, 105])
        expected = [np.nan, 0.05]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=10)
    
    def test_07_edge_case_zero_price(self):
        """Test 7: Edge case - zero price (division by zero)."""
        result = daily_simple_returns([100, 0, 102])
        
        # Return[1] = (0-100)/100 = -1.0
        # Return[2] = NaN (division by zero: previous price is 0)
        self.assertAlmostEqual(result.iloc[1], -1.0, places=10)
        self.assertTrue(np.isnan(result.iloc[2]))
    
    def test_08_edge_case_nan_values(self):
        """Test 8: Edge case - NaN values in data."""
        result = daily_simple_returns([100, np.nan, 102])
        
        # Return[0] = NaN
        # Return[1] = NaN (current is NaN)
        # Return[2] = NaN (previous is NaN)
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertTrue(np.isnan(result.iloc[2]))
    
    def test_09_edge_case_all_same_prices(self):
        """Test 9: Edge case - all same prices (zero returns)."""
        prices = [100] * 10
        result = daily_simple_returns(prices)
        
        # All returns should be 0.0 except first (NaN)
        self.assertTrue(np.isnan(result.iloc[0]))
        for i in range(1, 10):
            self.assertAlmostEqual(result.iloc[i], 0.0, places=10)
    
    def test_10_pandas_series_input(self):
        """Test 10: Test with pandas Series input."""
        prices = pd.Series([100, 102, 101, 103])
        result = daily_simple_returns(prices)
        expected = [np.nan, 0.02, -0.00980392, 0.01980198]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_11_numpy_array_input(self):
        """Test 11: Test with numpy array input."""
        prices = np.array([100, 102, 101, 103])
        result = daily_simple_returns(prices)
        expected = [np.nan, 0.02, -0.00980392, 0.01980198]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)
    
    def test_12_large_gains_losses(self):
        """Test 12: Test with large price movements."""
        prices = [100, 200, 50, 150]
        result = daily_simple_returns(prices)
        
        # Manual calculation:
        # Return[1] = (200-100)/100 = 1.0 (100% gain)
        # Return[2] = (50-200)/200 = -150/200 = -0.75 (75% loss)
        # Return[3] = (150-50)/50 = 100/50 = 2.0 (200% gain)
        expected = [np.nan, 1.0, -0.75, 2.0]
        
        np.testing.assert_array_almost_equal(result.values, expected, decimal=10)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        unittest.main(verbosity=2)
    else:
        print("Usage: python test_daily_returns.py run")
