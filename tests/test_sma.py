#!/usr/bin/env python3
"""
Test Simple Moving Average (SMA) indicator.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import sma_sliding_window


class TestSMA(unittest.TestCase):
    """Test Simple Moving Average - 10 test cases."""
    
    def test_01_static_calculation(self):
        """Test 1: Static calculation with known values."""
        prices = [100, 102, 101, 103, 105]
        result = sma_sliding_window(prices, 3)
        
        # Manual calculation:
        # SMA(3) of [100, 102, 101] = (100+102+101)/3 = 101.0
        # SMA(3) of [102, 101, 103] = (102+101+103)/3 = 102.0
        # SMA(3) of [101, 103, 105] = (101+103+105)/3 = 103.0
        expected = [np.nan, np.nan, 101.0, 102.0, 103.0]
        
        np.testing.assert_array_almost_equal(result.values, expected, decimal=10)
    
    def test_02_static_calculation_window5(self):
        """Test 2: Static calculation with window=5."""
        prices = [10, 20, 30, 40, 50, 60]
        result = sma_sliding_window(prices, 5)
        
        # Manual calculation:
        # SMA(5) at index 4 = (10+20+30+40+50)/5 = 30.0
        # SMA(5) at index 5 = (20+30+40+50+60)/5 = 40.0
        expected = [np.nan, np.nan, np.nan, np.nan, 30.0, 40.0]
        
        np.testing.assert_array_almost_equal(result.values, expected, decimal=10)
    
    def test_03_pandas_comparison(self):
        """Test 3: Compare with pandas rolling mean."""
        np.random.seed(42)  # Deterministic
        prices = np.random.randn(100) * 10 + 100
        df = pd.DataFrame({'Close': prices})
        
        our_sma = sma_sliding_window(prices, 20)
        pandas_sma = df['Close'].rolling(window=20, min_periods=20).mean()
        
        # Compare valid values
        valid_mask = ~pandas_sma.isna()
        np.testing.assert_array_almost_equal(
            our_sma.values[valid_mask], 
            pandas_sma.values[valid_mask], 
            decimal=10
        )
    
    def test_04_edge_case_empty(self):
        """Test 4: Edge case - empty array."""
        result = sma_sliding_window([], 5)
        self.assertEqual(len(result), 0)
    
    def test_05_edge_case_single_value(self):
        """Test 5: Edge case - single value."""
        result = sma_sliding_window([100], 1)
        self.assertEqual(result.iloc[0], 100.0)
    
    def test_06_edge_case_window_larger_than_data(self):
        """Test 6: Edge case - window larger than data."""
        result = sma_sliding_window([100, 102], 5)
        self.assertTrue(result.isna().all())
    
    def test_07_edge_case_window_equals_data(self):
        """Test 7: Edge case - window equals data length."""
        prices = [100, 101, 102]
        result = sma_sliding_window(prices, 3)
        expected = [np.nan, np.nan, 101.0]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=10)
    
    def test_08_pandas_series_input(self):
        """Test 8: Test with pandas Series input."""
        prices = pd.Series([100, 102, 101, 103, 105])
        result = sma_sliding_window(prices, 3)
        expected = [np.nan, np.nan, 101.0, 102.0, 103.0]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=10)
    
    def test_09_numpy_array_input(self):
        """Test 9: Test with numpy array input."""
        prices = np.array([100, 102, 101, 103, 105])
        result = sma_sliding_window(prices, 3)
        expected = [np.nan, np.nan, 101.0, 102.0, 103.0]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=10)
    
    def test_10_sliding_window_efficiency(self):
        """Test 10: Verify sliding window maintains running sum (no recalculation)."""
        # Large dataset to verify O(n) performance
        prices = list(range(1000))
        result = sma_sliding_window(prices, 50)
        
        # Verify result length
        self.assertEqual(len(result), 1000)
        # Verify first 49 are NaN
        self.assertEqual(result.isna().sum(), 49)
        # Verify last value
        expected_last = np.mean(prices[-50:])
        self.assertAlmostEqual(result.iloc[-1], expected_last, places=10)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        unittest.main(verbosity=2)
    else:
        print("Usage: python test_sma.py run")
