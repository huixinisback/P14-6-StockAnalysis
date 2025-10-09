#!/usr/bin/env python3
"""
Test Bollinger Bands indicator.
Comprehensive test suite with static calculations, pandas comparison, and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import bollinger_bands, sma_sliding_window


class TestBollingerBands(unittest.TestCase):
    """Test Bollinger Bands - 12 test cases."""
    
    def test_01_static_calculation_simple(self):
        """Test 1: Static calculation with simple data."""
        # Use period=3 for easy manual calculation
        prices = [10, 12, 14, 16, 18, 20]
        middle, upper, lower = bollinger_bands(prices, period=3, std_dev=2.0)
        
        # Manual calculation at index 2:
        # SMA(3) = (10+12+14)/3 = 12.0
        # std = sqrt(((10-12)^2 + (12-12)^2 + (14-12)^2)/3) = sqrt((4+0+4)/3) = sqrt(2.667) = 1.633
        # Upper = 12 + 2*1.633 = 15.266
        # Lower = 12 - 2*1.633 = 8.734
        
        self.assertAlmostEqual(middle.iloc[2], 12.0, places=6)
        self.assertAlmostEqual(upper.iloc[2], 15.266, places=2)
        self.assertAlmostEqual(lower.iloc[2], 8.734, places=2)
    
    def test_02_static_calculation_known_values(self):
        """Test 2: Static calculation with controlled values."""
        # Prices with known variance
        prices = [100, 100, 100, 110, 110]
        middle, upper, lower = bollinger_bands(prices, period=5, std_dev=1.0)
        
        # At index 4:
        # SMA = (100+100+100+110+110)/5 = 104.0
        # Variance = mean of squared deviations
        # Values: [100, 100, 100, 110, 110], mean=104
        # Deviations: [-4, -4, -4, 6, 6]
        # Squared: [16, 16, 16, 36, 36] -> mean = 120/5 = 24
        # Std = sqrt(24) = 4.899
        # Upper = 104 + 1*4.899 = 108.899
        # Lower = 104 - 1*4.899 = 99.101
        
        self.assertAlmostEqual(middle.iloc[4], 104.0, places=6)
        self.assertAlmostEqual(upper.iloc[4], 108.899, places=2)
        self.assertAlmostEqual(lower.iloc[4], 99.101, places=2)
    
    def test_03_middle_band_equals_sma(self):
        """Test 3: Middle band should equal SMA."""
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100
        middle, upper, lower = bollinger_bands(prices, period=20, std_dev=2.0)
        sma = sma_sliding_window(prices, 20)
        
        # Middle band should exactly equal SMA
        valid_mask = ~middle.isna()
        np.testing.assert_array_almost_equal(
            middle[valid_mask].values, 
            sma[valid_mask].values, 
            decimal=10
        )
    
    def test_04_pandas_comparison(self):
        """Test 4: Compare with pandas rolling std implementation."""
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100
        df = pd.DataFrame({'Close': prices})
        
        our_middle, our_upper, our_lower = bollinger_bands(prices, period=20, std_dev=2.0)
        
        # Pandas implementation (use ddof=0 for population std)
        pandas_middle = df['Close'].rolling(window=20, min_periods=20).mean()
        pandas_std = df['Close'].rolling(window=20, min_periods=20).std(ddof=0)
        pandas_upper = pandas_middle + (2.0 * pandas_std)
        pandas_lower = pandas_middle - (2.0 * pandas_std)
        
        # Compare valid values
        valid_mask = ~pandas_middle.isna()
        if valid_mask.any():
            np.testing.assert_array_almost_equal(
                our_middle[valid_mask].values,
                pandas_middle[valid_mask].values,
                decimal=8
            )
            np.testing.assert_array_almost_equal(
                our_upper[valid_mask].values,
                pandas_upper[valid_mask].values,
                decimal=6
            )
            np.testing.assert_array_almost_equal(
                our_lower[valid_mask].values,
                pandas_lower[valid_mask].values,
                decimal=6
            )
    
    def test_05_edge_case_insufficient_data(self):
        """Test 5: Edge case - insufficient data."""
        middle, upper, lower = bollinger_bands([100, 102], period=20)
        self.assertTrue(middle.isna().all())
        self.assertTrue(upper.isna().all())
        self.assertTrue(lower.isna().all())
    
    def test_06_edge_case_exact_period_data(self):
        """Test 6: Edge case - exactly period data points."""
        prices = list(range(100, 120))  # 20 points
        middle, upper, lower = bollinger_bands(prices, period=20, std_dev=2.0)
        
        # Should have exactly 1 valid value at index 19
        self.assertEqual(middle.notna().sum(), 1)
        self.assertFalse(middle.isna().iloc[-1])
    
    def test_07_edge_case_empty(self):
        """Test 7: Edge case - empty array."""
        middle, upper, lower = bollinger_bands([], period=20)
        self.assertEqual(len(middle), 0)
        self.assertEqual(len(upper), 0)
        self.assertEqual(len(lower), 0)
    
    def test_08_band_relationships(self):
        """Test 8: Verify upper > middle > lower relationship."""
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100
        middle, upper, lower = bollinger_bands(prices, period=20, std_dev=2.0)
        
        valid_mask = ~middle.isna()
        if valid_mask.any():
            # Upper should always be >= middle
            self.assertTrue(all(upper[valid_mask] >= middle[valid_mask]))
            # Lower should always be <= middle
            self.assertTrue(all(lower[valid_mask] <= middle[valid_mask]))
            # Upper should be > lower (unless std=0)
            self.assertTrue(all(upper[valid_mask] >= lower[valid_mask]))
    
    def test_09_std_dev_multiplier(self):
        """Test 9: Test different std_dev multipliers."""
        prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118] * 3  # 30 points
        
        middle1, upper1, lower1 = bollinger_bands(prices, period=20, std_dev=1.0)
        middle2, upper2, lower2 = bollinger_bands(prices, period=20, std_dev=2.0)
        middle3, upper3, lower3 = bollinger_bands(prices, period=20, std_dev=3.0)
        
        # Middle should be the same regardless of std_dev
        valid_mask = ~middle1.isna()
        if valid_mask.any():
            np.testing.assert_array_almost_equal(
                middle1[valid_mask].values,
                middle2[valid_mask].values,
                decimal=10
            )
            np.testing.assert_array_almost_equal(
                middle2[valid_mask].values,
                middle3[valid_mask].values,
                decimal=10
            )
        
        # Bands should widen with larger std_dev
        valid_idx = middle1.last_valid_index()
        if valid_idx is not None:
            band_width1 = upper1.iloc[valid_idx] - lower1.iloc[valid_idx]
            band_width2 = upper2.iloc[valid_idx] - lower2.iloc[valid_idx]
            band_width3 = upper3.iloc[valid_idx] - lower3.iloc[valid_idx]
            
            self.assertLess(band_width1, band_width2)
            self.assertLess(band_width2, band_width3)
    
    def test_10_pandas_series_input(self):
        """Test 10: Test with pandas Series input."""
        prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118] * 3)
        middle, upper, lower = bollinger_bands(prices, period=20, std_dev=2.0)
        
        # Should preserve datetime index if present
        self.assertEqual(len(middle), len(prices))
        self.assertGreater(middle.notna().sum(), 0)
    
    def test_11_numpy_array_input(self):
        """Test 11: Test with numpy array input."""
        prices = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118] * 3)
        middle, upper, lower = bollinger_bands(prices, period=20, std_dev=2.0)
        
        self.assertEqual(len(middle), len(prices))
        self.assertGreater(middle.notna().sum(), 0)
    
    def test_12_sliding_window_variance(self):
        """Test 12: Verify sliding window variance calculation."""
        # Test that variance is calculated efficiently using sliding window
        prices = list(range(100, 200))  # 100 points
        middle, upper, lower = bollinger_bands(prices, period=20, std_dev=2.0)
        
        # Should have 100 - 19 = 81 valid values
        self.assertEqual(middle.notna().sum(), 81)
        
        # Verify band width at a specific index (manual calculation)
        idx = 50
        window = prices[idx-19:idx+1]  # 20 values
        expected_mean = np.mean(window)
        expected_std = np.std(window, ddof=0)
        expected_upper = expected_mean + 2.0 * expected_std
        expected_lower = expected_mean - 2.0 * expected_std
        
        self.assertAlmostEqual(middle.iloc[idx], expected_mean, places=6)
        self.assertAlmostEqual(upper.iloc[idx], expected_upper, places=6)
        self.assertAlmostEqual(lower.iloc[idx], expected_lower, places=6)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        unittest.main(verbosity=2)
    else:
        print("Usage: python test_bollinger.py run")
