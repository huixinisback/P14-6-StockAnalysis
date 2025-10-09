#!/usr/bin/env python3
"""
Test Buy/Sell Signal Generation (SMA Crossover).
Comprehensive test suite with static calculations and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import compute_buy_sell_signals, sma_sliding_window


class TestBuySellSignals(unittest.TestCase):
    """Test SMA Crossover Signals - 12 test cases."""
    
    def test_01_static_calculation_single_buy(self):
        """Test 1: Static calculation - single buy signal."""
        # Clear crossover: price crosses FROM BELOW to ABOVE SMA
        prices = np.array([10, 10, 11, 13, 12])
        sma = np.array([np.nan, np.nan, 12.0, 12.0, 12.0])
        
        result = compute_buy_sell_signals(prices, sma)
        
        # At index 3: price=13 > sma=12, prev_price=11 < prev_sma=12 -> BUY
        self.assertEqual(len(result['buy_indices']), 1)
        self.assertEqual(result['buy_indices'][0], 3)
        self.assertAlmostEqual(result['buy_prices'][0], 13.0, places=6)
    
    def test_02_static_calculation_single_sell(self):
        """Test 2: Static calculation - single sell signal."""
        # Clear crossover: price crosses FROM ABOVE to BELOW SMA
        prices = np.array([13, 13, 12, 9, 10])
        sma = np.array([np.nan, np.nan, 10.0, 10.0, 10.0])
        
        result = compute_buy_sell_signals(prices, sma)
        
        # At index 3: price=9 < sma=10, prev_price=12 > prev_sma=10 -> SELL
        self.assertEqual(len(result['buy_indices']), 0)
        self.assertEqual(len(result['sell_indices']), 1)
        self.assertEqual(result['sell_indices'][0], 3)
        self.assertAlmostEqual(result['sell_prices'][0], 9.0, places=6)
    
    def test_03_static_calculation_multiple_signals(self):
        """Test 3: Static calculation - multiple buy/sell signals."""
        # Create crossover pattern
        prices = np.array([10, 11, 12, 11, 10, 11, 12, 13])
        # Calculate actual SMA for window=3
        sma = sma_sliding_window(prices, 3).values
        
        result = compute_buy_sell_signals(prices, sma)
        
        # Should detect crossovers
        total_signals = len(result['buy_indices']) + len(result['sell_indices'])
        self.assertGreater(total_signals, 0)
    
    def test_04_edge_case_empty_arrays(self):
        """Test 4: Edge case - empty arrays."""
        result = compute_buy_sell_signals(np.array([]), np.array([]))
        
        self.assertEqual(result['buy_indices'], [])
        self.assertEqual(result['sell_indices'], [])
        self.assertEqual(result['buy_prices'], [])
        self.assertEqual(result['sell_prices'], [])
    
    def test_05_edge_case_single_element(self):
        """Test 5: Edge case - single element."""
        result = compute_buy_sell_signals(np.array([100]), np.array([100]))
        
        self.assertEqual(result['buy_indices'], [])
        self.assertEqual(result['sell_indices'], [])
    
    def test_06_edge_case_different_lengths(self):
        """Test 6: Edge case - different array lengths."""
        result = compute_buy_sell_signals(
            np.array([100, 102, 104]),
            np.array([100, 102])
        )
        
        # Should return empty results
        self.assertEqual(result['buy_indices'], [])
        self.assertEqual(result['sell_indices'], [])
    
    def test_07_edge_case_all_nan_sma(self):
        """Test 7: Edge case - all NaN SMA."""
        prices = np.array([100, 102, 104, 106])
        sma = np.array([np.nan, np.nan, np.nan, np.nan])
        
        result = compute_buy_sell_signals(prices, sma)
        
        # No signals when SMA is all NaN
        self.assertEqual(result['buy_indices'], [])
        self.assertEqual(result['sell_indices'], [])
    
    def test_08_edge_case_price_always_above(self):
        """Test 8: Edge case - price always above SMA (no crossover)."""
        prices = np.array([100, 102, 104, 106, 108])
        sma = np.array([np.nan, np.nan, 95, 96, 97])
        
        result = compute_buy_sell_signals(prices, sma)
        
        # No crossover = no signals
        self.assertEqual(result['buy_indices'], [])
        self.assertEqual(result['sell_indices'], [])
    
    def test_09_edge_case_price_always_below(self):
        """Test 9: Edge case - price always below SMA (no crossover)."""
        prices = np.array([100, 102, 104, 106, 108])
        sma = np.array([np.nan, np.nan, 110, 112, 114])
        
        result = compute_buy_sell_signals(prices, sma)
        
        # No crossover = no signals
        self.assertEqual(result['buy_indices'], [])
        self.assertEqual(result['sell_indices'], [])
    
    def test_10_buy_then_sell_sequence(self):
        """Test 10: Test buy followed by sell."""
        # Price crosses above then below SMA
        # Need extra point so crossover happens AFTER first valid SMA
        prices = np.array([9, 9, 9, 13, 13, 9, 9])
        sma = np.array([np.nan, np.nan, 11.0, 11.0, 11.0, 11.0, 11.0])
        
        result = compute_buy_sell_signals(prices, sma)
        
        # start_idx=2, loop from i=3 onwards
        # At i=3: price=13 > sma=11, prev_price=9 < prev_sma=11 -> BUY
        # At i=5: price=9 < sma=11, prev_price=13 > prev_sma=11 -> SELL
        self.assertGreater(len(result['buy_indices']), 0)
        self.assertGreater(len(result['sell_indices']), 0)
        
        # Buy should come before sell
        first_buy = result['buy_indices'][0]
        first_sell = result['sell_indices'][0]
        self.assertLess(first_buy, first_sell)
    
    def test_11_signal_indices_match_prices(self):
        """Test 11: Verify indices match prices correctly."""
        prices = np.array([100, 105, 110, 105, 100, 105])
        sma = sma_sliding_window(prices, 3).values
        
        result = compute_buy_sell_signals(prices, sma)
        
        # Verify buy_prices match prices at buy_indices
        for idx, price in zip(result['buy_indices'], result['buy_prices']):
            self.assertAlmostEqual(price, prices[idx], places=10)
        
        # Verify sell_prices match prices at sell_indices
        for idx, price in zip(result['sell_indices'], result['sell_prices']):
            self.assertAlmostEqual(price, prices[idx], places=10)
    
    def test_12_crossover_detection(self):
        """Test 12: Verify crossover detection logic."""
        # Explicit crossover scenario
        # Price: below, below, ABOVE (buy signal), above, BELOW (sell signal)
        prices = np.array([9, 9, 11, 11, 9])
        sma = np.array([np.nan, np.nan, 10.0, 10.0, 10.0])
        
        result = compute_buy_sell_signals(prices, sma)
        
        # At index 2: price=11 > sma=10, prev_price=9 < prev_sma=NaN -> skip (prev_sma is NaN)
        # Actually start_idx=2, so loop starts at i=3
        # At index 3: price=11 > sma=10, prev_price=11 > prev_sma=10 -> no signal (already above)
        # At index 4: price=9 < sma=10, prev_price=11 > prev_sma=10 -> SELL
        
        # Adjust: need crossover TO happen
        prices2 = np.array([9, 9, 9, 11, 9])
        sma2 = np.array([np.nan, np.nan, 10.0, 10.0, 10.0])
        
        result2 = compute_buy_sell_signals(prices2, sma2)
        
        # At i=3: price=11 > sma=10, prev_price=9 < prev_sma=10 -> BUY
        # At i=4: price=9 < sma=10, prev_price=11 > prev_sma=10 -> SELL
        self.assertIn(3, result2['buy_indices'])
        self.assertIn(4, result2['sell_indices'])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        unittest.main(verbosity=2)
    else:
        print("Usage: python test_signals.py run")

