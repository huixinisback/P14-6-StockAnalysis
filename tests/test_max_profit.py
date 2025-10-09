#!/usr/bin/env python3
"""
Test Max Profit (Multiple Transactions) calculation.
Comprehensive test suite with static calculations and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import max_profit_multiple_transactions


class TestMaxProfit(unittest.TestCase):
    """Test Max Profit - 12 test cases."""
    
    def test_01_static_calculation_leetcode_example(self):
        """Test 1: Static calculation - LeetCode example."""
        prices = pd.Series([7, 1, 5, 3, 6, 4])
        result = max_profit_multiple_transactions(prices)
        
        # Manual calculation:
        # Buy at 1, sell at 5: +4
        # Buy at 3, sell at 6: +3
        # Total: 7
        # Alternative: sum all positive diffs = (5-1) + (6-3) = 4 + 3 = 7
        expected = 7.0
        
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_02_static_calculation_all_increasing(self):
        """Test 2: Static calculation - all increasing prices."""
        prices = pd.Series([1, 2, 3, 4, 5])
        result = max_profit_multiple_transactions(prices)
        
        # Manual calculation:
        # Buy at 1, sell at 5: profit = 5-1 = 4
        # Or: (2-1) + (3-2) + (4-3) + (5-4) = 1+1+1+1 = 4
        expected = 4.0
        
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_03_static_calculation_all_decreasing(self):
        """Test 3: Static calculation - all decreasing prices."""
        prices = pd.Series([5, 4, 3, 2, 1])
        result = max_profit_multiple_transactions(prices)
        
        # Manual calculation:
        # No profitable transactions possible
        expected = 0.0
        
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_04_static_calculation_multiple_transactions(self):
        """Test 4: Static calculation - multiple profitable transactions."""
        prices = pd.Series([10, 20, 15, 25, 20, 30])
        result = max_profit_multiple_transactions(prices)
        
        # Manual calculation (sum all positive diffs):
        # (20-10) + (25-15) + (30-20) = 10 + 10 + 10 = 30
        expected = 30.0
        
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_05_edge_case_empty(self):
        """Test 5: Edge case - empty series."""
        prices = pd.Series([])
        result = max_profit_multiple_transactions(prices)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_06_edge_case_single_price(self):
        """Test 6: Edge case - single price."""
        prices = pd.Series([100])
        result = max_profit_multiple_transactions(prices)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_07_edge_case_two_prices_profit(self):
        """Test 7: Edge case - two prices with profit."""
        prices = pd.Series([100, 110])
        result = max_profit_multiple_transactions(prices)
        self.assertAlmostEqual(result, 10.0, places=10)
    
    def test_08_edge_case_two_prices_loss(self):
        """Test 8: Edge case - two prices with loss."""
        prices = pd.Series([110, 100])
        result = max_profit_multiple_transactions(prices)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_09_edge_case_flat_prices(self):
        """Test 9: Edge case - all same prices."""
        prices = pd.Series([100, 100, 100, 100])
        result = max_profit_multiple_transactions(prices)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_10_real_world_pattern(self):
        """Test 10: Real-world price pattern."""
        # Typical stock movement: up, down, up, down
        prices = pd.Series([100, 105, 103, 108, 106, 112, 110, 115])
        result = max_profit_multiple_transactions(prices)
        
        # Sum all positive changes:
        # (105-100) + (108-103) + (112-106) + (115-110) = 5+5+6+5 = 21
        expected = 21.0
        
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_11_greedy_algorithm_verification(self):
        """Test 11: Verify greedy algorithm captures all ups."""
        prices = pd.Series([1, 3, 2, 4, 3, 5])
        result = max_profit_multiple_transactions(prices)
        
        # Optimal: buy at 1, sell at 3 (+2), buy at 2, sell at 4 (+2), buy at 3, sell at 5 (+2)
        # Greedy sum: (3-1) + (4-2) + (5-3) = 2+2+2 = 6
        expected = 6.0
        
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_12_large_dataset(self):
        """Test 12: Test with large dataset."""
        # Generate trending data
        np.random.seed(42)
        trend = np.linspace(100, 200, 100)
        noise = np.random.randn(100) * 5
        prices = pd.Series(trend + noise)
        
        result = max_profit_multiple_transactions(prices)
        
        # Should be positive for uptrending data
        self.assertGreater(result, 0)
        # Total profit can exceed simple range due to capturing noise ups
        # Just verify it's reasonable (< 10x the range)
        self.assertLess(result, 1000)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        unittest.main(verbosity=2)
    else:
        print("Usage: python test_max_profit.py run")

