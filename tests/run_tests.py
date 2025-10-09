#!/usr/bin/env python3
"""
Simple test runner with command line arguments.
Usage: python run_tests.py [indicator]
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show_usage():
    """Show usage information."""
    print("=" * 50)
    print("TECHNICAL INDICATORS TEST RUNNER")
    print("=" * 50)
    print()
    print("Usage: python run_tests.py [indicator]")
    print()
    print("Available indicators:")
    print("  sma        - Simple Moving Average")
    print("  daily      - Daily Simple Returns")
    print("  ema        - Exponential Moving Average")
    print("  rsi        - Relative Strength Index")
    print("  bollinger  - Bollinger Bands")
    print("  signals    - Buy/Sell Signals")
    print("  profit     - Max Profit Calculation")
    print("  all        - All indicators")
    print()
    print("Examples:")
    print("  python run_tests.py sma")
    print("  python run_tests.py all")
    print()


def run_tests():
    """Run tests based on command line arguments."""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    indicator = sys.argv[1].lower()
    
    # Map indicators to test modules
    test_modules = {
        'sma': 'test_sma',
        'daily': 'test_daily_returns',
        'ema': 'test_ema',
        'rsi': 'test_rsi',
        'bollinger': 'test_bollinger',
        'signals': 'test_signals',
        'profit': 'test_max_profit'
    }
    
    if indicator not in test_modules and indicator != 'all':
        print(f"Error: Unknown indicator '{indicator}'")
        show_usage()
        return
    
    # Create test suite
    suite = unittest.TestSuite()
    
    if indicator == 'all':
        # Import and run all test modules
        for module_name in test_modules.values():
            try:
                module = __import__(module_name)
                suite.addTest(unittest.TestLoader().loadTestsFromModule(module))
            except ImportError as e:
                print(f"Error importing {module_name}: {e}")
    else:
        # Import and run specific test module
        module_name = test_modules[indicator]
        try:
            module = __import__(module_name)
            suite.addTest(unittest.TestLoader().loadTestsFromModule(module))
        except ImportError as e:
            print(f"Error importing {module_name}: {e}")
            return
    
    # Run tests
    print(f"Running {indicator.upper()} tests")
    print("=" * 50)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success: {'✅ PASSED' if result.wasSuccessful() else '❌ FAILED'}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    print(f"{'='*50}")


if __name__ == '__main__':
    run_tests()