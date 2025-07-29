#!/usr/bin/env python3
"""
Comprehensive test suite for the Dynamic Pricing System
This script runs all major components to ensure everything works correctly.
"""

import unittest
import pandas as pd
import numpy as np
from pricing_system import rule_based_pricing, predict_optimal_price, compare_pricing_strategies
from pricing_api import PricingAPI
import os

class TestDynamicPricingSystem(unittest.TestCase):
    """Test suite for the dynamic pricing system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Ensure dataset exists
        if not os.path.exists('elves_marketplace_data.csv'):
            print("📊 Generating test dataset...")
            import subprocess
            subprocess.run(['python', 'generate_dataset.py'], check=True)
        
        # Load data for tests
        cls.df = pd.read_csv('elves_marketplace_data.csv')
        cls.api = PricingAPI()
    
    def test_rule_based_pricing_basic(self):
        """Test basic rule-based pricing functionality"""
        price, adjustments = rule_based_pricing(100.0, 50, False, False, 'Regular')
        self.assertEqual(price, 100.0)  # No adjustments should be applied
        self.assertEqual(len(adjustments), 0)
    
    def test_rule_based_pricing_low_inventory(self):
        """Test rule-based pricing with low inventory"""
        price, adjustments = rule_based_pricing(100.0, 5, False, False, 'Regular')
        self.assertEqual(price, 125.0)  # 25% increase for low inventory
        self.assertIn('Low stock', adjustments[0])
    
    def test_rule_based_pricing_holiday(self):
        """Test rule-based pricing during holidays"""
        price, adjustments = rule_based_pricing(100.0, 50, True, False, 'Regular')
        self.assertEqual(price, 120.0)  # 20% holiday premium
        self.assertIn('Holiday', adjustments[0])
    
    def test_rule_based_pricing_loyal_customer(self):
        """Test rule-based pricing for loyal customers"""
        price, adjustments = rule_based_pricing(100.0, 50, False, False, 'Loyal')
        self.assertEqual(price, 90.0)  # 10% loyalty discount
        self.assertIn('Loyal', adjustments[0])
    
    def test_rule_based_pricing_combined(self):
        """Test rule-based pricing with multiple factors"""
        price, adjustments = rule_based_pricing(100.0, 5, True, True, 'Loyal')
        # 25% * 20% * 10% * -10% = 1.25 * 1.20 * 1.10 * 0.90 = 148.5
        self.assertAlmostEqual(price, 148.5, places=2)
        self.assertEqual(len(adjustments), 4)  # All adjustments should be applied
    
    def test_rule_based_pricing_invalid_inputs(self):
        """Test rule-based pricing with invalid inputs"""
        with self.assertRaises(ValueError):
            rule_based_pricing(0, 50, False, False, 'Regular')  # Price = 0
        
        with self.assertRaises(ValueError):
            rule_based_pricing(100.0, 50, False, False, 'Invalid')  # Invalid segment
    
    def test_ai_pricing_basic(self):
        """Test AI-powered pricing functionality"""
        result = predict_optimal_price(100.0, 50, False, False, 105.0)
        
        self.assertIn('predicted_price', result)
        self.assertIn('confidence_score', result)
        self.assertIn('revenue_estimate', result)
        self.assertIn('demand_forecast', result)
        
        # Predicted price should be reasonable
        self.assertGreater(result['predicted_price'], 50.0)
        self.assertLess(result['predicted_price'], 200.0)
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(result['confidence_score'], 0.0)
        self.assertLessEqual(result['confidence_score'], 1.0)
    
    def test_ai_pricing_invalid_inputs(self):
        """Test AI pricing with invalid inputs"""
        with self.assertRaises(ValueError):
            predict_optimal_price(0, 50, False, False, 105.0)  # Price = 0
        
        with self.assertRaises(ValueError):
            predict_optimal_price(100.0, 50, False, False, 0)  # Competitor price = 0
    
    def test_pricing_strategy_comparison(self):
        """Test pricing strategy comparison functionality"""
        result = compare_pricing_strategies(self.df, sample_size=100)
        
        # Should have results for all three strategies
        expected_strategies = ['Static Pricing', 'Rule-Based Magic', 'AI-Powered Sorcery']
        for strategy in expected_strategies:
            self.assertIn(strategy, result)
            self.assertIn('total_revenue', result[strategy])
            self.assertIn('average_price', result[strategy])
    
    def test_api_system_info(self):
        """Test API system information"""
        info = self.api.get_system_info()
        self.assertIn('system_name', info)
        self.assertIn('version', info)
        self.assertIn('available_endpoints', info)
    
    def test_api_rule_based_pricing(self):
        """Test API rule-based pricing"""
        result = self.api.calculate_rule_based_price(100.0, 50, False, False, 'Regular')
        
        self.assertTrue(result['success'])
        self.assertEqual(result['method'], 'rule-based')
        self.assertIn('recommended_price', result['result'])
        self.assertEqual(result['result']['recommended_price'], 100.0)
    
    def test_api_ai_pricing(self):
        """Test API AI pricing"""
        result = self.api.calculate_ai_optimized_price(100.0, 50, False, False, 105.0)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['method'], 'ai-optimized')
        self.assertIn('recommended_price', result['result'])
        self.assertIn('confidence', result['result'])
    
    def test_api_comparison(self):
        """Test API pricing method comparison"""
        result = self.api.compare_pricing_methods(100.0, 50, False, False, 'Regular', 105.0)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['method'], 'comparison')
        self.assertIn('results', result)
        self.assertIn('recommendation', result)
        
        # Should have all three pricing methods
        results = result['results']
        self.assertIn('static_pricing', results)
        self.assertIn('rule_based', results)
        self.assertIn('ai_optimized', results)
    
    def test_data_integrity(self):
        """Test that the generated dataset has correct structure"""
        expected_columns = [
            'transaction_id', 'product_id', 'product_name', 'category',
            'original_price', 'price_paid', 'quantity', 'timestamp',
            'customer_id', 'customer_segment', 'inventory_level_before_sale',
            'competitor_price_avg', 'holiday_season'
        ]
        
        for col in expected_columns:
            self.assertIn(col, self.df.columns)
        
        # Check data types and ranges
        self.assertTrue(self.df['original_price'].min() > 0)
        self.assertTrue(self.df['price_paid'].min() > 0)
        self.assertTrue(self.df['quantity'].min() > 0)
        self.assertTrue(self.df['inventory_level_before_sale'].min() >= 0)
        self.assertTrue(self.df['inventory_level_before_sale'].max() <= 100)


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("🧪 Running Comprehensive Test Suite for Dynamic Pricing System...")
    print("=" * 70)
    
    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDynamicPricingSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"  ✅ Tests Run: {result.testsRun}")
    print(f"  ❌ Failures: {len(result.failures)}")
    print(f"  ⚠️ Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Test Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️ Test Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n🎉 All tests passed! The Dynamic Pricing System is ready for production! ✨")
    else:
        print(f"\n⚠️ Some tests failed. Please review the issues above.")
    
    return success


if __name__ == "__main__":
    run_comprehensive_tests()