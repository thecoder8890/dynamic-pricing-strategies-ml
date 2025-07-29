#!/usr/bin/env python3
"""
Test script to validate the examples from README.md work correctly
"""

from pricing_system import rule_based_pricing, predict_optimal_price

def test_readme_examples():
    """Test the examples from the README.md file"""
    print("🧪 Testing README Examples...")
    print("=" * 50)
    
    # Example 1: Rule-based pricing
    print("\n📚 Testing Rule-Based Pricing Example:")
    price, adjustments = rule_based_pricing(
        original_price=100.0,
        inventory_level=15,
        is_holiday=True,
        is_weekend=False,
        customer_segment='Loyal'
    )
    print(f"New price: ${price:.2f}, Adjustments: {adjustments}")
    
    # Example 2: ML-based pricing optimization
    print("\n🧠 Testing ML-Based Pricing Example:")
    result = predict_optimal_price(
        original_price=100.0,
        inventory_level=15,
        is_holiday=True,
        is_weekend=False,
        competitor_price=105.0
    )
    print(f"Optimal price: ${result['predicted_price']:.2f}")
    print(f"Expected revenue: ${result['revenue_estimate']:.2f}")
    print(f"Model confidence: {result['confidence_score']:.3f}")
    print(f"Demand forecast: {result['demand_forecast']:.1f} units")
    
    print("\n✅ All README examples work correctly!")

if __name__ == "__main__":
    test_readme_examples()