#!/usr/bin/env python3
"""
Simple integration test
"""

import pandas as pd
import numpy as np
from pricing_system import rule_based_pricing, predict_optimal_price

def test_integration():
    print('📊 Loading dataset...')
    df = pd.read_csv('elves_marketplace_data.csv')
    print(f'✅ Loaded {len(df)} transactions')

    # Test the pricing functions work with the data
    print('🧪 Testing pricing functions...')
    sample_row = df.iloc[0]
    price, adj = rule_based_pricing(
        sample_row['original_price'], 
        sample_row['inventory_level_before_sale'],
        bool(sample_row['holiday_season']),
        False,
        sample_row['customer_segment']
    )
    print(f'✅ Rule-based pricing: ${price:.2f}')

    result = predict_optimal_price(
        sample_row['original_price'],
        sample_row['inventory_level_before_sale'], 
        bool(sample_row['holiday_season']),
        False,
        sample_row['competitor_price_avg']
    )
    print(f'✅ AI pricing: ${result["predicted_price"]:.2f}')
    print('🎉 All systems working!')

if __name__ == "__main__":
    test_integration()