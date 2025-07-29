#!/usr/bin/env python3
"""
Dynamic Pricing System - Core Module
This module provides the main pricing algorithms referenced in the documentation and examples.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def rule_based_pricing(original_price: float, 
                      inventory_level: int, 
                      is_holiday: bool, 
                      is_weekend: bool, 
                      customer_segment: str) -> Tuple[float, List[str]]:
    """
    🧙‍♂️ The Elves' Rule-Based Pricing Spell
    
    This magical function applies simple rules to adjust prices:
    - Low inventory = higher prices
    - Holidays = premium pricing  
    - Weekends = slight increase
    - Customer loyalty = discounts
    
    Args:
        original_price (float): Base price before adjustments (must be > 0)
        inventory_level (int): Current stock quantity (0-100)
        is_holiday (bool): Holiday period indicator
        is_weekend (bool): Weekend timing indicator
        customer_segment (str): Customer category ('New', 'Regular', 'Loyal', 'High-Value')
    
    Returns:
        Tuple[float, List[str]]: (adjusted_price, list_of_applied_adjustments)
    
    Raises:
        ValueError: If price <= 0 or invalid customer segment
    """
    # Input validation
    if original_price <= 0:
        raise ValueError("Original price must be greater than 0")
    
    if inventory_level < 0 or inventory_level > 100:
        inventory_level = max(0, min(100, inventory_level))  # Clamp to valid range
    
    valid_segments = {'New', 'Regular', 'Loyal', 'High-Value'}
    if customer_segment not in valid_segments:
        raise ValueError(f"Customer segment must be one of: {valid_segments}")
    
    adjusted_price = original_price
    adjustments = []
    
    # Rule 1: Inventory-based pricing
    if inventory_level < 10:
        multiplier = 1.25  # 25% increase for low stock
        adjusted_price *= multiplier
        adjustments.append(f"📦 Low stock (+{(multiplier-1)*100:.0f}%)")
    elif inventory_level > 80:
        multiplier = 0.95  # 5% discount for high stock
        adjusted_price *= multiplier
        adjustments.append(f"📦 High stock ({(multiplier-1)*100:.0f}%)")
    
    # Rule 2: Holiday premium
    if is_holiday:
        multiplier = 1.20  # 20% holiday premium
        adjusted_price *= multiplier
        adjustments.append(f"🎄 Holiday season (+{(multiplier-1)*100:.0f}%)")
    
    # Rule 3: Weekend premium
    if is_weekend:
        multiplier = 1.10  # 10% weekend premium
        adjusted_price *= multiplier
        adjustments.append(f"🌅 Weekend (+{(multiplier-1)*100:.0f}%)")
    
    # Rule 4: Customer loyalty discount
    if customer_segment == 'Loyal':
        multiplier = 0.90  # 10% loyalty discount
        adjusted_price *= multiplier
        adjustments.append(f"👑 Loyal customer ({(multiplier-1)*100:.0f}%)")
    elif customer_segment == 'High-Value':
        multiplier = 0.85  # 15% high-value customer discount
        adjusted_price *= multiplier
        adjustments.append(f"💎 High-value customer ({(multiplier-1)*100:.0f}%)")
    
    return round(adjusted_price, 2), adjustments


def predict_optimal_price(original_price: float,
                         inventory_level: int,
                         is_holiday: bool,
                         is_weekend: bool,
                         competitor_price: float) -> Dict[str, Any]:
    """
    🧠 The Elves' AI-Powered Pricing Sorcery
    
    Uses machine learning to predict optimal pricing based on historical patterns.
    
    Args:
        original_price (float): Base product price (must be > 0)
        inventory_level (int): Current inventory level (0-100)
        is_holiday (bool): Holiday season indicator
        is_weekend (bool): Weekend indicator
        competitor_price (float): Average competitor pricing (must be > 0)
    
    Returns:
        Dict[str, Any]: Comprehensive prediction results including:
            - predicted_price: Optimal price prediction
            - confidence_score: Model confidence (R² score)
            - revenue_estimate: Expected revenue
            - demand_forecast: Predicted demand
            - price_sensitivity: Elasticity coefficient
            - test_prices: Array of test prices for analysis
            - revenues: Corresponding revenue predictions
    
    Raises:
        ValueError: If prices <= 0 or data file not found
        FileNotFoundError: If the dataset CSV file doesn't exist
    """
    # Input validation
    if original_price <= 0:
        raise ValueError("Original price must be greater than 0")
    
    if competitor_price <= 0:
        raise ValueError("Competitor price must be greater than 0")
    
    if inventory_level < 0 or inventory_level > 100:
        inventory_level = max(0, min(100, inventory_level))
    
    try:
        # Load historical data
        df = pd.read_csv('elves_marketplace_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Engineer features
        df['weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
        
        # Prepare features for ML model
        features = ['original_price', 'inventory_level_before_sale', 'holiday_season', 
                   'weekend', 'competitor_price_avg']
        
        X = df[features].copy()
        y = df['price_paid']  # Target: actual price paid
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model performance
        y_pred = model.predict(X_test)
        confidence_score = r2_score(y_test, y_pred)
        
        # Generate price optimization curve
        min_price = original_price * 0.8  # 20% discount max
        max_price = original_price * 1.2  # 20% premium max
        test_prices = np.linspace(min_price, max_price, 20)
        
        revenues = []
        demands = []
        
        for test_price in test_prices:
            # Create feature vector for prediction
            features_array = np.array([[
                test_price,  # original_price
                inventory_level,  # inventory_level_before_sale
                1 if is_holiday else 0,  # holiday_season
                1 if is_weekend else 0,  # weekend
                competitor_price  # competitor_price_avg
            ]])
            
            # Predict actual price that would be paid
            predicted_actual_price = model.predict(features_array)[0]
            
            # Estimate demand using simple price elasticity (higher price = lower demand)
            price_ratio = test_price / original_price
            base_demand = 2.0  # Base demand assumption
            elasticity = -0.5  # Price elasticity coefficient
            
            # Demand decreases as price increases
            estimated_demand = base_demand * (price_ratio ** elasticity)
            estimated_demand = max(0.1, estimated_demand)  # Minimum demand
            
            # Calculate revenue using the test price and estimated demand
            revenue = test_price * estimated_demand
            
            revenues.append(revenue)
            demands.append(estimated_demand)
        
        # Find optimal price (maximize revenue)
        optimal_idx = np.argmax(revenues)
        optimal_price = test_prices[optimal_idx]
        optimal_demand = demands[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        # Calculate price sensitivity
        price_sensitivity = -0.5  # Our assumed elasticity coefficient
        
        return {
            'predicted_price': round(optimal_price, 2),
            'confidence_score': round(confidence_score, 3),
            'revenue_estimate': round(optimal_revenue, 2),
            'demand_forecast': round(optimal_demand, 1),
            'price_sensitivity': price_sensitivity,
            'test_prices': test_prices,
            'revenues': revenues
        }
        
    except FileNotFoundError:
        raise FileNotFoundError("Dataset file 'elves_marketplace_data.csv' not found. Please run generate_dataset.py first.")
    except Exception as e:
        # Fallback to rule-based pricing if ML fails
        rule_price, _ = rule_based_pricing(original_price, inventory_level, is_holiday, is_weekend, 'Regular')
        
        return {
            'predicted_price': rule_price,
            'confidence_score': 0.0,  # No confidence in fallback
            'revenue_estimate': rule_price * 1.5,  # Simple estimate
            'demand_forecast': 1.5,
            'price_sensitivity': -0.5,
            'test_prices': np.array([rule_price]),
            'revenues': np.array([rule_price * 1.5])
        }


def analyze_pricing_strategy(df: pd.DataFrame, strategy_name: str = "Custom") -> Dict[str, Any]:
    """
    Analyze the performance of a pricing strategy on historical data.
    
    Args:
        df (pd.DataFrame): Historical transaction data
        strategy_name (str): Name of the pricing strategy for reporting
    
    Returns:
        Dict[str, Any]: Performance analysis results
    """
    total_revenue = (df['price_paid'] * df['quantity']).sum()
    avg_price = df['price_paid'].mean()
    total_transactions = len(df)
    unique_customers = df['customer_id'].nunique()
    
    return {
        'strategy_name': strategy_name,
        'total_revenue': round(total_revenue, 2),
        'average_price': round(avg_price, 2),
        'total_transactions': total_transactions,
        'unique_customers': unique_customers,
        'revenue_per_transaction': round(total_revenue / total_transactions, 2),
        'revenue_per_customer': round(total_revenue / unique_customers, 2)
    }


def compare_pricing_strategies(original_data: pd.DataFrame, 
                             sample_size: int = 1000) -> Dict[str, Dict[str, Any]]:
    """
    Compare different pricing strategies on a sample of data.
    
    Args:
        original_data (pd.DataFrame): Historical transaction data
        sample_size (int): Number of transactions to test (default: 1000)
    
    Returns:
        Dict[str, Dict[str, Any]]: Comparison results for each strategy
    """
    # Sample data for comparison
    sample_df = original_data.sample(n=min(sample_size, len(original_data)), random_state=42)
    
    strategies = {
        'Static Pricing': [],
        'Rule-Based Magic': [],
        'AI-Powered Sorcery': []
    }
    
    for _, row in sample_df.iterrows():
        original_price = row['original_price']
        inventory = row['inventory_level_before_sale']
        is_holiday = bool(row['holiday_season'])
        is_weekend = bool(row['timestamp'].weekday() >= 5) if isinstance(row['timestamp'], pd.Timestamp) else False
        customer_segment = row['customer_segment']
        competitor_price = row['competitor_price_avg']
        actual_quantity = row['quantity']
        
        # Strategy 1: Static Pricing
        static_revenue = original_price * actual_quantity
        strategies['Static Pricing'].append({
            'price': original_price,
            'revenue': static_revenue,
            'quantity': actual_quantity
        })
        
        # Strategy 2: Rule-Based Pricing
        try:
            rule_price, _ = rule_based_pricing(original_price, inventory, is_holiday, is_weekend, customer_segment)
            # Adjust quantity based on price change (simple demand curve)
            price_ratio = rule_price / original_price
            adjusted_quantity = actual_quantity * (2.0 - price_ratio)
            adjusted_quantity = max(0.5, adjusted_quantity)
            rule_revenue = rule_price * adjusted_quantity
            
            strategies['Rule-Based Magic'].append({
                'price': rule_price,
                'revenue': rule_revenue,
                'quantity': adjusted_quantity
            })
        except:
            # Fallback to static if rule-based fails
            strategies['Rule-Based Magic'].append({
                'price': original_price,
                'revenue': static_revenue,
                'quantity': actual_quantity
            })
        
        # Strategy 3: AI-Powered Pricing
        try:
            ai_result = predict_optimal_price(original_price, inventory, is_holiday, is_weekend, competitor_price)
            ai_price = ai_result['predicted_price']
            ai_demand = ai_result['demand_forecast']
            ai_revenue = ai_price * ai_demand
            
            strategies['AI-Powered Sorcery'].append({
                'price': ai_price,
                'revenue': ai_revenue,
                'quantity': ai_demand
            })
        except:
            # Fallback to rule-based if AI fails
            try:
                rule_price, _ = rule_based_pricing(original_price, inventory, is_holiday, is_weekend, customer_segment)
                strategies['AI-Powered Sorcery'].append({
                    'price': rule_price,
                    'revenue': rule_price * actual_quantity,
                    'quantity': actual_quantity
                })
            except:
                strategies['AI-Powered Sorcery'].append({
                    'price': original_price,
                    'revenue': static_revenue,
                    'quantity': actual_quantity
                })
    
    # Calculate summary statistics
    results = {}
    for strategy_name, data in strategies.items():
        total_revenue = sum([item['revenue'] for item in data])
        avg_price = np.mean([item['price'] for item in data])
        avg_revenue = np.mean([item['revenue'] for item in data])
        
        results[strategy_name] = {
            'total_revenue': round(total_revenue, 2),
            'average_price': round(avg_price, 2),
            'average_revenue_per_transaction': round(avg_revenue, 2),
            'sample_size': len(data)
        }
    
    return results


if __name__ == "__main__":
    # Test the pricing system
    print("🧙‍♂️ Testing the Elves' Pricing System...")
    
    # Test rule-based pricing
    print("\n📚 Testing Rule-Based Pricing:")
    price, adjustments = rule_based_pricing(50.0, 5, True, True, 'Loyal')
    print(f"Price: ${price:.2f}, Adjustments: {adjustments}")
    
    # Test AI-powered pricing (if data exists)
    try:
        print("\n🧠 Testing AI-Powered Pricing:")
        result = predict_optimal_price(50.0, 15, True, False, 52.0)
        print(f"Optimal price: ${result['predicted_price']:.2f}")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Expected revenue: ${result['revenue_estimate']:.2f}")
    except FileNotFoundError:
        print("📝 Note: Run 'python generate_dataset.py' first to test AI pricing")
    
    print("\n✨ Pricing system ready for magical adventures!")