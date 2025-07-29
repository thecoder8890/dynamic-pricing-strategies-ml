#!/usr/bin/env python3
"""
Dynamic Pricing Demo - Showcase the magical pricing strategies
This script demonstrates the different pricing approaches with visual comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pricing_system import rule_based_pricing, predict_optimal_price, compare_pricing_strategies
import warnings
warnings.filterwarnings('ignore')

def run_pricing_demo():
    """Run a comprehensive demo of the pricing system"""
    print("🧝‍♂️ Welcome to the Dynamic Pricing Magic Demo!")
    print("=" * 60)
    
    # Load the dataset
    print("\n📊 Loading Elves' Marketplace data...")
    try:
        df = pd.read_csv('elves_marketplace_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ Loaded {len(df):,} transactions from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    except FileNotFoundError:
        print("❌ Dataset not found! Please run 'python generate_dataset.py' first.")
        return
    
    # Demo 1: Rule-based pricing scenarios
    print("\n🧪 Demo 1: Rule-Based Pricing Scenarios")
    print("-" * 40)
    
    scenarios = [
        {"name": "🎄 Holiday Rush", "price": 75.0, "inventory": 8, "holiday": True, "weekend": True, "segment": "New"},
        {"name": "📦 Clearance Sale", "price": 50.0, "inventory": 95, "holiday": False, "weekend": False, "segment": "Regular"}, 
        {"name": "👑 VIP Customer", "price": 100.0, "inventory": 25, "holiday": False, "weekend": False, "segment": "Loyal"},
        {"name": "🏪 Regular Day", "price": 60.0, "inventory": 45, "holiday": False, "weekend": False, "segment": "Regular"}
    ]
    
    for scenario in scenarios:
        price, adjustments = rule_based_pricing(
            scenario["price"], scenario["inventory"], scenario["holiday"], 
            scenario["weekend"], scenario["segment"]
        )
        change_pct = ((price / scenario["price"]) - 1) * 100
        print(f"\n{scenario['name']}:")
        print(f"  Original: ${scenario['price']:.2f} → Final: ${price:.2f} ({change_pct:+.1f}%)")
        print(f"  Applied: {', '.join(adjustments) if adjustments else 'No adjustments'}")
    
    # Demo 2: AI vs Rule-based comparison
    print("\n\n🤖 Demo 2: AI vs Rule-Based Pricing")
    print("-" * 40)
    
    test_products = [
        {"name": "Enchanted Potion", "price": 45.0, "inventory": 12, "competitor": 48.0},
        {"name": "Magic Scroll", "price": 25.0, "inventory": 75, "competitor": 23.0}, 
        {"name": "Crystal Amulet", "price": 120.0, "inventory": 3, "competitor": 125.0}
    ]
    
    for product in test_products:
        print(f"\n📦 {product['name']} (Holiday Weekend):")
        
        # Rule-based approach
        rule_price, rule_adj = rule_based_pricing(
            product["price"], product["inventory"], True, True, "Regular"
        )
        
        # AI approach
        ai_result = predict_optimal_price(
            product["price"], product["inventory"], True, True, product["competitor"]
        )
        
        print(f"  📚 Rule-based: ${rule_price:.2f}")
        print(f"  🧠 AI-powered: ${ai_result['predicted_price']:.2f} (confidence: {ai_result['confidence_score']:.3f})")
        print(f"  📈 Expected revenue: ${ai_result['revenue_estimate']:.2f}")
        print(f"  📦 Demand forecast: {ai_result['demand_forecast']:.1f} units")
    
    # Demo 3: Strategy comparison
    print("\n\n📊 Demo 3: Strategy Performance Comparison")
    print("-" * 40)
    
    print("🔄 Comparing strategies on sample data...")
    comparison_results = compare_pricing_strategies(df, sample_size=500)
    
    print("\n🏆 Results Summary:")
    strategies = list(comparison_results.keys())
    revenues = [comparison_results[s]['total_revenue'] for s in strategies]
    
    # Find best performer
    best_strategy = max(comparison_results.keys(), 
                       key=lambda x: comparison_results[x]['total_revenue'])
    
    for strategy, results in comparison_results.items():
        is_best = "👑 " if strategy == best_strategy else "   "
        print(f"{is_best}{strategy}:")
        print(f"     💰 Total Revenue: ${results['total_revenue']:,.2f}")
        print(f"     🏷️  Avg Price: ${results['average_price']:.2f}")
        print(f"     📈 Avg Revenue/Transaction: ${results['average_revenue_per_transaction']:.2f}")
    
    # Calculate improvement
    static_revenue = comparison_results['Static Pricing']['total_revenue']
    best_revenue = comparison_results[best_strategy]['total_revenue']
    improvement = ((best_revenue / static_revenue) - 1) * 100
    
    print(f"\n🎯 Best Strategy: {best_strategy}")
    print(f"💎 Improvement over static pricing: {improvement:+.1f}%")
    print(f"💰 Extra revenue: ${best_revenue - static_revenue:,.2f}")
    
    # Demo 4: Visual comparison
    print("\n\n📈 Demo 4: Creating Visual Comparison")
    print("-" * 40)
    
    try:
        # Create visualization
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🧝‍♂️ Dynamic Pricing Strategy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Revenue comparison bar chart
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        bars = ax1.bar(strategies, revenues, color=colors)
        ax1.set_title('💰 Total Revenue Comparison')
        ax1.set_ylabel('Revenue ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, revenue in zip(bars, revenues):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${revenue:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Average price comparison
        avg_prices = [comparison_results[s]['average_price'] for s in strategies]
        ax2.bar(strategies, avg_prices, color=colors)
        ax2.set_title('🏷️ Average Price Comparison')
        ax2.set_ylabel('Average Price ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Sample data insights - Category distribution
        category_counts = df['category'].value_counts()
        ax3.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title('🛍️ Product Category Distribution')
        
        # 4. Monthly revenue trend
        df['month'] = df['timestamp'].dt.month
        monthly_revenue = df.groupby('month').apply(lambda x: (x['price_paid'] * x['quantity']).sum())
        ax4.plot(monthly_revenue.index, monthly_revenue.values, 'o-', linewidth=3, markersize=8, color='gold')
        ax4.set_title('📈 Monthly Revenue Trend')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Revenue ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pricing_demo_results.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as 'pricing_demo_results.png'")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
    
    # Demo 5: Interactive pricing scenarios
    print("\n\n🎮 Demo 5: Interactive Pricing Scenarios")
    print("-" * 40)
    print("Try these scenarios yourself:")
    
    interactive_scenarios = [
        "🎄 Holiday season with low inventory (< 10 items)",
        "📦 End-of-season clearance (> 80 items)", 
        "👑 Loyal customer on regular day",
        "🏪 Weekend shopping with competitor analysis"
    ]
    
    for scenario in interactive_scenarios:
        print(f"  • {scenario}")
    
    print(f"\n🔧 Use: from pricing_system import rule_based_pricing, predict_optimal_price")
    print(f"📚 Or explore the Jupyter notebook: jupyter notebook dynamic_pricing_elf_guide.ipynb")
    
    print(f"\n🎉 Demo completed! The pricing magic is ready for your adventures!")

if __name__ == "__main__":
    run_pricing_demo()