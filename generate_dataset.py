#!/usr/bin/env python3
"""
Generate synthetic e-commerce dataset for the Elves' Marketplace
This script creates realistic transaction data for demonstrating dynamic pricing strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_elves_marketplace_data():
    """Generate synthetic e-commerce transaction data for the Elves' Marketplace"""
    
    # Configuration
    n_transactions = 25000  # 25k transactions for good variety
    n_products = 35  # 35 unique products
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Product data
    product_categories = ["Potions", "Tools", "Jewelry", "Scrolls", "Enchanted Items"]
    product_names = [
        # Potions
        "Shimmering Health Potion", "Mystic Mana Elixir", "Sparkling Stamina Brew", 
        "Golden Healing Draught", "Emerald Energy Potion", "Ruby Restoration Tonic",
        "Moonlight Memory Potion",
        
        # Tools  
        "Enchanted Quill", "Magic Measuring Tape", "Glowing Hammer", "Mystical Chisel",
        "Starlight Saw", "Ethereal Axe", "Crystalline Compass", "Luminous Lantern",
        
        # Jewelry
        "Glimmering Gem Ring", "Sparkling Silver Necklace", "Radiant Ruby Earrings",
        "Mystical Moonstone Bracelet", "Ethereal Emerald Pendant", "Dazzling Diamond Tiara",
        "Enchanted Gold Amulet", "Celestial Crystal Crown",
        
        # Scrolls
        "Ancient Wisdom Scroll", "Spell of Prosperity", "Map of Hidden Treasures",
        "Recipe for Happiness", "Guide to Garden Magic", "Tome of Trading Secrets",
        
        # Enchanted Items
        "Magic Mirror", "Flying Carpet (Small)", "Wishing Well Water", 
        "Singing Crystal", "Dancing Shoes"
    ]
    
    # Create product master data
    products = []
    for i, name in enumerate(product_names):
        category = product_categories[i % len(product_categories)]
        # Base price varies by category
        if category == "Potions":
            base_price = np.random.uniform(15, 45)
        elif category == "Tools":
            base_price = np.random.uniform(25, 80)
        elif category == "Jewelry":
            base_price = np.random.uniform(50, 200)
        elif category == "Scrolls":
            base_price = np.random.uniform(10, 30)
        else:  # Enchanted Items
            base_price = np.random.uniform(100, 500)
            
        products.append({
            'product_id': f"ELF_{i+1:03d}",
            'product_name': name,
            'category': category,
            'base_price': round(base_price, 2)
        })
    
    # Customer segments
    customer_segments = ["New", "Loyal", "High-Value", "Regular"]
    
    # Generate transactions
    transactions = []
    
    for trans_id in range(1, n_transactions + 1):
        # Random timestamp
        days_from_start = random.randint(0, (end_date - start_date).days)
        timestamp = start_date + timedelta(days=days_from_start)
        
        # Holiday seasons (roughly 20% of the year)
        holiday_seasons = [
            (datetime(2023, 3, 15), datetime(2023, 4, 15)),  # Spring Festival
            (datetime(2023, 6, 15), datetime(2023, 7, 15)),  # Midsummer Festival  
            (datetime(2023, 11, 20), datetime(2023, 12, 25)), # Winter Celebration
        ]
        
        is_holiday = any(start <= timestamp <= end for start, end in holiday_seasons)
        
        # Select random product
        product = random.choice(products)
        product_id = product['product_id']
        product_name = product['product_name']
        category = product['category']
        original_price = product['base_price']
        
        # Dynamic pricing factors
        # 1. Holiday premium (10-30% increase)
        holiday_multiplier = 1.0
        if is_holiday:
            holiday_multiplier = np.random.uniform(1.10, 1.30)
            
        # 2. Demand variation (weekends vs weekdays)
        weekend_multiplier = 1.0
        if timestamp.weekday() >= 5:  # Weekend
            weekend_multiplier = np.random.uniform(1.05, 1.15)
            
        # 3. Inventory level effect
        inventory_level = random.randint(0, 100)
        inventory_multiplier = 1.0
        if inventory_level < 10:  # Low stock
            inventory_multiplier = np.random.uniform(1.15, 1.25)
        elif inventory_level > 80:  # High stock  
            inventory_multiplier = np.random.uniform(0.90, 0.95)
            
        # 4. Random market fluctuation
        market_noise = np.random.uniform(0.95, 1.05)
        
        # Calculate final price
        price_paid = original_price * holiday_multiplier * weekend_multiplier * inventory_multiplier * market_noise
        price_paid = round(price_paid, 2)
        
        # Quantity - higher prices tend to have lower quantities
        if price_paid > original_price * 1.2:
            quantity = random.choices([1, 2], weights=[0.8, 0.2])[0]
        else:
            quantity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            
        # Customer info
        customer_id = f"CUST_{random.randint(1, 5000):04d}"
        customer_segment = random.choice(customer_segments)
        
        # Competitor price (similar to our price with some variation)
        competitor_price_avg = round(price_paid * np.random.uniform(0.90, 1.10), 2)
        
        transaction = {
            'transaction_id': f"TXN_{trans_id:06d}",
            'product_id': product_id,
            'product_name': product_name,
            'category': category,
            'original_price': original_price,
            'price_paid': price_paid,
            'quantity': quantity,
            'timestamp': timestamp,
            'customer_id': customer_id,
            'customer_segment': customer_segment,
            'inventory_level_before_sale': inventory_level,
            'competitor_price_avg': competitor_price_avg,
            'holiday_season': 1 if is_holiday else 0
        }
        
        transactions.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def main():
    """Generate and save the dataset"""
    print("🧝 Generating Elves' Marketplace Dataset...")
    df = generate_elves_marketplace_data()
    
    # Save to CSV
    output_file = "elves_marketplace_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✨ Generated {len(df)} transactions and saved to {output_file}")
    print(f"📊 Dataset covers {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"🛍️  {df['product_id'].nunique()} unique products across {df['category'].nunique()} categories")
    print(f"👥 {df['customer_id'].nunique()} unique customers")
    print("\n🎯 Dataset Overview:")
    print(df.info())
    print("\n📈 Sample data:")
    print(df.head())

if __name__ == "__main__":
    main()