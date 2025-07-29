# Dynamic Pricing ML System - Developer Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Architecture](#code-architecture)
4. [Implementation Examples](#implementation-examples)
5. [Testing Guide](#testing-guide)
6. [Extension Patterns](#extension-patterns)
7. [Deployment Strategies](#deployment-strategies)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 5-Minute Setup

```bash
# Clone the repository
git clone https://github.com/thecoder8890/dynamic-pricing-strategies-ml.git
cd dynamic-pricing-strategies-ml

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python generate_dataset.py

# Start Jupyter notebook
jupyter notebook dynamic_pricing_elf_guide.ipynb
```

### Basic Usage Example

```python
import pandas as pd
from pricing_system import rule_based_pricing, predict_optimal_price

# Load historical data
df = pd.read_csv('elves_marketplace_data.csv')

# Rule-based pricing
price, adjustments = rule_based_pricing(
    original_price=100.0,
    inventory_level=15,
    is_holiday=True,
    is_weekend=False,
    customer_segment='Loyal'
)
print(f"New price: ${price:.2f}, Adjustments: {adjustments}")

# ML-based pricing
result = predict_optimal_price(
    original_price=100.0,
    inventory_level=15,
    is_holiday=True,
    is_weekend=False,
    competitor_price=105.0
)
print(f"Optimal price: ${result['predicted_price']:.2f}")
```

---

## Development Environment Setup

### Python Environment

**Recommended Python Version**: 3.8+

```bash
# Create virtual environment
python -m venv pricing_env
source pricing_env/bin/activate  # Linux/Mac
# or
pricing_env\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

### Required Dependencies

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Jupyter environment
jupyter>=1.0.0
ipywidgets>=8.0.0

# Optional development tools
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.defaultInterpreterPath": "./pricing_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "jupyter.askForKernelRestart": false
}
```

#### PyCharm Configuration
- Set interpreter to virtual environment
- Enable numpy and pandas integration
- Configure Jupyter notebook support

---

## Code Architecture

### Project Structure

```
dynamic-pricing-strategies-ml/
├── generate_dataset.py          # Data generation module
├── create_notebook.py           # Notebook creation script
├── complete_notebook.py         # Notebook completion script  
├── dynamic_pricing_elf_guide.ipynb  # Main tutorial notebook
├── elves_marketplace_data.csv   # Generated dataset
├── requirements.txt             # Python dependencies
├── docs/                        # Documentation and diagrams
│   ├── *.png                   # Flow diagrams
│   └── README.md               # Documentation index
├── TECHNICAL_DOCUMENTATION.md  # Comprehensive tech docs
├── API_REFERENCE.md            # Function reference
└── README.md                   # Project overview
```

### Core Modules Design

#### 1. Data Generation Module (`generate_dataset.py`)

**Design Pattern**: Factory Pattern
```python
class DatasetFactory:
    """Factory for creating different types of synthetic datasets"""
    
    @staticmethod
    def create_marketplace_data(config: Dict) -> pd.DataFrame:
        """Create e-commerce marketplace dataset"""
        generator = MarketplaceDataGenerator(config)
        return generator.generate()
    
    @staticmethod
    def create_custom_data(schema: Dict, n_records: int) -> pd.DataFrame:
        """Create custom dataset with user-defined schema"""
        pass

class MarketplaceDataGenerator:
    """Generates realistic e-commerce transaction data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._setup_random_seeds()
    
    def generate(self) -> pd.DataFrame:
        """Main generation pipeline"""
        products = self._create_product_catalog()
        transactions = self._generate_transactions(products)
        return self._post_process(transactions)
```

#### 2. Pricing Algorithm Module

**Design Pattern**: Strategy Pattern
```python
from abc import ABC, abstractmethod

class PricingStrategy(ABC):
    """Abstract base class for pricing strategies"""
    
    @abstractmethod
    def calculate_price(self, context: PricingContext) -> PricingResult:
        pass

class RuleBasedPricingStrategy(PricingStrategy):
    """Rule-based pricing implementation"""
    
    def __init__(self, rules: List[PricingRule]):
        self.rules = rules
    
    def calculate_price(self, context: PricingContext) -> PricingResult:
        price = context.original_price
        adjustments = []
        
        for rule in self.rules:
            if rule.applies(context):
                price, adjustment = rule.apply(price, context)
                adjustments.append(adjustment)
        
        return PricingResult(price, adjustments)

class MLPricingStrategy(PricingStrategy):
    """Machine learning-based pricing implementation"""
    
    def __init__(self, model: Any, scaler: Any = None):
        self.model = model
        self.scaler = scaler
    
    def calculate_price(self, context: PricingContext) -> PricingResult:
        features = self._extract_features(context)
        if self.scaler:
            features = self.scaler.transform([features])
        
        predicted_price = self.model.predict([features])[0]
        return PricingResult(predicted_price, ['ML prediction'])
```

#### 3. Data Transfer Objects

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PricingContext:
    """Input context for pricing calculations"""
    original_price: float
    inventory_level: int
    is_holiday: bool
    is_weekend: bool
    customer_segment: str
    competitor_price: Optional[float] = None
    
    def validate(self) -> None:
        """Validate input parameters"""
        if self.original_price <= 0:
            raise ValueError("original_price must be positive")
        if not 0 <= self.inventory_level <= 100:
            raise ValueError("inventory_level must be 0-100")

@dataclass
class PricingResult:
    """Output result from pricing calculations"""
    final_price: float
    adjustments: List[str]
    confidence: Optional[float] = None
    
    def price_change_percent(self, original_price: float) -> float:
        """Calculate percentage change from original price"""
        return ((self.final_price / original_price) - 1) * 100
```

---

## Implementation Examples

### Example 1: Custom Pricing Strategy

```python
class SeasonalPricingStrategy(PricingStrategy):
    """Seasonal pricing with different rules for each season"""
    
    def __init__(self):
        self.seasonal_multipliers = {
            'spring': {'Potions': 1.15, 'Tools': 1.05},
            'summer': {'Tools': 1.20, 'Jewelry': 1.10},
            'fall': {'Scrolls': 1.25, 'Enchanted Items': 1.15},
            'winter': {'Potions': 1.30, 'Jewelry': 1.20}
        }
    
    def calculate_price(self, context: PricingContext) -> PricingResult:
        season = self._get_season(context.timestamp)
        category = context.product_category
        
        multiplier = self.seasonal_multipliers.get(season, {}).get(category, 1.0)
        adjusted_price = context.original_price * multiplier
        
        adjustments = []
        if multiplier != 1.0:
            change = (multiplier - 1) * 100
            adjustments.append(f"🌟 {season.title()} season ({change:+.0f}%)")
        
        return PricingResult(round(adjusted_price, 2), adjustments)
    
    def _get_season(self, timestamp: datetime) -> str:
        """Determine season from timestamp"""
        month = timestamp.month
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'fall'
        else:
            return 'winter'
```

### Example 2: Advanced ML Model with Feature Engineering

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

class AdvancedMLPricingStrategy(PricingStrategy):
    """Advanced ML pricing with feature engineering"""
    
    def __init__(self):
        self.model = None
        self.feature_pipeline = self._create_feature_pipeline()
        self.label_encoders = {}
    
    def _create_feature_pipeline(self) -> Pipeline:
        """Create feature engineering pipeline"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the model with feature engineering"""
        features = self._engineer_features(df)
        target = df['price_paid']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = self.feature_pipeline.fit(X_train, y_train)
        
        # Evaluate performance
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return metrics
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        features = df.copy()
        
        # Time-based features
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        features['month'] = df['timestamp'].dt.month
        features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Price-based features
        features['price_vs_competitor'] = df['price_paid'] / df['competitor_price_avg']
        features['markup_percentage'] = (df['price_paid'] / df['original_price'] - 1) * 100
        
        # Inventory-based features
        features['inventory_critical'] = (df['inventory_level_before_sale'] < 10).astype(int)
        features['inventory_excess'] = (df['inventory_level_before_sale'] > 80).astype(int)
        
        # Customer-based features
        if 'customer_segment' not in self.label_encoders:
            self.label_encoders['customer_segment'] = LabelEncoder()
            features['customer_segment_encoded'] = self.label_encoders['customer_segment'].fit_transform(df['customer_segment'])
        else:
            features['customer_segment_encoded'] = self.label_encoders['customer_segment'].transform(df['customer_segment'])
        
        # Category-based features
        if 'category' not in self.label_encoders:
            self.label_encoders['category'] = LabelEncoder()
            features['category_encoded'] = self.label_encoders['category'].fit_transform(df['category'])
        else:
            features['category_encoded'] = self.label_encoders['category'].transform(df['category'])
        
        # Interaction features
        features['price_inventory_interaction'] = features['original_price'] * features['inventory_level_before_sale']
        features['holiday_weekend_interaction'] = features['holiday_season'] * features['is_weekend']
        
        return features
    
    def calculate_price(self, context: PricingContext) -> PricingResult:
        """Calculate optimal price using trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert context to feature vector
        feature_dict = self._context_to_features(context)
        features_df = pd.DataFrame([feature_dict])
        
        # Make prediction
        predicted_price = self.model.predict(features_df)[0]
        
        # Get feature importance for explanation
        feature_importance = self._get_feature_importance(features_df)
        
        return PricingResult(
            final_price=round(predicted_price, 2),
            adjustments=[f"ML prediction based on {len(feature_dict)} features"],
            confidence=self._calculate_confidence(features_df)
        )
```

### Example 3: Hybrid Pricing System

```python
class HybridPricingSystem:
    """Combines multiple pricing strategies"""
    
    def __init__(self):
        self.strategies = {
            'rule_based': RuleBasedPricingStrategy([
                InventoryRule(),
                HolidayRule(),
                WeekendRule(),
                LoyaltyRule()
            ]),
            'ml_basic': MLPricingStrategy(load_basic_model()),
            'ml_advanced': AdvancedMLPricingStrategy(),
            'seasonal': SeasonalPricingStrategy()
        }
        self.weights = {
            'rule_based': 0.3,
            'ml_basic': 0.4,
            'ml_advanced': 0.2,
            'seasonal': 0.1
        }
    
    def calculate_price(self, context: PricingContext) -> PricingResult:
        """Calculate weighted average price from all strategies"""
        results = {}
        total_weight = 0
        
        for name, strategy in self.strategies.items():
            try:
                result = strategy.calculate_price(context)
                results[name] = result
                total_weight += self.weights[name]
            except Exception as e:
                print(f"Strategy {name} failed: {e}")
                continue
        
        if not results:
            raise ValueError("All pricing strategies failed")
        
        # Calculate weighted average
        weighted_price = sum(
            result.final_price * self.weights[name] 
            for name, result in results.items()
        ) / total_weight
        
        # Combine adjustments
        all_adjustments = []
        for name, result in results.items():
            weight = self.weights[name]
            all_adjustments.append(f"{name}: ${result.final_price:.2f} (weight: {weight})")
        
        return PricingResult(
            final_price=round(weighted_price, 2),
            adjustments=all_adjustments,
            confidence=self._calculate_ensemble_confidence(results)
        )
    
    def _calculate_ensemble_confidence(self, results: Dict) -> float:
        """Calculate confidence based on agreement between strategies"""
        prices = [result.final_price for result in results.values()]
        price_std = np.std(prices)
        mean_price = np.mean(prices)
        
        # Lower standard deviation relative to mean = higher confidence
        coefficient_of_variation = price_std / mean_price
        confidence = max(0, 1 - coefficient_of_variation)
        
        return round(confidence, 3)
```

### Example 4: Real-time Pricing API

```python
from flask import Flask, request, jsonify
from datetime import datetime
import redis

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class PricingAPI:
    """REST API for real-time pricing"""
    
    def __init__(self):
        self.pricing_system = HybridPricingSystem()
        self.cache_ttl = 300  # 5 minutes
    
    def get_price(self, product_id: str, customer_id: str = None) -> Dict:
        """Get price with caching"""
        cache_key = f"price:{product_id}:{customer_id or 'anonymous'}"
        
        # Check cache first
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Get product and customer context
        context = self._build_pricing_context(product_id, customer_id)
        
        # Calculate price
        result = self.pricing_system.calculate_price(context)
        
        # Prepare response
        response = {
            'product_id': product_id,
            'customer_id': customer_id,
            'original_price': context.original_price,
            'final_price': result.final_price,
            'adjustments': result.adjustments,
            'confidence': result.confidence,
            'timestamp': datetime.utcnow().isoformat(),
            'cache_hit': False
        }
        
        # Cache result
        redis_client.setex(
            cache_key, 
            self.cache_ttl, 
            json.dumps(response)
        )
        
        return response

@app.route('/api/price/<product_id>', methods=['GET'])
def get_product_price(product_id):
    """API endpoint for getting product price"""
    try:
        customer_id = request.args.get('customer_id')
        api = PricingAPI()
        result = api.get_price(product_id, customer_id)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/bulk-price', methods=['POST'])
def get_bulk_prices():
    """API endpoint for bulk pricing requests"""
    try:
        requests = request.json.get('requests', [])
        api = PricingAPI()
        
        results = []
        for req in requests:
            result = api.get_price(
                req['product_id'], 
                req.get('customer_id')
            )
            results.append(result)
        
        return jsonify({
            'results': results,
            'count': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
```

---

## Testing Guide

### Unit Testing Framework

```python
import unittest
from unittest.mock import Mock, patch
import pandas as pd

class TestRuleBasedPricing(unittest.TestCase):
    """Test cases for rule-based pricing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_price = 100.0
        self.normal_inventory = 50
        self.low_inventory = 5
        self.high_inventory = 95
    
    def test_no_adjustments(self):
        """Test pricing with no adjustments"""
        price, adjustments = rule_based_pricing(
            original_price=self.base_price,
            inventory_level=self.normal_inventory,
            is_holiday=False,
            is_weekend=False,
            customer_segment='Regular'
        )
        
        self.assertEqual(price, self.base_price)
        self.assertEqual(len(adjustments), 0)
    
    def test_low_inventory_adjustment(self):
        """Test low inventory markup"""
        price, adjustments = rule_based_pricing(
            original_price=self.base_price,
            inventory_level=self.low_inventory,
            is_holiday=False,
            is_weekend=False,
            customer_segment='Regular'
        )
        
        expected_price = self.base_price * 1.25
        self.assertEqual(price, expected_price)
        self.assertIn('Low stock', ' '.join(adjustments))
    
    def test_holiday_premium(self):
        """Test holiday premium pricing"""
        price, adjustments = rule_based_pricing(
            original_price=self.base_price,
            inventory_level=self.normal_inventory,
            is_holiday=True,
            is_weekend=False,
            customer_segment='Regular'
        )
        
        expected_price = self.base_price * 1.20
        self.assertEqual(price, expected_price)
        self.assertIn('Holiday', ' '.join(adjustments))
    
    def test_compound_adjustments(self):
        """Test multiple adjustments applied together"""
        price, adjustments = rule_based_pricing(
            original_price=self.base_price,
            inventory_level=self.low_inventory,
            is_holiday=True,
            is_weekend=True,
            customer_segment='Loyal'
        )
        
        # Calculate expected price: base * 1.25 * 1.20 * 1.10 * 0.90
        expected_price = self.base_price * 1.25 * 1.20 * 1.10 * 0.90
        self.assertAlmostEqual(price, expected_price, places=2)
        self.assertEqual(len(adjustments), 4)
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        with self.assertRaises(ValueError):
            rule_based_pricing(
                original_price=-10.0,  # Invalid negative price
                inventory_level=50,
                is_holiday=False,
                is_weekend=False,
                customer_segment='Regular'
            )
        
        with self.assertRaises(ValueError):
            rule_based_pricing(
                original_price=100.0,
                inventory_level=150,  # Invalid inventory level
                is_holiday=False,
                is_weekend=False,
                customer_segment='Regular'
            )

class TestMLPricing(unittest.TestCase):
    """Test cases for ML-based pricing"""
    
    def setUp(self):
        """Set up test fixtures with mock data"""
        self.mock_data = pd.DataFrame({
            'original_price': [50, 75, 100],
            'price_paid': [55, 82, 110],
            'inventory_level_before_sale': [20, 60, 80],
            'holiday_season': [1, 0, 1],
            'weekend': [0, 1, 0],
            'competitor_price_avg': [52, 78, 105]
        })
    
    @patch('pandas.read_csv')
    def test_model_training(self, mock_read_csv):
        """Test ML model training process"""
        mock_read_csv.return_value = self.mock_data
        
        # Mock the training process
        with patch('sklearn.linear_model.LinearRegression') as mock_lr:
            mock_model = Mock()
            mock_model.score.return_value = 0.85
            mock_lr.return_value = mock_model
            
            metrics = train_pricing_model(self.mock_data)
            
            self.assertIn('r2_score', metrics)
            self.assertGreater(metrics['r2_score'], 0.7)
    
    def test_prediction_output_format(self):
        """Test that prediction returns correct format"""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.mock_data
            
            result = predict_optimal_price(
                original_price=100.0,
                inventory_level=50,
                is_holiday=True,
                is_weekend=False,
                competitor_price=105.0
            )
            
            # Check result structure
            required_keys = [
                'predicted_price', 'confidence_score', 
                'revenue_estimate', 'demand_forecast'
            ]
            
            for key in required_keys:
                self.assertIn(key, result)
            
            # Check data types
            self.assertIsInstance(result['predicted_price'], float)
            self.assertIsInstance(result['confidence_score'], float)

class TestDataGeneration(unittest.TestCase):
    """Test cases for data generation"""
    
    def test_dataset_structure(self):
        """Test generated dataset has correct structure"""
        df = generate_elves_marketplace_data()
        
        # Check shape
        self.assertEqual(len(df), 25000)
        
        # Check required columns
        required_columns = [
            'transaction_id', 'product_id', 'product_name', 
            'category', 'original_price', 'price_paid'
        ]
        
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_price_relationships(self):
        """Test that price relationships are realistic"""
        df = generate_elves_marketplace_data()
        
        # Price paid should be related to but different from original price
        correlation = df['original_price'].corr(df['price_paid'])
        self.assertGreater(correlation, 0.8)  # Strong positive correlation
        
        # Price paid should vary (not always equal to original)
        price_differences = (df['price_paid'] != df['original_price']).sum()
        self.assertGreater(price_differences, len(df) * 0.5)  # At least 50% different

# Performance Tests
class TestPerformance(unittest.TestCase):
    """Performance testing for pricing functions"""
    
    def test_rule_based_pricing_speed(self):
        """Test rule-based pricing performance"""
        import time
        
        start_time = time.time()
        
        # Run 1000 pricing calculations
        for _ in range(1000):
            rule_based_pricing(
                original_price=100.0,
                inventory_level=50,
                is_holiday=False,
                is_weekend=False,
                customer_segment='Regular'
            )
        
        elapsed_time = time.time() - start_time
        
        # Should complete 1000 calculations in under 1 second
        self.assertLess(elapsed_time, 1.0)
        
        # Calculate throughput
        throughput = 1000 / elapsed_time
        print(f"Rule-based pricing throughput: {throughput:.0f} requests/second")

# Integration Tests
class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data generation to pricing"""
        # Generate data
        df = generate_elves_marketplace_data()
        self.assertIsNotNone(df)
        
        # Train model
        df['weekend'] = (pd.to_datetime(df['timestamp']).dt.weekday >= 5).astype(int)
        
        # Get pricing recommendations
        sample_product = df.iloc[0]
        
        # Rule-based pricing
        rule_price, rule_adjustments = rule_based_pricing(
            original_price=sample_product['original_price'],
            inventory_level=sample_product['inventory_level_before_sale'],
            is_holiday=bool(sample_product['holiday_season']),
            is_weekend=bool(sample_product['weekend']),
            customer_segment=sample_product['customer_segment']
        )
        
        self.assertIsInstance(rule_price, float)
        self.assertIsInstance(rule_adjustments, list)
        
        # ML-based pricing
        ml_result = predict_optimal_price(
            original_price=sample_product['original_price'],
            inventory_level=sample_product['inventory_level_before_sale'],
            is_holiday=bool(sample_product['holiday_season']),
            is_weekend=bool(sample_product['weekend']),
            competitor_price=sample_product['competitor_price_avg']
        )
        
        self.assertIn('predicted_price', ml_result)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
```

### Test Data Management

```python
# test_fixtures.py
import pandas as pd
import tempfile
import os

class TestDataManager:
    """Manages test data for consistent testing"""
    
    @staticmethod
    def create_minimal_dataset() -> pd.DataFrame:
        """Create minimal dataset for testing"""
        return pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
            'product_id': ['ELF_001', 'ELF_002', 'ELF_003'],
            'original_price': [50.0, 75.0, 100.0],
            'price_paid': [55.0, 80.0, 95.0],
            'inventory_level_before_sale': [10, 50, 90],
            'holiday_season': [1, 0, 1],
            'weekend': [0, 1, 0],
            'competitor_price_avg': [52.0, 78.0, 98.0],
            'customer_segment': ['New', 'Loyal', 'Regular']
        })
    
    @staticmethod
    def create_temp_csv(df: pd.DataFrame) -> str:
        """Create temporary CSV file for testing"""
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, 'test_data.csv')
        df.to_csv(temp_file, index=False)
        return temp_file
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        """Clean up temporary test files"""
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
```

---

## Extension Patterns

### Adding New Pricing Rules

```python
from abc import ABC, abstractmethod

class PricingRule(ABC):
    """Abstract base class for pricing rules"""
    
    @abstractmethod
    def applies(self, context: PricingContext) -> bool:
        """Check if rule applies to given context"""
        pass
    
    @abstractmethod
    def apply(self, price: float, context: PricingContext) -> Tuple[float, str]:
        """Apply rule and return new price and description"""
        pass

class CompetitorPriceRule(PricingRule):
    """Rule that adjusts price based on competitor pricing"""
    
    def __init__(self, max_premium: float = 0.10):
        self.max_premium = max_premium
    
    def applies(self, context: PricingContext) -> bool:
        return context.competitor_price is not None
    
    def apply(self, price: float, context: PricingContext) -> Tuple[float, str]:
        competitor_price = context.competitor_price
        
        # Don't exceed competitor price by more than max_premium
        max_allowed_price = competitor_price * (1 + self.max_premium)
        
        if price > max_allowed_price:
            adjusted_price = max_allowed_price
            discount = ((price - adjusted_price) / price) * 100
            return adjusted_price, f"🏪 Competitor match (-{discount:.1f}%)"
        
        return price, ""

class DemandSurgeRule(PricingRule):
    """Rule that increases prices during high demand periods"""
    
    def __init__(self, surge_threshold: int = 100):
        self.surge_threshold = surge_threshold
    
    def applies(self, context: PricingContext) -> bool:
        # Would need real-time demand data
        return hasattr(context, 'recent_demand') and context.recent_demand > self.surge_threshold
    
    def apply(self, price: float, context: PricingContext) -> Tuple[float, str]:
        demand_ratio = context.recent_demand / self.surge_threshold
        surge_multiplier = min(1.5, 1 + (demand_ratio - 1) * 0.3)  # Max 50% increase
        
        adjusted_price = price * surge_multiplier
        increase = (surge_multiplier - 1) * 100
        
        return adjusted_price, f"🚀 High demand (+{increase:.1f}%)"
```

### Custom Machine Learning Models

```python
from sklearn.base import BaseEstimator, RegressorMixin

class CustomPricingModel(BaseEstimator, RegressorMixin):
    """Custom ML model for pricing with domain-specific features"""
    
    def __init__(self, price_elasticity: float = -1.2):
        self.price_elasticity = price_elasticity
        self.base_model = None
    
    def fit(self, X, y):
        """Train the model with custom feature engineering"""
        # Add domain-specific features
        X_enhanced = self._add_pricing_features(X)
        
        # Use ensemble of models
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        
        self.base_model = VotingRegressor([
            ('linear', LinearRegression()),
            ('tree', DecisionTreeRegressor(max_depth=10)),
        ])
        
        self.base_model.fit(X_enhanced, y)
        return self
    
    def predict(self, X):
        """Make predictions with enhanced features"""
        X_enhanced = self._add_pricing_features(X)
        return self.base_model.predict(X_enhanced)
    
    def _add_pricing_features(self, X):
        """Add domain-specific pricing features"""
        X_enhanced = X.copy()
        
        # Price elasticity features
        if 'original_price' in X.columns:
            X_enhanced['price_squared'] = X['original_price'] ** 2
            X_enhanced['price_log'] = np.log(X['original_price'])
        
        # Inventory interaction features
        if 'inventory_level_before_sale' in X.columns and 'original_price' in X.columns:
            X_enhanced['price_inventory_ratio'] = X['original_price'] / (X['inventory_level_before_sale'] + 1)
        
        return X_enhanced
```

### Plugin Architecture

```python
import importlib
import os
from typing import Dict, Any

class PricingPluginManager:
    """Manages pricing strategy plugins"""
    
    def __init__(self, plugin_directory: str = "plugins"):
        self.plugin_directory = plugin_directory
        self.loaded_plugins = {}
    
    def load_plugins(self):
        """Dynamically load all plugins from plugin directory"""
        if not os.path.exists(self.plugin_directory):
            return
        
        for filename in os.listdir(self.plugin_directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                plugin_name = filename[:-3]  # Remove .py extension
                self._load_plugin(plugin_name)
    
    def _load_plugin(self, plugin_name: str):
        """Load a specific plugin"""
        try:
            module_path = f"{self.plugin_directory}.{plugin_name}"
            module = importlib.import_module(module_path)
            
            # Look for PricingStrategy class in the module
            if hasattr(module, 'PricingStrategy'):
                strategy_class = getattr(module, 'PricingStrategy')
                self.loaded_plugins[plugin_name] = strategy_class()
                print(f"Loaded plugin: {plugin_name}")
            
        except Exception as e:
            print(f"Failed to load plugin {plugin_name}: {e}")
    
    def get_plugin(self, plugin_name: str) -> PricingStrategy:
        """Get a loaded plugin by name"""
        return self.loaded_plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all loaded plugins"""
        return list(self.loaded_plugins.keys())

# Example plugin structure (plugins/custom_strategy.py):
"""
from pricing_system import PricingStrategy, PricingContext, PricingResult

class PricingStrategy:
    def calculate_price(self, context: PricingContext) -> PricingResult:
        # Custom pricing logic here
        adjusted_price = context.original_price * 1.1
        return PricingResult(adjusted_price, ["Custom strategy applied"])
"""
```

---

## Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 pricing_user && chown -R pricing_user:pricing_user /app
USER pricing_user

# Expose port
EXPOSE 8000

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  pricing-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/pricing
    depends_on:
      - redis
      - postgres
    volumes:
      - ./data:/app/data
    
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: pricing
      POSTGRES_User: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pricing-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pricing-api
  template:
    metadata:
      labels:
        app: pricing-api
    spec:
      containers:
      - name: pricing-api
        image: pricing-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: pricing-api-service
spec:
  selector:
    app: pricing-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### AWS Lambda Deployment

```python
# lambda_handler.py
import json
import boto3
from pricing_system import rule_based_pricing, predict_optimal_price

def lambda_handler(event, context):
    """AWS Lambda handler for pricing requests"""
    
    try:
        # Parse request
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        pricing_type = body.get('type', 'rule_based')
        params = body.get('parameters', {})
        
        # Route to appropriate pricing function
        if pricing_type == 'rule_based':
            price, adjustments = rule_based_pricing(**params)
            result = {
                'final_price': price,
                'adjustments': adjustments,
                'type': pricing_type
            }
        
        elif pricing_type == 'ml_optimal':
            result = predict_optimal_price(**params)
            result['type'] = pricing_type
        
        else:
            raise ValueError(f"Unknown pricing type: {pricing_type}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'type': 'error'
            })
        }
```

```yaml
# serverless.yml (Serverless Framework)
service: pricing-api

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  memorySize: 512
  timeout: 30
  environment:
    STAGE: ${self:provider.stage}

functions:
  pricing:
    handler: lambda_handler.lambda_handler
    events:
      - http:
          path: /price
          method: post
          cors: true
      - http:
          path: /price
          method: get
          cors: true

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Training Fails

**Symptom**: Low R² score or training errors
```python
ModelTrainingError: Model performance below threshold (R²: 0.45)
```

**Possible Causes**:
- Insufficient training data
- Poor feature quality
- Data leakage or invalid features

**Solutions**:
```python
# Check data quality
def diagnose_training_data(df):
    print("Data Quality Report:")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Check feature correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Identify highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print("Highly correlated features:")
        for col1, col2, corr in high_corr_pairs:
            print(f"  {col1} - {col2}: {corr:.3f}")

# Improve model with feature selection
from sklearn.feature_selection import SelectKBest, f_regression

def improve_model_performance(X, y):
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=10)
    X_selected = selector.fit_transform(X, y)
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    model = LinearRegression()
    scores = cross_val_score(model, X_selected, y, cv=5)
    print(f"CV scores: {scores}")
    print(f"Mean CV score: {scores.mean():.3f}")
    
    return selector, model
```

#### 2. Pricing API Performance Issues

**Symptom**: Slow response times or timeouts
```
API Response Time: 5000ms (Target: <100ms)
```

**Solutions**:
```python
# Add caching layer
import functools
import time

def cache_with_ttl(ttl_seconds=300):
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            # Clean old entries periodically
            current_time = time.time()
            expired_keys = [
                k for k, (_, ts) in cache.items() 
                if current_time - ts >= ttl_seconds
            ]
            for k in expired_keys:
                del cache[k]
            
            return result
        return wrapper
    return decorator

# Apply caching to pricing functions
@cache_with_ttl(300)  # 5-minute cache
def cached_rule_based_pricing(*args, **kwargs):
    return rule_based_pricing(*args, **kwargs)

# Add async processing for bulk requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_bulk_pricing(requests):
    """Process multiple pricing requests concurrently"""
    with ThreadPoolExecutor(max_workers=10) as executor:
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(
                executor, 
                cached_rule_based_pricing, 
                **req
            )
            for req in requests
        ]
        
        results = await asyncio.gather(*tasks)
        return results
```

#### 3. Memory Usage Issues

**Symptom**: High memory consumption or out-of-memory errors
```python
# Monitor memory usage
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

# Optimize data loading
def load_data_efficiently(filepath, chunk_size=10000):
    """Load large datasets in chunks"""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Process chunk
        processed_chunk = process_data_chunk(chunk)
        chunks.append(processed_chunk)
        
        # Explicit garbage collection
        gc.collect()
    
    return pd.concat(chunks, ignore_index=True)

# Use memory-efficient data types
def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert strings to categories if beneficial
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            # Downcast floats if possible
            if df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    
    return df
```

#### 4. Data Quality Issues

**Symptom**: Unrealistic pricing results or poor model performance

**Diagnostic Tools**:
```python
def validate_data_quality(df):
    """Comprehensive data quality validation"""
    issues = []
    
    # Price validation
    if (df['price_paid'] <= 0).any():
        issues.append("Found non-positive prices")
    
    if (df['price_paid'] > df['original_price'] * 3).any():
        issues.append("Found prices more than 3x original price")
    
    # Inventory validation
    if (df['inventory_level_before_sale'] < 0).any():
        issues.append("Found negative inventory levels")
    
    # Date validation
    if df['timestamp'].isnull().any():
        issues.append("Found missing timestamps")
    
    # Business logic validation
    weekend_prices = df[df['weekend'] == 1]['price_paid']
    weekday_prices = df[df['weekend'] == 0]['price_paid']
    
    if weekend_prices.mean() < weekday_prices.mean():
        issues.append("Weekend prices lower than weekday prices (unexpected)")
    
    return issues

# Fix common data issues
def clean_data(df):
    """Clean and fix common data issues"""
    df_clean = df.copy()
    
    # Remove outliers using IQR method
    Q1 = df_clean['price_paid'].quantile(0.25)
    Q3 = df_clean['price_paid'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df_clean[
        (df_clean['price_paid'] >= lower_bound) & 
        (df_clean['price_paid'] <= upper_bound)
    ]
    
    # Fix data types
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    df_clean['holiday_season'] = df_clean['holiday_season'].astype(bool)
    
    return df_clean
```

### Debugging Tools

```python
# Pricing decision debugger
class PricingDebugger:
    """Debug pricing decisions step by step"""
    
    def __init__(self):
        self.debug_mode = True
        self.log_steps = []
    
    def debug_rule_based_pricing(self, *args, **kwargs):
        """Debug rule-based pricing with detailed logging"""
        if not self.debug_mode:
            return rule_based_pricing(*args, **kwargs)
        
        self.log_steps = []
        original_price = kwargs.get('original_price', args[0])
        
        self.log_steps.append(f"Starting price: ${original_price:.2f}")
        
        # Step through each rule
        current_price = original_price
        
        # Inventory rule
        inventory_level = kwargs.get('inventory_level', args[1])
        if inventory_level < 10:
            current_price *= 1.25
            self.log_steps.append(f"Low inventory rule: ${current_price:.2f} (+25%)")
        elif inventory_level > 80:
            current_price *= 0.95
            self.log_steps.append(f"High inventory rule: ${current_price:.2f} (-5%)")
        
        # Continue with other rules...
        
        return current_price, self.log_steps
    
    def print_debug_log(self):
        """Print detailed debug information"""
        print("Pricing Decision Debug Log:")
        print("=" * 40)
        for i, step in enumerate(self.log_steps, 1):
            print(f"{i}. {step}")

# Performance profiler
import cProfile
import pstats

def profile_pricing_function(func, *args, **kwargs):
    """Profile pricing function performance"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
    
    return result
```

This developer guide provides comprehensive information for implementing, extending, testing, and deploying the dynamic pricing system. It covers everything from basic setup to advanced deployment strategies and troubleshooting techniques.