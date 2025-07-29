# Dynamic Pricing ML System - API Reference

## Table of Contents

1. [Core Data Generation API](#core-data-generation-api)
2. [Pricing Algorithm APIs](#pricing-algorithm-apis)
3. [Machine Learning APIs](#machine-learning-apis)
4. [Interactive Components API](#interactive-components-api)
5. [Visualization APIs](#visualization-apis)
6. [Utility Functions](#utility-functions)
7. [Error Handling](#error-handling)
8. [Performance Specifications](#performance-specifications)

---

## Core Data Generation API

### `generate_elves_marketplace_data()`

Generates synthetic e-commerce transaction data for pricing strategy analysis.

**Function Signature:**
```python
def generate_elves_marketplace_data() -> pd.DataFrame
```

**Description:**
Creates a realistic dataset simulating one year of e-commerce transactions with dynamic pricing factors. Uses deterministic random generation for reproducible results.

**Parameters:**
- None (all configuration is internal)

**Returns:**
- `pd.DataFrame`: Transaction dataset with 25,000 records

**DataFrame Schema:**
```python
{
    'transaction_id': str,           # Format: "TXN_XXXXXX"
    'product_id': str,               # Format: "ELF_XXX"  
    'product_name': str,             # Human-readable product name
    'category': str,                 # Product category (5 categories)
    'original_price': float,         # Base price before adjustments
    'price_paid': float,             # Final transaction price
    'quantity': int,                 # Units purchased (1-3)
    'timestamp': datetime,           # Transaction datetime
    'customer_id': str,              # Format: "CUST_XXXX"
    'customer_segment': str,         # Customer category
    'inventory_level_before_sale': int,  # Stock level (0-100)
    'competitor_price_avg': float,   # Market comparison price
    'holiday_season': int            # Binary holiday indicator
}
```

**Internal Configuration:**
```python
CONFIG = {
    'n_transactions': 25000,
    'n_products': 34,
    'date_range': ('2023-01-01', '2023-12-31'),
    'categories': ['Potions', 'Tools', 'Jewelry', 'Scrolls', 'Enchanted Items'],
    'customer_segments': ['New', 'Loyal', 'High-Value', 'Regular'],
    'random_seed': 42
}
```

**Pricing Factors Applied:**
1. **Holiday Multiplier**: 1.10 - 1.30× during holiday periods
2. **Weekend Effect**: 1.05 - 1.15× on weekends
3. **Inventory Impact**: 
   - Low stock (< 10): 1.15 - 1.25× markup
   - High stock (> 80): 0.90 - 0.95× discount
4. **Market Noise**: 0.95 - 1.05× random variation

**Usage Example:**
```python
import pandas as pd

# Generate dataset
df = generate_elves_marketplace_data()

# Basic statistics
print(f"Records: {len(df):,}")
print(f"Products: {df['product_id'].nunique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Save to file
df.to_csv('marketplace_data.csv', index=False)
```

**Performance:**
- **Execution time**: ~2-3 seconds
- **Memory usage**: ~15MB during generation
- **Output size**: ~2.5MB CSV file

**Side Effects:**
- Sets `numpy.random.seed(42)` and `random.seed(42)` for reproducibility
- Prints generation progress to stdout

---

## Pricing Algorithm APIs

### `rule_based_pricing()`

Applies business rule-based price adjustments using conditional logic.

**Function Signature:**
```python
def rule_based_pricing(original_price: float, 
                      inventory_level: int, 
                      is_holiday: bool, 
                      is_weekend: bool, 
                      customer_segment: str) -> Tuple[float, List[str]]
```

**Description:**
Implements a rule-based pricing engine that applies sequential business rules to determine optimal pricing. Each rule can modify the price and adds an explanation to the adjustment list.

**Parameters:**

| Parameter | Type | Required | Range/Values | Description |
|-----------|------|----------|--------------|-------------|
| `original_price` | `float` | Yes | > 0.0 | Base product price before adjustments |
| `inventory_level` | `int` | Yes | 0 - 100 | Current stock quantity |
| `is_holiday` | `bool` | Yes | True/False | Holiday period indicator |
| `is_weekend` | `bool` | Yes | True/False | Weekend timing indicator |
| `customer_segment` | `str` | Yes | 'New', 'Regular', 'Loyal', 'High-Value' | Customer category |

**Returns:**
- `Tuple[float, List[str]]`: 
  - `float`: Final adjusted price (rounded to 2 decimal places)
  - `List[str]`: List of applied adjustment descriptions

**Business Rules (Applied Sequentially):**

1. **Inventory-Based Pricing:**
   ```python
   if inventory_level < 10:
       price *= 1.25  # +25% scarcity premium
   elif inventory_level > 80:
       price *= 0.95  # -5% clearance discount
   ```

2. **Holiday Premium:**
   ```python
   if is_holiday:
       price *= 1.20  # +20% holiday markup
   ```

3. **Weekend Premium:**
   ```python
   if is_weekend:
       price *= 1.10  # +10% weekend markup
   ```

4. **Customer Loyalty Discount:**
   ```python
   if customer_segment == 'Loyal':
       price *= 0.90  # -10% loyalty discount
   ```

**Usage Examples:**

```python
# Example 1: High-demand scenario
price, adjustments = rule_based_pricing(
    original_price=100.0,
    inventory_level=5,      # Low stock
    is_holiday=True,        # Holiday period
    is_weekend=True,        # Weekend
    customer_segment='New'  # New customer
)
# Result: (165.00, ['📦 Low stock (+25%)', '🎄 Holiday season (+20%)', '🌅 Weekend (+10%)'])

# Example 2: Loyalty discount scenario
price, adjustments = rule_based_pricing(
    original_price=50.0,
    inventory_level=90,      # High stock
    is_holiday=False,
    is_weekend=False,
    customer_segment='Loyal'
)
# Result: (42.75, ['📦 High stock (-5%)', '👑 Loyal customer (-10%)'])

# Example 3: No adjustments
price, adjustments = rule_based_pricing(
    original_price=75.0,
    inventory_level=50,      # Normal stock
    is_holiday=False,
    is_weekend=False,
    customer_segment='Regular'
)
# Result: (75.00, [])
```

**Error Handling:**
```python
# Input validation
if original_price <= 0:
    raise ValueError("original_price must be positive")
if inventory_level < 0 or inventory_level > 100:
    raise ValueError("inventory_level must be between 0 and 100")
if customer_segment not in ['New', 'Regular', 'Loyal', 'High-Value']:
    raise ValueError("Invalid customer_segment")
```

**Algorithm Complexity:**
- **Time Complexity**: O(1) - constant time execution
- **Space Complexity**: O(1) - constant memory usage
- **Deterministic**: Same inputs always produce same outputs

---

### `predict_optimal_price()`

Machine learning-based price optimization using linear regression and revenue maximization.

**Function Signature:**
```python
def predict_optimal_price(original_price: float,
                         inventory_level: int,
                         is_holiday: bool,
                         is_weekend: bool,
                         competitor_price: float) -> Dict[str, Any]
```

**Description:**
Uses a pre-trained linear regression model to predict optimal pricing based on historical patterns. Performs revenue optimization by testing multiple price points and selecting the one that maximizes expected revenue.

**Parameters:**

| Parameter | Type | Required | Range/Values | Description |
|-----------|------|----------|--------------|-------------|
| `original_price` | `float` | Yes | > 0.0 | Base product price |
| `inventory_level` | `int` | Yes | 0 - 100 | Current inventory level |
| `is_holiday` | `bool` | Yes | True/False | Holiday season indicator |
| `is_weekend` | `bool` | Yes | True/False | Weekend indicator |
| `competitor_price` | `float` | Yes | > 0.0 | Average competitor pricing |

**Returns:**
- `Dict[str, Any]`: Comprehensive prediction results

**Return Value Schema:**
```python
{
    'predicted_price': float,        # Optimal price prediction
    'confidence_score': float,       # Model R² score (0-1)
    'revenue_estimate': float,       # Expected revenue at optimal price
    'demand_forecast': float,        # Predicted demand in units
    'price_sensitivity': float,     # Elasticity coefficient
    'optimization_details': {
        'test_prices': List[float],      # Array of tested price points
        'revenues': List[float],         # Corresponding revenue predictions
        'optimal_index': int,            # Index of optimal price in test_prices
        'revenue_curve': List[Tuple]     # (price, revenue) pairs
    },
    'model_performance': {
        'r2_score': float,               # Coefficient of determination
        'rmse': float,                   # Root mean squared error
        'training_samples': int          # Number of training records used
    }
}
```

**Machine Learning Pipeline:**

1. **Feature Engineering:**
   ```python
   features = [
       'original_price',
       'inventory_level_before_sale', 
       'holiday_season',
       'weekend',  # Derived from timestamp
       'competitor_price_avg'
   ]
   ```

2. **Model Training:**
   ```python
   from sklearn.linear_model import LinearRegression
   from sklearn.model_selection import train_test_split
   
   X_train, X_test, y_train, y_test = train_test_split(
       features, target, test_size=0.2, random_state=42
   )
   model = LinearRegression().fit(X_train, y_train)
   ```

3. **Revenue Optimization:**
   ```python
   # Test price range: 80% to 120% of original price
   test_prices = np.linspace(original_price * 0.8, original_price * 1.2, 20)
   
   # Predict demand for each price (using elasticity model)
   demands = [predict_demand(price, features) for price in test_prices]
   
   # Calculate revenue for each price point
   revenues = [price * demand for price, demand in zip(test_prices, demands)]
   
   # Select price that maximizes revenue
   optimal_index = np.argmax(revenues)
   optimal_price = test_prices[optimal_index]
   ```

**Usage Example:**
```python
# Load historical data (required for model training)
df = pd.read_csv('elves_marketplace_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)

# Predict optimal price
result = predict_optimal_price(
    original_price=75.0,
    inventory_level=45,
    is_holiday=True,
    is_weekend=False,
    competitor_price=78.0
)

# Access results
print(f"Optimal price: ${result['predicted_price']:.2f}")
print(f"Expected revenue: ${result['revenue_estimate']:.2f}")
print(f"Model confidence: {result['confidence_score']:.3f}")
print(f"Demand forecast: {result['demand_forecast']:.1f} units")

# Plot optimization curve
import matplotlib.pyplot as plt
plt.plot(result['optimization_details']['test_prices'], 
         result['optimization_details']['revenues'])
plt.xlabel('Price')
plt.ylabel('Revenue')
plt.title('Price-Revenue Optimization Curve')
plt.show()
```

**Model Performance Specifications:**
- **Training Time**: ~50ms on 25,000 records
- **Prediction Time**: <1ms per request
- **Memory Usage**: ~15MB for loaded model
- **Typical R² Score**: 0.75 - 0.85
- **RMSE**: $2-5 for prices in $10-500 range

**Dependencies:**
- Historical transaction data (CSV file)
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0

---

## Machine Learning APIs

### Model Training Functions

#### `train_pricing_model()`

**Function Signature:**
```python
def train_pricing_model(df: pd.DataFrame, 
                       features: List[str] = None,
                       target: str = 'price_paid',
                       test_size: float = 0.2) -> Tuple[LinearRegression, Dict[str, float]]
```

**Description:**
Trains a linear regression model on historical pricing data and returns the trained model with performance metrics.

**Parameters:**
- `df` (pd.DataFrame): Historical transaction data
- `features` (List[str], optional): Feature columns to use for training
- `target` (str): Target column for prediction
- `test_size` (float): Fraction of data to use for testing

**Returns:**
- `Tuple[LinearRegression, Dict[str, float]]`: (trained_model, performance_metrics)

**Default Features:**
```python
DEFAULT_FEATURES = [
    'original_price',
    'inventory_level_before_sale',
    'holiday_season', 
    'weekend',
    'competitor_price_avg'
]
```

**Performance Metrics:**
```python
{
    'r2_score': float,      # Coefficient of determination
    'rmse': float,          # Root mean squared error  
    'mae': float,           # Mean absolute error
    'training_samples': int, # Number of training records
    'test_samples': int     # Number of test records
}
```

---

### Demand Modeling Functions

#### `calculate_price_elasticity()`

**Function Signature:**
```python
def calculate_price_elasticity(df: pd.DataFrame, 
                              price_col: str = 'price_paid',
                              quantity_col: str = 'quantity') -> float
```

**Description:**
Calculates price elasticity of demand from historical data using log-log regression.

**Parameters:**
- `df` (pd.DataFrame): Historical sales data
- `price_col` (str): Column containing prices
- `quantity_col` (str): Column containing quantities sold

**Returns:**
- `float`: Price elasticity coefficient (typically negative)

**Formula:**
```
Elasticity = d(log(quantity)) / d(log(price))
```

**Usage Example:**
```python
elasticity = calculate_price_elasticity(df)
print(f"Price elasticity: {elasticity:.3f}")
# Example output: Price elasticity: -1.245
```

---

## Interactive Components API

### Widget Integration Functions

#### `create_pricing_widget()`

**Function Signature:**
```python
def create_pricing_widget() -> widgets.VBox
```

**Description:**
Creates an interactive Jupyter widget for real-time pricing experimentation.

**Returns:**
- `widgets.VBox`: Composite widget containing all pricing controls

**Widget Components:**
```python
{
    'price_slider': FloatSlider(min=10.0, max=200.0, value=50.0),
    'inventory_slider': IntSlider(min=0, max=100, value=50),
    'holiday_checkbox': Checkbox(value=False),
    'weekend_checkbox': Checkbox(value=False),
    'segment_dropdown': Dropdown(options=['New', 'Regular', 'Loyal', 'High-Value']),
    'output_area': Output()
}
```

**Usage Example:**
```python
# Create and display widget
pricing_widget = create_pricing_widget()
display(pricing_widget)

# Widget automatically updates output when values change
```

---

#### `interactive_pricing()`

**Function Signature:**
```python
def interactive_pricing(original_price: float, 
                       inventory: int, 
                       is_holiday: bool, 
                       is_weekend: bool, 
                       customer_segment: str) -> None
```

**Description:**
Widget-compatible function that displays formatted pricing analysis results.

**Parameters:**
- Same as `rule_based_pricing()` function

**Returns:**
- `None` (prints results to stdout/widget output)

**Output Format:**
```
🏷️ Original Price: $50.00 gold coins
✨ Adjusted Price: $66.00 gold coins
📊 Total Change: +32.0%

🔮 Applied Spells:
   • 📦 Low stock (+25%)
   • 🎄 Holiday season (+20%)

📈 Price increased by $16.00 gold coins
```

---

## Visualization APIs

### `create_pricing_visualization()`

**Function Signature:**
```python
def create_pricing_visualization(original_price: float = 50.0,
                                inventory_level: int = 50,
                                is_holiday: bool = False,
                                is_weekend: bool = False,
                                competitor_price: float = 52.0) -> None
```

**Description:**
Generates interactive Plotly visualizations for pricing analysis including revenue curves and demand relationships.

**Parameters:**
- `original_price` (float): Base price for analysis
- `inventory_level` (int): Inventory level for analysis
- `is_holiday` (bool): Holiday status for analysis
- `is_weekend` (bool): Weekend status for analysis
- `competitor_price` (float): Competitor price for comparison

**Generated Visualizations:**

1. **Revenue vs Price Curve**
   - X-axis: Test prices (80% - 120% of original)
   - Y-axis: Predicted revenue
   - Highlights optimal price point

2. **Demand vs Price Relationship**
   - X-axis: Test prices
   - Y-axis: Predicted demand
   - Shows price elasticity effect

3. **Profit Margin Analysis**
   - Comparative view of margins at different price points
   - Includes competitor price benchmark

4. **Price Sensitivity Heatmap**
   - Shows demand response to price changes
   - Interactive hover information

**Usage Example:**
```python
# Create visualization for specific scenario
create_pricing_visualization(
    original_price=75.0,
    inventory_level=25,
    is_holiday=True,
    is_weekend=False,
    competitor_price=80.0
)
```

**Dependencies:**
- plotly >= 5.15.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

---

### `plot_price_optimization_curve()`

**Function Signature:**
```python
def plot_price_optimization_curve(result: Dict[str, Any], 
                                 title: str = "Price Optimization Curve") -> None
```

**Description:**
Creates a detailed price optimization curve plot from ML prediction results.

**Parameters:**
- `result` (Dict): Output from `predict_optimal_price()` function
- `title` (str): Plot title

**Plot Features:**
- Price-revenue curve with optimal point highlighted
- Confidence intervals (if available)
- Current price marker
- Competitor price benchmark
- Interactive hover tooltips

---

## Utility Functions

### Data Processing Utilities

#### `prepare_features()`

**Function Signature:**
```python
def prepare_features(df: pd.DataFrame) -> pd.DataFrame
```

**Description:**
Prepares and engineers features for machine learning model training.

**Feature Engineering Steps:**
1. Extract weekend indicator from timestamp
2. Normalize price-related features
3. Create interaction terms
4. Handle missing values

**Parameters:**
- `df` (pd.DataFrame): Raw transaction data

**Returns:**
- `pd.DataFrame`: Feature-engineered dataset

---

#### `validate_pricing_inputs()`

**Function Signature:**
```python
def validate_pricing_inputs(original_price: float,
                           inventory_level: int,
                           customer_segment: str) -> bool
```

**Description:**
Validates input parameters for pricing functions.

**Validation Rules:**
- `original_price` must be positive
- `inventory_level` must be 0-100
- `customer_segment` must be valid enum value

**Parameters:**
- `original_price` (float): Price to validate
- `inventory_level` (int): Inventory to validate  
- `customer_segment` (str): Segment to validate

**Returns:**
- `bool`: True if all inputs are valid

**Raises:**
- `ValueError`: If validation fails with descriptive message

---

### Configuration Utilities

#### `load_config()`

**Function Signature:**
```python
def load_config(config_path: str = "config.json") -> Dict[str, Any]
```

**Description:**
Loads configuration parameters from JSON file.

**Default Configuration Structure:**
```json
{
    "data_generation": {
        "n_transactions": 25000,
        "n_products": 34,
        "random_seed": 42
    },
    "pricing_rules": {
        "low_stock_threshold": 10,
        "high_stock_threshold": 80,
        "holiday_premium": 0.20,
        "weekend_premium": 0.10,
        "loyalty_discount": 0.10
    },
    "ml_model": {
        "test_size": 0.2,
        "min_r2_score": 0.7,
        "price_test_range": [0.8, 1.2]
    }
}
```

---

## Error Handling

### Exception Classes

#### `PricingError`

Base exception class for pricing-related errors.

```python
class PricingError(Exception):
    """Base exception for pricing system errors"""
    pass
```

#### `ModelTrainingError`

Exception raised when ML model training fails.

```python
class ModelTrainingError(PricingError):
    """Exception raised for model training failures"""
    def __init__(self, message: str, r2_score: float = None):
        self.r2_score = r2_score
        super().__init__(message)
```

#### `InvalidInputError`

Exception raised for invalid input parameters.

```python
class InvalidInputError(PricingError):
    """Exception raised for invalid input parameters"""
    def __init__(self, parameter: str, value: Any, expected: str):
        message = f"Invalid {parameter}: {value}. Expected: {expected}"
        super().__init__(message)
```

### Error Handling Examples

```python
try:
    price, adjustments = rule_based_pricing(
        original_price=-10.0,  # Invalid negative price
        inventory_level=50,
        is_holiday=False,
        is_weekend=False,
        customer_segment='Regular'
    )
except InvalidInputError as e:
    print(f"Input error: {e}")
    # Output: Input error: Invalid original_price: -10.0. Expected: positive number

try:
    result = predict_optimal_price(
        original_price=50.0,
        inventory_level=150,  # Invalid inventory level
        is_holiday=False,
        is_weekend=False,
        competitor_price=52.0
    )
except InvalidInputError as e:
    print(f"Input error: {e}")
    # Output: Input error: Invalid inventory_level: 150. Expected: 0-100
```

---

## Performance Specifications

### Benchmarks

| Function | Dataset Size | Avg Time | Memory Usage | Throughput |
|----------|-------------|----------|--------------|------------|
| `generate_elves_marketplace_data()` | 25k records | 2.3s | 15MB | 11k records/s |
| `rule_based_pricing()` | Single request | <1ms | <1MB | >10k requests/s |
| `predict_optimal_price()` | Single request | <5ms | 15MB | >200 requests/s |
| `train_pricing_model()` | 25k records | 45ms | 25MB | - |

### Scalability Guidelines

#### For Production Deployment:

1. **Batch Processing:**
   ```python
   # Process multiple pricing requests efficiently
   def batch_rule_based_pricing(requests: List[Dict]) -> List[Tuple]:
       results = []
       for req in requests:
           price, adj = rule_based_pricing(**req)
           results.append((price, adj))
       return results
   ```

2. **Caching Strategy:**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_ml_prediction(price, inventory, holiday, weekend, competitor):
       return predict_optimal_price(price, inventory, holiday, weekend, competitor)
   ```

3. **Database Integration:**
   ```python
   # Use database for large datasets instead of CSV
   import sqlalchemy
   
   def load_data_from_db(connection_string: str) -> pd.DataFrame:
       engine = sqlalchemy.create_engine(connection_string)
       return pd.read_sql("SELECT * FROM transactions", engine)
   ```

#### Resource Requirements:

- **Minimum System Requirements:**
  - CPU: 2 cores, 2.4 GHz
  - RAM: 4GB
  - Storage: 1GB free space

- **Recommended for Production:**
  - CPU: 4+ cores, 3.0+ GHz
  - RAM: 8GB+
  - Storage: SSD with 10GB+ free space
  - Python 3.8+ with optimized libraries

#### Optimization Tips:

1. **Use vectorized operations** for batch processing
2. **Pre-compute common scenarios** and cache results
3. **Implement circuit breakers** for ML model failures
4. **Use async processing** for I/O-bound operations
5. **Monitor memory usage** and implement cleanup routines

---

This API reference provides comprehensive documentation for all functions and components in the Dynamic Pricing ML system, enabling developers to effectively integrate and extend the pricing algorithms.