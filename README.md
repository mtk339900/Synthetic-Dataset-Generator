# Synthetic Dataset Generator

A comprehensive Python library for generating synthetic datasets for simulation, testing, and training purposes. Create realistic datasets with configurable data types, distributions, correlations, and output formats.

## Features

### 🎯 **Data Types**
- **Numeric**: Normal, Uniform, Exponential, Poisson, Beta, Gamma distributions
- **Categorical**: Custom categories with probability weights
- **DateTime**: Time-series with trends, seasonality, and configurable frequency
- **Boolean**: Adjustable true/false probabilities
- **Text**: Pattern-based text generation (words, sentences, custom vocabularies)

### ⚙️ **Advanced Capabilities**
- **Correlations**: Define relationships between numeric columns
- **Time Series**: Realistic temporal data with trends and seasonal patterns
- **Output Formats**: CSV, JSON, Excel support
- **Reproducibility**: Random seed control for consistent results
- **Validation**: Comprehensive parameter validation and error handling

## Installation

```bash
pip install numpy pandas openpyxl
```

## Quick Start

```python
from datetime import datetime
from synthetic_dataset_generator import DatasetBuilder, NumericDistribution, OutputFormat

# Create a dataset with method chaining
builder = DatasetBuilder(seed=42)
df = (builder
    .add_numeric_column("revenue", NumericDistribution.NORMAL, 1000, 50000)
    .add_categorical_column("department", ["Sales", "Marketing", "Engineering"])
    .add_datetime_column("date", datetime(2020, 1, 1), datetime(2024, 12, 31))
    .add_boolean_column("is_active", true_probability=0.8)
    .generate(1000))

print(df.head())
```

## Detailed Usage

### Numeric Columns

Generate numeric data with various statistical distributions:

```python
# Normal distribution
builder.add_numeric_column(
    name="sales_amount",
    distribution=NumericDistribution.NORMAL,
    min_value=0,
    max_value=10000,
    mean=5000,
    std=1500,
    decimals=2
)

# Beta distribution for bounded values
builder.add_numeric_column(
    name="completion_rate",
    distribution=NumericDistribution.BETA,
    min_value=0,
    max_value=1,
    alpha=2.0,
    beta=3.0,
    decimals=3
)

# Exponential distribution
builder.add_numeric_column(
    name="wait_time",
    distribution=NumericDistribution.EXPONENTIAL,
    min_value=0,
    max_value=120,
    lambda_param=0.5
)
```

### Categorical Columns

Create categorical data with custom categories and weights:

```python
# Equal probability categories
builder.add_categorical_column(
    name="product_category",
    categories=["Electronics", "Clothing", "Books", "Home"]
)

# Weighted categories
builder.add_categorical_column(
    name="customer_tier",
    categories=["Bronze", "Silver", "Gold", "Platinum"],
    weights=[0.4, 0.3, 0.2, 0.1]
)
```

### DateTime Columns

Generate time-series data with various patterns:

```python
# Daily data with increasing trend
builder.add_datetime_column(
    name="transaction_date",
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    frequency="D",  # Daily
    trend="increasing"
)

# Hourly data with seasonal pattern
builder.add_datetime_column(
    name="timestamp",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    frequency="H",  # Hourly
    trend="seasonal"
)
```

### Boolean Columns

```python
# 70% probability of True
builder.add_boolean_column(
    name="is_premium_customer",
    true_probability=0.7
)
```

### Text Columns

Generate realistic text data:

```python
# Sentence-like text
builder.add_text_column(
    name="product_description",
    min_length=20,
    max_length=100,
    patterns=["sentence"],
    word_list=["innovative", "reliable", "premium", "efficient", "advanced"]
)

# Word-based text
builder.add_text_column(
    name="tags",
    min_length=5,
    max_length=25,
    patterns=["word"]
)
```

### Adding Correlations

Create realistic relationships between numeric columns:

```python
builder = (DatasetBuilder(seed=42)
    .add_numeric_column("marketing_spend", NumericDistribution.NORMAL, 1000, 10000)
    .add_numeric_column("sales_revenue", NumericDistribution.NORMAL, 5000, 50000)
    .add_correlation("marketing_spend", "sales_revenue", 0.75))  # Strong positive correlation
```

### Advanced Configuration with DataGenerator

For complex scenarios, use the `DataGenerator` class directly:

```python
from synthetic_dataset_generator import (
    DataGenerator, ColumnSpec, NumericConfig, CategoricalConfig,
    DataType, NumericDistribution, OutputFormat
)

generator = DataGenerator(seed=42)

# Custom numeric configuration
numeric_config = NumericConfig(
    distribution=NumericDistribution.GAMMA,
    min_value=0,
    max_value=1000,
    alpha=2.0,
    beta=1.5,
    decimals=2
)
generator.add_column(ColumnSpec("response_time", DataType.NUMERIC, numeric_config))

# Custom categorical configuration
categorical_config = CategoricalConfig(
    categories=["Low", "Medium", "High", "Critical"],
    weights=[0.1, 0.3, 0.4, 0.2]
)
generator.add_column(ColumnSpec("priority", DataType.CATEGORICAL, categorical_config))

# Generate and save
df = generator.generate(5000)
generator.save(df, "complex_dataset.xlsx", OutputFormat.EXCEL)
```

## Output Formats

Save your datasets in multiple formats:

```python
# CSV
generator.save(df, "dataset.csv", OutputFormat.CSV)

# JSON
generator.save(df, "dataset.json", OutputFormat.JSON)

# Excel
generator.save(df, "dataset.xlsx", OutputFormat.EXCEL)
```

## Real-World Examples

### E-commerce Dataset

```python
from datetime import datetime

ecommerce_data = (DatasetBuilder(seed=123)
    .add_numeric_column("order_amount", NumericDistribution.GAMMA, 10, 500, alpha=2, beta=2)
    .add_categorical_column("product_category", 
                          ["Electronics", "Clothing", "Books", "Home", "Sports"],
                          [0.25, 0.3, 0.15, 0.2, 0.1])
    .add_datetime_column("order_date", datetime(2023, 1, 1), datetime(2024, 12, 31))
    .add_boolean_column("is_return", true_probability=0.08)
    .add_text_column("customer_review", min_length=10, max_length=200, patterns=["sentence"])
    .add_numeric_column("customer_satisfaction", NumericDistribution.BETA, 1, 5, alpha=3, beta=1)
    .add_correlation("order_amount", "customer_satisfaction", 0.4)
    .generate(10000))
```

### Financial Dataset

```python
financial_data = (DatasetBuilder(seed=456)
    .add_numeric_column("account_balance", NumericDistribution.NORMAL, 0, 100000, mean=25000, std=15000)
    .add_numeric_column("monthly_income", NumericDistribution.NORMAL, 2000, 15000, mean=6000, std=2500)
    .add_categorical_column("risk_profile", ["Conservative", "Moderate", "Aggressive"], [0.4, 0.4, 0.2])
    .add_boolean_column("has_loan", true_probability=0.35)
    .add_datetime_column("account_opened", datetime(2015, 1, 1), datetime(2024, 1, 1))
    .add_correlation("monthly_income", "account_balance", 0.65)
    .generate(5000))
```

### IoT Sensor Dataset

```python
sensor_data = (DatasetBuilder(seed=789)
    .add_datetime_column("timestamp", datetime(2024, 1, 1), datetime(2024, 1, 31), 
                        frequency="H", trend="seasonal")
    .add_numeric_column("temperature", NumericDistribution.NORMAL, -10, 40, mean=20, std=8)
    .add_numeric_column("humidity", NumericDistribution.BETA, 0, 100, alpha=2, beta=2)
    .add_numeric_column("pressure", NumericDistribution.NORMAL, 980, 1040, mean=1013, std=10)
    .add_categorical_column("sensor_status", ["OK", "Warning", "Error"], [0.85, 0.12, 0.03])
    .add_boolean_column("is_calibrated", true_probability=0.95)
    .add_correlation("temperature", "humidity", -0.3)  # Negative correlation
    .generate(744))  # 31 days * 24 hours
```

## API Reference

### DatasetBuilder Methods

| Method | Description |
|--------|-------------|
| `add_numeric_column()` | Add numeric column with distribution |
| `add_categorical_column()` | Add categorical column with categories and weights |
| `add_datetime_column()` | Add datetime column with trends |
| `add_boolean_column()` | Add boolean column with probability |
| `add_text_column()` | Add text column with patterns |
| `add_correlation()` | Define correlation between numeric columns |
| `generate()` | Generate the dataset |
| `generate_and_save()` | Generate and save in one step |

### Supported Distributions

| Distribution | Parameters | Use Case |
|-------------|------------|----------|
| Normal | mean, std | General continuous data |
| Uniform | min_value, max_value | Evenly distributed data |
| Exponential | lambda_param | Wait times, lifetimes |
| Poisson | lambda_param | Count data |
| Beta | alpha, beta | Bounded continuous data (0-1) |
| Gamma | alpha, beta | Skewed positive data |

### DateTime Frequencies

| Frequency | Description |
|-----------|-------------|
| `"D"` or `"daily"` | Daily intervals |
| `"H"` or `"hourly"` | Hourly intervals |
| `"W"` or `"weekly"` | Weekly intervals |
| `"M"` or `"monthly"` | Monthly intervals |

### DateTime Trends

| Trend | Description |
|-------|-------------|
| `"none"` | No trend |
| `"increasing"` | Upward trend |
| `"decreasing"` | Downward trend |
| `"seasonal"` | Seasonal pattern |

## Error Handling

The library provides comprehensive validation:

```python
# Invalid correlation strength
try:
    builder.add_correlation("col1", "col2", 1.5)  # > 1.0
except ValueError as e:
    print(f"Error: {e}")

# Invalid column name
try:
    builder.add_numeric_column("123invalid")  # Starts with number
except ValueError as e:
    print(f"Error: {e}")

# Non-positive definite correlation matrix
# Automatically handled with warnings
```

## Performance Considerations

- **Memory Usage**: Large datasets (>1M rows) may require significant memory
- **Correlations**: Complex correlation matrices increase generation time
- **Text Generation**: Text columns are slower than numeric/categorical columns
- **File Size**: Excel format is slower and larger than CSV for large datasets

## Best Practices

1. **Set Random Seed**: Always use a seed for reproducible results
2. **Validate Correlations**: Keep correlation strengths reasonable (< 0.9)
3. **Choose Appropriate Distributions**: Match distributions to your data characteristics
4. **Memory Management**: Generate large datasets in chunks if memory is limited
5. **File Formats**: Use CSV for large datasets, Excel for smaller ones with formatting needs

## Contributing

This is a standalone implementation. For enhancements:

1. Add new distribution types in `NumericDistribution` enum
2. Extend correlation support to categorical variables
3. Add more text generation patterns
4. Implement data streaming for very large datasets

## License

This implementation is provided as-is for educational and commercial use.

## Requirements

- Python 3.7+
- numpy
- pandas
- openpyxl (for Excel support)

---

**Happy Dataset Generation!** 🎲📊