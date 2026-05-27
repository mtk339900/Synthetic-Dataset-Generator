# 🎲 Synthetic Dataset Generator

> **Generate realistic, production-ready datasets for AI training, software testing, and data analysis — in minutes, not weeks.**

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 💡 Why This Tool Exists

Collecting real-world data for AI/ML projects is slow, expensive, and often restricted by privacy laws. This library solves that by generating **statistically realistic synthetic datasets** that behave like real data — complete with correlations, trends, and distributions — so your team can build and test faster.

**Use this if you need to:**
- Train or test machine learning models without waiting for real data
- Generate realistic dummy data for software QA and demos
- Simulate business scenarios (e-commerce, finance, IoT sensors)
- Create privacy-compliant datasets for data science projects

---

## ✨ What It Can Generate

| Data Type | Examples |
|-----------|---------|
| **Numeric** | Revenue, age, temperature — with 6 statistical distributions |
| **Categorical** | Product categories, customer tiers — with custom probability weights |
| **DateTime** | Time-series with trends (increasing, seasonal, decreasing) |
| **Boolean** | Flags like `is_premium`, `is_active` with custom probabilities |
| **Text** | Product descriptions, customer reviews, tags |
| **Correlated columns** | Marketing spend ↔ Revenue (define the relationship strength) |

**Output formats:** CSV · JSON · Excel

---

## 🚀 Quick Start

```bash
pip install numpy pandas openpyxl
```

```python
from datetime import datetime
from synthetic_dataset_generator import DatasetBuilder, NumericDistribution

# Generate a realistic e-commerce dataset in 5 lines
df = (DatasetBuilder(seed=42)
    .add_numeric_column("order_amount", NumericDistribution.GAMMA, 10, 500)
    .add_categorical_column("category", ["Electronics", "Clothing", "Books"], [0.4, 0.35, 0.25])
    .add_datetime_column("order_date", datetime(2023, 1, 1), datetime(2024, 12, 31))
    .add_boolean_column("is_return", true_probability=0.08)
    .generate(10000))

print(df.head())
```

**Output:**
```
   order_amount    category  order_date          is_return
0        247.3   Electronics 2023-03-15          False
1         89.1    Clothing   2023-07-22          False
2        412.8   Electronics 2024-01-09          True
3         34.2      Books    2023-11-30          False
```

---

## 🏭 Real-World Examples

<details>
<summary><b>📦 E-Commerce Dataset (10,000 orders)</b></summary>

```python
ecommerce_data = (DatasetBuilder(seed=123)
    .add_numeric_column("order_amount", NumericDistribution.GAMMA, 10, 500, alpha=2, beta=2)
    .add_categorical_column("product_category",
                          ["Electronics", "Clothing", "Books", "Home", "Sports"],
                          [0.25, 0.3, 0.15, 0.2, 0.1])
    .add_datetime_column("order_date", datetime(2023, 1, 1), datetime(2024, 12, 31))
    .add_boolean_column("is_return", true_probability=0.08)
    .add_numeric_column("customer_satisfaction", NumericDistribution.BETA, 1, 5, alpha=3, beta=1)
    .add_correlation("order_amount", "customer_satisfaction", 0.4)
    .generate(10000))
```
</details>

<details>
<summary><b>💰 Financial Dataset (5,000 accounts)</b></summary>

```python
financial_data = (DatasetBuilder(seed=456)
    .add_numeric_column("account_balance", NumericDistribution.NORMAL, 0, 100000, mean=25000, std=15000)
    .add_numeric_column("monthly_income", NumericDistribution.NORMAL, 2000, 15000, mean=6000, std=2500)
    .add_categorical_column("risk_profile", ["Conservative", "Moderate", "Aggressive"], [0.4, 0.4, 0.2])
    .add_boolean_column("has_loan", true_probability=0.35)
    .add_correlation("monthly_income", "account_balance", 0.65)
    .generate(5000))
```
</details>

<details>
<summary><b>📡 IoT Sensor Dataset (744 hours = 1 month)</b></summary>

```python
sensor_data = (DatasetBuilder(seed=789)
    .add_datetime_column("timestamp", datetime(2024, 1, 1), datetime(2024, 1, 31),
                        frequency="H", trend="seasonal")
    .add_numeric_column("temperature", NumericDistribution.NORMAL, -10, 40, mean=20, std=8)
    .add_numeric_column("humidity", NumericDistribution.BETA, 0, 100, alpha=2, beta=2)
    .add_categorical_column("sensor_status", ["OK", "Warning", "Error"], [0.85, 0.12, 0.03])
    .add_correlation("temperature", "humidity", -0.3)
    .generate(744))
```
</details>

---

## ⚙️ Supported Distributions

| Distribution | Best For |
|-------------|---------|
| `NORMAL` | General data — salaries, heights, test scores |
| `UNIFORM` | Random IDs, evenly spread values |
| `EXPONENTIAL` | Wait times, inter-arrival times |
| `POISSON` | Event counts per time period |
| `BETA` | Rates and percentages (bounded 0–1) |
| `GAMMA` | Skewed data — order amounts, response times |

---

## 📋 Requirements

- Python 3.7+
- numpy
- pandas
- openpyxl *(for Excel export)*

---

## 📄 License

MIT — free for personal and commercial use.

---

## 👨‍💻 About the Author

**Mohammed Mustafa** — Python Developer specializing in AI tools, data automation, and ML pipelines.

If you need a **custom dataset generator, data pipeline, or AI automation tool** built for your specific use case, feel free to reach out:

- 🔗 [LinkedIn](https://www.linkedin.com/in/mohammed-mostafa-658013408)
- 💼 [Upwork Profile](https://www.upwork.com/freelancers/~01117f10fdcfeac100?mp_source=share)
- 📧 Available for freelance projects

---

*Built with ❤️ for the data science and AI community*
