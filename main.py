import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import warnings


class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"


class NumericDistribution(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    BETA = "beta"
    GAMMA = "gamma"


class OutputFormat(Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"


@dataclass
class NumericConfig:
    distribution: NumericDistribution = NumericDistribution.NORMAL
    min_value: float = 0.0
    max_value: float = 100.0
    mean: Optional[float] = None
    std: Optional[float] = None
    lambda_param: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    decimals: int = 2
    
    def __post_init__(self):
        if self.distribution == NumericDistribution.NORMAL:
            if self.mean is None:
                self.mean = (self.min_value + self.max_value) / 2
            if self.std is None:
                self.std = (self.max_value - self.min_value) / 6


@dataclass
class CategoricalConfig:
    categories: List[str] = field(default_factory=list)
    weights: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.categories:
            self.categories = [f"Category_{i}" for i in range(1, 6)]
        if self.weights and len(self.weights) != len(self.categories):
            raise ValueError("Weights must match the number of categories")


@dataclass
class DateTimeConfig:
    start_date: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2024, 12, 31))
    frequency: str = "D"  # D=daily, H=hourly, M=monthly, etc.
    trend: str = "none"  # none, increasing, decreasing, seasonal
    seasonality_period: Optional[int] = None


@dataclass
class BooleanConfig:
    true_probability: float = 0.5
    
    def __post_init__(self):
        if not 0 <= self.true_probability <= 1:
            raise ValueError("True probability must be between 0 and 1")


@dataclass
class TextConfig:
    min_length: int = 5
    max_length: int = 50
    patterns: List[str] = field(default_factory=lambda: ["word", "sentence"])
    word_list: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.word_list is None:
            self.word_list = [
                "data", "analysis", "machine", "learning", "artificial", "intelligence",
                "algorithm", "model", "training", "prediction", "classification",
                "regression", "neural", "network", "feature", "target", "dataset",
                "validation", "testing", "optimization", "performance", "accuracy"
            ]


@dataclass
class ColumnSpec:
    name: str
    data_type: DataType
    config: Union[NumericConfig, CategoricalConfig, DateTimeConfig, BooleanConfig, TextConfig]
    correlation_targets: Optional[List[str]] = None
    correlation_strength: float = 0.0


class CorrelationMatrix:
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.matrix = np.eye(len(columns))
        self.column_index = {col: i for i, col in enumerate(columns)}
    
    def add_correlation(self, col1: str, col2: str, strength: float):
        if not -1 <= strength <= 1:
            raise ValueError("Correlation strength must be between -1 and 1")
        
        i = self.column_index[col1]
        j = self.column_index[col2]
        self.matrix[i, j] = strength
        self.matrix[j, i] = strength
    
    def is_positive_definite(self) -> bool:
        try:
            np.linalg.cholesky(self.matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def make_positive_definite(self):
        eigenvals, eigenvecs = np.linalg.eigh(self.matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        self.matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        np.fill_diagonal(self.matrix, 1.0)


class DataGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.columns: List[ColumnSpec] = []
        self.correlation_matrix: Optional[CorrelationMatrix] = None
    
    def add_column(self, column_spec: ColumnSpec):
        if any(col.name == column_spec.name for col in self.columns):
            raise ValueError(f"Column '{column_spec.name}' already exists")
        
        self._validate_column_spec(column_spec)
        self.columns.append(column_spec)
    
    def _validate_column_spec(self, spec: ColumnSpec):
        if not spec.name or not isinstance(spec.name, str):
            raise ValueError("Column name must be a non-empty string")
        
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', spec.name):
            raise ValueError("Column name must be a valid identifier")
        
        if spec.data_type == DataType.NUMERIC and not isinstance(spec.config, NumericConfig):
            raise ValueError("Numeric columns require NumericConfig")
        elif spec.data_type == DataType.CATEGORICAL and not isinstance(spec.config, CategoricalConfig):
            raise ValueError("Categorical columns require CategoricalConfig")
        elif spec.data_type == DataType.DATETIME and not isinstance(spec.config, DateTimeConfig):
            raise ValueError("DateTime columns require DateTimeConfig")
        elif spec.data_type == DataType.BOOLEAN and not isinstance(spec.config, BooleanConfig):
            raise ValueError("Boolean columns require BooleanConfig")
        elif spec.data_type == DataType.TEXT and not isinstance(spec.config, TextConfig):
            raise ValueError("Text columns require TextConfig")
    
    def _setup_correlations(self):
        numeric_columns = [col.name for col in self.columns if col.data_type == DataType.NUMERIC]
        
        if not numeric_columns:
            return
        
        self.correlation_matrix = CorrelationMatrix(numeric_columns)
        
        for col in self.columns:
            if (col.data_type == DataType.NUMERIC and 
                col.correlation_targets and 
                col.correlation_strength != 0):
                
                for target in col.correlation_targets:
                    if target in numeric_columns:
                        self.correlation_matrix.add_correlation(
                            col.name, target, col.correlation_strength
                        )
        
        if not self.correlation_matrix.is_positive_definite():
            warnings.warn("Correlation matrix is not positive definite. Adjusting...")
            self.correlation_matrix.make_positive_definite()
    
    def _generate_numeric_data(self, spec: ColumnSpec, n_rows: int) -> np.ndarray:
        config = spec.config
        
        if config.distribution == NumericDistribution.NORMAL:
            data = np.random.normal(config.mean, config.std, n_rows)
        elif config.distribution == NumericDistribution.UNIFORM:
            data = np.random.uniform(config.min_value, config.max_value, n_rows)
        elif config.distribution == NumericDistribution.EXPONENTIAL:
            scale = config.lambda_param or 1.0
            data = np.random.exponential(scale, n_rows)
            data = np.clip(data, config.min_value, config.max_value)
        elif config.distribution == NumericDistribution.POISSON:
            lam = config.lambda_param or 5.0
            data = np.random.poisson(lam, n_rows).astype(float)
        elif config.distribution == NumericDistribution.BETA:
            alpha = config.alpha or 2.0
            beta = config.beta or 2.0
            data = np.random.beta(alpha, beta, n_rows)
            data = data * (config.max_value - config.min_value) + config.min_value
        elif config.distribution == NumericDistribution.GAMMA:
            shape = config.alpha or 2.0
            scale = config.beta or 1.0
            data = np.random.gamma(shape, scale, n_rows)
            data = np.clip(data, config.min_value, config.max_value)
        else:
            raise ValueError(f"Unsupported distribution: {config.distribution}")
        
        # Apply bounds for non-uniform distributions
        if config.distribution != NumericDistribution.UNIFORM:
            data = np.clip(data, config.min_value, config.max_value)
        
        return np.round(data, config.decimals)
    
    def _generate_categorical_data(self, spec: ColumnSpec, n_rows: int) -> List[str]:
        config = spec.config
        weights = config.weights if config.weights else None
        return np.random.choice(config.categories, size=n_rows, p=weights).tolist()
    
    def _generate_datetime_data(self, spec: ColumnSpec, n_rows: int) -> List[datetime]:
        config = spec.config
        
        if config.frequency in ['D', 'daily']:
            delta = timedelta(days=1)
        elif config.frequency in ['H', 'hourly']:
            delta = timedelta(hours=1)
        elif config.frequency in ['M', 'monthly']:
            delta = timedelta(days=30)
        elif config.frequency in ['W', 'weekly']:
            delta = timedelta(weeks=1)
        else:
            delta = timedelta(days=1)
        
        time_span = config.end_date - config.start_date
        max_periods = int(time_span / delta)
        
        if n_rows > max_periods:
            # Generate random dates within the range
            timestamps = []
            for _ in range(n_rows):
                random_days = random.randint(0, (config.end_date - config.start_date).days)
                timestamp = config.start_date + timedelta(days=random_days)
                timestamps.append(timestamp)
        else:
            # Generate sequential dates with trend
            base_dates = [config.start_date + i * delta for i in range(n_rows)]
            
            if config.trend == "increasing":
                # Add slight forward drift
                timestamps = [date + timedelta(days=i*0.1) for i, date in enumerate(base_dates)]
            elif config.trend == "decreasing":
                # Add slight backward drift
                timestamps = [date - timedelta(days=i*0.1) for i, date in enumerate(base_dates)]
            elif config.trend == "seasonal" and config.seasonality_period:
                # Add seasonal pattern
                timestamps = []
                for i, date in enumerate(base_dates):
                    seasonal_offset = np.sin(2 * np.pi * i / config.seasonality_period) * 5
                    timestamps.append(date + timedelta(days=seasonal_offset))
            else:
                timestamps = base_dates
        
        return timestamps
    
    def _generate_boolean_data(self, spec: ColumnSpec, n_rows: int) -> List[bool]:
        config = spec.config
        return np.random.choice([True, False], size=n_rows, 
                               p=[config.true_probability, 1-config.true_probability]).tolist()
    
    def _generate_text_data(self, spec: ColumnSpec, n_rows: int) -> List[str]:
        config = spec.config
        texts = []
        
        for _ in range(n_rows):
            length = random.randint(config.min_length, config.max_length)
            
            if "sentence" in config.patterns:
                # Generate sentence-like text
                num_words = max(1, length // 6)  # Approximate words based on length
                words = random.choices(config.word_list, k=num_words)
                text = " ".join(words).capitalize() + "."
            elif "word" in config.patterns:
                # Generate single words or concatenated words
                if length <= 15:
                    text = random.choice(config.word_list)
                else:
                    words = random.choices(config.word_list, k=2)
                    text = "_".join(words)
            else:
                # Generate random character sequence
                chars = "abcdefghijklmnopqrstuvwxyz"
                text = "".join(random.choices(chars, k=min(length, 20)))
            
            # Truncate or pad to desired length
            if len(text) > length:
                text = text[:length]
            elif len(text) < config.min_length:
                text = text + "x" * (config.min_length - len(text))
            
            texts.append(text)
        
        return texts
    
    def _apply_correlations(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.correlation_matrix:
            return data_dict
        
        numeric_columns = self.correlation_matrix.columns
        if len(numeric_columns) < 2:
            return data_dict
        
        # Extract numeric data
        numeric_data = np.column_stack([data_dict[col] for col in numeric_columns])
        
        # Standardize
        means = np.mean(numeric_data, axis=0)
        stds = np.std(numeric_data, axis=0)
        standardized = (numeric_data - means) / stds
        
        # Apply correlation using Cholesky decomposition
        try:
            chol = np.linalg.cholesky(self.correlation_matrix.matrix)
            correlated = standardized @ chol.T
            
            # Unstandardize
            correlated = correlated * stds + means
            
            # Update the data dictionary
            for i, col in enumerate(numeric_columns):
                data_dict[col] = correlated[:, i]
        
        except np.linalg.LinAlgError:
            warnings.warn("Could not apply correlations due to matrix decomposition failure")
        
        return data_dict
    
    def generate(self, n_rows: int) -> pd.DataFrame:
        if n_rows <= 0:
            raise ValueError("Number of rows must be positive")
        
        if not self.columns:
            raise ValueError("No columns specified")
        
        # Setup correlations
        self._setup_correlations()
        
        data_dict = {}
        
        # Generate data for each column
        for spec in self.columns:
            if spec.data_type == DataType.NUMERIC:
                data_dict[spec.name] = self._generate_numeric_data(spec, n_rows)
            elif spec.data_type == DataType.CATEGORICAL:
                data_dict[spec.name] = self._generate_categorical_data(spec, n_rows)
            elif spec.data_type == DataType.DATETIME:
                data_dict[spec.name] = self._generate_datetime_data(spec, n_rows)
            elif spec.data_type == DataType.BOOLEAN:
                data_dict[spec.name] = self._generate_boolean_data(spec, n_rows)
            elif spec.data_type == DataType.TEXT:
                data_dict[spec.name] = self._generate_text_data(spec, n_rows)
        
        # Apply correlations to numeric columns
        data_dict = self._apply_correlations(data_dict)
        
        return pd.DataFrame(data_dict)
    
    def save(self, df: pd.DataFrame, filepath: str, format_type: OutputFormat):
        if format_type == OutputFormat.CSV:
            df.to_csv(filepath, index=False)
        elif format_type == OutputFormat.JSON:
            df.to_json(filepath, orient='records', date_format='iso', indent=2)
        elif format_type == OutputFormat.EXCEL:
            df.to_excel(filepath, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported output format: {format_type}")


class DatasetBuilder:
    """Convenience class for building datasets with method chaining"""
    
    def __init__(self, seed: Optional[int] = None):
        self.generator = DataGenerator(seed)
    
    def add_numeric_column(self, name: str, distribution: NumericDistribution = NumericDistribution.NORMAL,
                          min_value: float = 0.0, max_value: float = 100.0,
                          mean: Optional[float] = None, std: Optional[float] = None,
                          decimals: int = 2, **kwargs) -> 'DatasetBuilder':
        config = NumericConfig(
            distribution=distribution,
            min_value=min_value,
            max_value=max_value,
            mean=mean,
            std=std,
            decimals=decimals,
            **kwargs
        )
        spec = ColumnSpec(name, DataType.NUMERIC, config)
        self.generator.add_column(spec)
        return self
    
    def add_categorical_column(self, name: str, categories: List[str],
                             weights: Optional[List[float]] = None) -> 'DatasetBuilder':
        config = CategoricalConfig(categories=categories, weights=weights)
        spec = ColumnSpec(name, DataType.CATEGORICAL, config)
        self.generator.add_column(spec)
        return self
    
    def add_datetime_column(self, name: str, start_date: datetime, end_date: datetime,
                           frequency: str = "D", trend: str = "none") -> 'DatasetBuilder':
        config = DateTimeConfig(
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            trend=trend
        )
        spec = ColumnSpec(name, DataType.DATETIME, config)
        self.generator.add_column(spec)
        return self
    
    def add_boolean_column(self, name: str, true_probability: float = 0.5) -> 'DatasetBuilder':
        config = BooleanConfig(true_probability=true_probability)
        spec = ColumnSpec(name, DataType.BOOLEAN, config)
        self.generator.add_column(spec)
        return self
    
    def add_text_column(self, name: str, min_length: int = 5, max_length: int = 50,
                       patterns: List[str] = None, word_list: List[str] = None) -> 'DatasetBuilder':
        if patterns is None:
            patterns = ["word", "sentence"]
        config = TextConfig(
            min_length=min_length,
            max_length=max_length,
            patterns=patterns,
            word_list=word_list
        )
        spec = ColumnSpec(name, DataType.TEXT, config)
        self.generator.add_column(spec)
        return self
    
    def add_correlation(self, col1: str, col2: str, strength: float) -> 'DatasetBuilder':
        # Find the column and add correlation target
        for col in self.generator.columns:
            if col.name == col1:
                if col.correlation_targets is None:
                    col.correlation_targets = []
                col.correlation_targets.append(col2)
                col.correlation_strength = strength
                break
        return self
    
    def generate(self, n_rows: int) -> pd.DataFrame:
        return self.generator.generate(n_rows)
    
    def generate_and_save(self, n_rows: int, filepath: str, 
                         format_type: OutputFormat) -> pd.DataFrame:
        df = self.generate(n_rows)
        self.generator.save(df, filepath, format_type)
        return df