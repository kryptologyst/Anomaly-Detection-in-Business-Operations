"""Synthetic operational data generation for anomaly detection testing.

This module provides functions to generate realistic operational data
with various anomaly patterns for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def generate_operational_data(
    n_samples: int = 1000,
    anomaly_rate: float = 0.05,
    include_equipment: bool = True,
    include_seasonality: bool = True,
    include_trend: bool = True,
    random_state: int = 42,
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    """Generate synthetic operational data with anomalies.
    
    Args:
        n_samples: Number of samples to generate
        anomaly_rate: Proportion of samples that should be anomalies
        include_equipment: Whether to include equipment identifiers
        include_seasonality: Whether to include seasonal patterns
        include_trend: Whether to include trend components
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'data': DataFrame with features and metadata
            - 'features': Feature matrix for ML algorithms
            - 'labels': Binary anomaly labels (1=anomaly, 0=normal)
            - 'timestamps': DateTime index
    """
    np.random.seed(random_state)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=n_samples//24)  # Assuming hourly data
    timestamps = pd.date_range(
        start=start_time, 
        periods=n_samples, 
        freq='h'  # Use lowercase 'h' instead of 'H'
    )
    
    # Generate base normal data
    base_value = 50.0
    noise_std = 5.0
    
    # Add trend component
    if include_trend:
        trend = np.linspace(0, 2, n_samples)  # Gradual upward trend
    else:
        trend = np.zeros(n_samples)
    
    # Add seasonal component
    if include_seasonality:
        # Daily seasonality (24-hour cycle)
        daily_seasonality = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
        # Weekly seasonality
        weekly_seasonality = 2 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))
        seasonality = daily_seasonality + weekly_seasonality
    else:
        seasonality = np.zeros(n_samples)
    
    # Generate normal values
    normal_values = (
        base_value + 
        trend + 
        seasonality + 
        np.random.normal(0, noise_std, n_samples)
    )
    
    # Generate anomaly labels
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(
        n_samples, 
        size=n_anomalies, 
        replace=False
    )
    
    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1
    
    # Generate anomaly values (various types)
    values = normal_values.copy()
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'drop', 'drift'])
        
        if anomaly_type == 'spike':
            # Sudden spike
            values[idx] += np.random.uniform(20, 40)
        elif anomaly_type == 'drop':
            # Sudden drop
            values[idx] -= np.random.uniform(20, 40)
        else:  # drift
            # Gradual drift (affect multiple consecutive points)
            drift_length = min(5, n_samples - idx)
            drift_values = np.random.uniform(5, 15, drift_length)
            values[idx:idx+drift_length] += drift_values
    
    # Create DataFrame
    data_dict = {
        'timestamp': timestamps,
        'metric_value': values,
        'is_anomaly': labels.astype(int),
    }
    
    if include_equipment:
        # Generate equipment IDs
        equipment_ids = np.random.choice(
            [f'EQ_{i:03d}' for i in range(1, 11)], 
            size=n_samples
        )
        data_dict['equipment_id'] = equipment_ids
        
        # Add equipment-specific patterns
        for eq_id in np.unique(equipment_ids):
            eq_mask = equipment_ids == eq_id
            # Add slight equipment-specific bias
            equipment_bias = np.random.normal(0, 2)
            values[eq_mask] += equipment_bias
    
    df = pd.DataFrame(data_dict)
    df.set_index('timestamp', inplace=True)
    
    # Prepare features for ML algorithms
    feature_columns = ['metric_value']
    if include_equipment:
        # One-hot encode equipment IDs
        equipment_dummies = pd.get_dummies(df['equipment_id'], prefix='equipment')
        df = pd.concat([df, equipment_dummies], axis=1)
        feature_columns.extend(equipment_dummies.columns.tolist())
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    feature_columns.extend(['hour', 'day_of_week', 'is_weekend'])
    
    features = df[feature_columns].values
    
    logger.info(f"Generated {n_samples} samples with {n_anomalies} anomalies ({anomaly_rate:.1%})")
    
    return {
        'data': df,
        'features': features,
        'labels': labels,
        'timestamps': timestamps,
        'feature_names': feature_columns,
    }


def generate_multivariate_operational_data(
    n_samples: int = 1000,
    n_metrics: int = 5,
    anomaly_rate: float = 0.05,
    correlation_strength: float = 0.7,
    random_state: int = 42,
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    """Generate multivariate operational data with correlated metrics.
    
    Args:
        n_samples: Number of samples to generate
        n_metrics: Number of operational metrics
        anomaly_rate: Proportion of samples that should be anomalies
        correlation_strength: Strength of correlation between metrics
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing multivariate data with correlations
    """
    np.random.seed(random_state)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=n_samples//24)
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq='h')
    
    # Generate correlated normal data
    mean_values = np.random.uniform(30, 70, n_metrics)
    cov_matrix = np.eye(n_metrics) * 5  # Base variance
    
    # Add correlations
    for i in range(n_metrics):
        for j in range(i+1, n_metrics):
            cov_matrix[i, j] = cov_matrix[j, i] = correlation_strength * 5
    
    normal_data = np.random.multivariate_normal(mean_values, cov_matrix, n_samples)
    
    # Generate anomaly labels
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    
    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1
    
    # Create anomaly patterns
    data = normal_data.copy()
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['global', 'local', 'shift'])
        
        if anomaly_type == 'global':
            # All metrics affected
            shift = np.random.uniform(-20, 20, n_metrics)
            data[idx] += shift
        elif anomaly_type == 'local':
            # Only some metrics affected
            affected_metrics = np.random.choice(n_metrics, size=np.random.randint(1, n_metrics), replace=False)
            shift = np.random.uniform(-15, 15, len(affected_metrics))
            data[idx, affected_metrics] += shift
        else:  # shift
            # Systematic shift in all metrics
            shift = np.random.uniform(-10, 10)
            data[idx] += shift
    
    # Create DataFrame
    metric_names = [f'metric_{i+1}' for i in range(n_metrics)]
    df_data = {
        'timestamp': timestamps,
        'is_anomaly': labels.astype(int),
    }
    
    for i, metric_name in enumerate(metric_names):
        df_data[metric_name] = data[:, i]
    
    df = pd.DataFrame(df_data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Generated multivariate data: {n_samples} samples, {n_metrics} metrics, {n_anomalies} anomalies")
    
    return {
        'data': df,
        'features': data,
        'labels': labels,
        'timestamps': timestamps,
        'metric_names': metric_names,
    }


def add_contextual_features(
    df: pd.DataFrame,
    include_weather: bool = True,
    include_maintenance: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add contextual features to operational data.
    
    Args:
        df: Input DataFrame with operational data
        include_weather: Whether to add weather-related features
        include_maintenance: Whether to add maintenance schedule features
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with additional contextual features
    """
    np.random.seed(random_state)
    
    df = df.copy()
    
    if include_weather:
        # Simulate weather impact
        df['temperature'] = np.random.normal(20, 10, len(df))
        df['humidity'] = np.random.uniform(30, 90, len(df))
        df['weather_impact'] = (
            -0.1 * (df['temperature'] - 20) + 
            0.05 * (df['humidity'] - 60)
        )
    
    if include_maintenance:
        # Simulate maintenance schedule
        maintenance_days = np.random.choice(len(df), size=len(df)//30, replace=False)
        df['days_since_maintenance'] = 0
        
        for i, day in enumerate(maintenance_days):
            if i == 0:
                df.loc[:day, 'days_since_maintenance'] = np.arange(day+1)
            else:
                prev_day = maintenance_days[i-1]
                df.loc[prev_day:day, 'days_since_maintenance'] = np.arange(day-prev_day+1)
        
        # Fill remaining days
        last_maintenance = maintenance_days[-1] if len(maintenance_days) > 0 else 0
        df.loc[last_maintenance:, 'days_since_maintenance'] = np.arange(len(df) - last_maintenance)
    
    return df
