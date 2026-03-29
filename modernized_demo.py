#!/usr/bin/env python3
"""Modernized anomaly detection demonstration script.

This script demonstrates the enhanced anomaly detection capabilities
with multiple algorithms, comprehensive evaluation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.synthetic_data import generate_operational_data
from models.anomaly_detector import AnomalyDetector
from evaluation.metrics import AnomalyEvaluator, create_leaderboard
from visualization.plots import AnomalyVisualizer

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """Main demonstration function."""
    print("=" * 60)
    print("ANOMALY DETECTION IN OPERATIONS - DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Generate synthetic operational data
    print("📊 Generating synthetic operational data...")
    data = generate_operational_data(
        n_samples=1000,
        anomaly_rate=0.05,
        include_equipment=True,
        include_seasonality=True,
        include_trend=True,
        random_state=42
    )
    
    print(f"   Generated {len(data['features'])} samples with {data['labels'].sum()} anomalies")
    print()
    
    # Initialize models to compare
    algorithms = {
        'Isolation Forest': AnomalyDetector(
            algorithm='isolation_forest',
            contamination=0.05,
            n_estimators=100,
            random_state=42
        ),
        'One-Class SVM': AnomalyDetector(
            algorithm='one_class_svm',
            kernel='rbf',
            nu=0.1
        ),
        'Local Outlier Factor': AnomalyDetector(
            algorithm='lof',
            n_neighbors=20,
            contamination=0.05
        )
    }
    
    # Train and evaluate models
    results = {}
    evaluator = AnomalyEvaluator(cost_false_positive=10.0, cost_false_negative=100.0)
    
    print("🤖 Training and evaluating models...")
    for name, detector in algorithms.items():
        print(f"   Training {name}...")
        
        # Train model
        detector.fit(data['features'])
        
        # Make predictions
        predictions = detector.predict(data['features'])
        scores = detector.decision_function(data['features'])
        
        # Evaluate model
        metrics = evaluator.evaluate(
            data['labels'],
            predictions,
            scores,
            k_values=[10, 50, 100]
        )
        
        results[name] = {
            'predictions': predictions,
            'scores': scores,
            'metrics': metrics
        }
        
        print(f"   {name} - F1: {metrics['f1_score']:.3f}, "
              f"Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}")
    
    print()
    
    # Create leaderboard
    print("📈 Model Performance Leaderboard:")
    print("-" * 50)
    
    metrics_dict = {name: result['metrics'] for name, result in results.items()}
    leaderboard = create_leaderboard(metrics_dict)
    
    print(leaderboard[['rank', 'f1_score', 'precision', 'recall', 'aucpr']].to_string(index=True))
    print()
    
    # Business impact analysis
    print("💰 Business Impact Analysis:")
    print("-" * 30)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name}:")
        print(f"  Total Cost: ${metrics['total_cost']:.2f}")
        print(f"  Alert Rate: {metrics['alert_rate']:.3f}")
        print(f"  Alert Efficiency: {metrics['alert_efficiency']:.3f}")
        print()
    
    # Create visualizations
    print("📊 Creating visualizations...")
    
    visualizer = AnomalyVisualizer(figsize=(12, 8))
    
    # Plot time series with anomalies for the best model
    best_model = leaderboard.index[0]
    best_predictions = results[best_model]['predictions']
    
    df = data['data'].copy()
    df['predicted_anomaly'] = best_predictions
    
    visualizer.plot_time_series_with_anomalies(
        df,
        metric_column='metric_value',
        anomaly_column='predicted_anomaly',
        title=f'Operational Metrics with Anomalies - {best_model}'
    )
    
    # Plot score distribution for the best model
    best_scores = results[best_model]['scores']
    visualizer.plot_anomaly_score_distribution(
        data['labels'],
        best_scores,
        model_name=best_model
    )
    
    # Plot model comparison
    visualizer.plot_model_comparison(
        metrics_dict,
        title="Model Comparison"
    )
    
    print("✅ Demonstration completed successfully!")
    print()
    print("🔍 Key Insights:")
    print(f"   • Best performing model: {best_model}")
    print(f"   • Best F1-Score: {leaderboard.iloc[0]['f1_score']:.3f}")
    print(f"   • Lowest cost: ${min([r['metrics']['total_cost'] for r in results.values()]):.2f}")
    print()
    print("⚠️  REMINDER: This is for research/education purposes only.")
    print("   Do NOT use for automated decision-making without human review.")
    print("=" * 60)


if __name__ == "__main__":
    main()
