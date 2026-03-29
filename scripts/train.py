#!/usr/bin/env python3
"""Training script for anomaly detection models.

This script trains anomaly detection models using the configuration
specified in the config file.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data.synthetic_data import generate_operational_data
from models.anomaly_detector import AnomalyDetector
from evaluation.metrics import AnomalyEvaluator, create_leaderboard
from visualization.plots import AnomalyVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_single_model(
    config: Dict[str, Any],
    data: Dict[str, Any],
    model_name: str = "model"
) -> Dict[str, Any]:
    """Train a single anomaly detection model.
    
    Args:
        config: Configuration dictionary
        data: Generated data dictionary
        model_name: Name for the model
        
    Returns:
        Dictionary with model results
    """
    logger.info(f"Training {model_name}...")
    
    # Initialize model
    model_config = config['model']
    detector = AnomalyDetector(
        algorithm=model_config['algorithm'],
        **model_config.get(model_config['algorithm'], {})
    )
    
    # Train model
    detector.fit(data['features'])
    
    # Make predictions
    predictions = detector.predict(data['features'])
    scores = detector.decision_function(data['features'])
    
    # Evaluate model
    evaluator = AnomalyEvaluator(
        cost_false_positive=config['evaluation']['cost_false_positive'],
        cost_false_negative=config['evaluation']['cost_false_negative']
    )
    
    metrics = evaluator.evaluate(
        data['labels'],
        predictions,
        scores,
        config['evaluation']['k_values']
    )
    
    # Save model
    model_path = f"models/{model_name}.pkl"
    os.makedirs("models", exist_ok=True)
    detector.save_model(model_path)
    
    logger.info(f"{model_name} training completed. F1: {metrics['f1_score']:.3f}")
    
    return {
        'model': detector,
        'predictions': predictions,
        'scores': scores,
        'metrics': metrics,
        'model_path': model_path
    }


def train_multiple_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train multiple models for comparison.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with all model results
    """
    logger.info("Generating synthetic data...")
    
    # Generate data
    data_config = config['data']
    data = generate_operational_data(
        n_samples=data_config['n_samples'],
        anomaly_rate=data_config['anomaly_rate'],
        include_equipment=data_config['include_equipment'],
        include_seasonality=data_config['include_seasonality'],
        include_trend=data_config['include_trend'],
        random_state=data_config['random_state']
    )
    
    logger.info(f"Generated {len(data['features'])} samples with {data['labels'].sum()} anomalies")
    
    # Train models
    results = {}
    models_config = config.get('models', {config['model']['algorithm']: config['model']})
    
    for model_name, model_config in models_config.items():
        # Update config for this model
        temp_config = config.copy()
        temp_config['model'] = model_config
        
        results[model_name] = train_single_model(temp_config, data, model_name)
    
    # Create leaderboard
    metrics_dict = {name: result['metrics'] for name, result in results.items()}
    leaderboard = create_leaderboard(metrics_dict)
    
    logger.info("Model comparison completed")
    logger.info(f"\nLeaderboard:\n{leaderboard}")
    
    # Save results
    os.makedirs("assets", exist_ok=True)
    leaderboard.to_csv("assets/leaderboard.csv")
    
    return {
        'data': data,
        'results': results,
        'leaderboard': leaderboard
    }


def create_visualizations(
    data: Dict[str, Any],
    results: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """Create visualization plots.
    
    Args:
        data: Generated data
        results: Model results
        config: Configuration
    """
    logger.info("Creating visualizations...")
    
    visualizer = AnomalyVisualizer(figsize=tuple(config['visualization']['figsize']))
    
    # Create plots for each model
    for model_name, result in results.items():
        logger.info(f"Creating plots for {model_name}...")
        
        # Time series plot
        df = data['data'].copy()
        df['predicted_anomaly'] = result['predictions']
        
        visualizer.plot_time_series_with_anomalies(
            df,
            metric_column='metric_value',
            anomaly_column='predicted_anomaly',
            title=f"Anomaly Detection Results - {model_name}",
            save_path=f"assets/{model_name}_time_series.png" if config['visualization']['save_plots'] else None
        )
        
        # Score distribution
        visualizer.plot_anomaly_score_distribution(
            data['labels'],
            result['scores'],
            model_name=model_name,
            save_path=f"assets/{model_name}_score_distribution.png" if config['visualization']['save_plots'] else None
        )
        
        # Dashboard
        if config['visualization'].get('create_dashboard', False):
            visualizer.create_dashboard(
                data['data'],
                data['labels'],
                result['predictions'],
                result['scores'],
                model_name=model_name,
                save_path=f"assets/{model_name}_dashboard.png" if config['visualization']['save_plots'] else None
            )
    
    # Model comparison plot
    if len(results) > 1:
        metrics_dict = {name: result['metrics'] for name, result in results.items()}
        visualizer.plot_model_comparison(
            metrics_dict,
            title="Model Comparison",
            save_path="assets/model_comparison.png" if config['visualization']['save_plots'] else None
        )
    
    logger.info("Visualizations completed")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for models and results"
    )
    
    args = parser.parse_args()
    
    # Change to output directory
    os.chdir(args.output_dir)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds
    np.random.seed(config['data']['random_state'])
    
    # Train models
    training_results = train_multiple_models(config)
    
    # Create visualizations
    create_visualizations(
        training_results['data'],
        training_results['results'],
        config
    )
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
