"""Tests for anomaly detection modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.synthetic_data import generate_operational_data
from models.anomaly_detector import AnomalyDetector, IsolationForestDetector
from evaluation.metrics import AnomalyEvaluator


class TestSyntheticData:
    """Test synthetic data generation."""
    
    def test_generate_operational_data_basic(self):
        """Test basic data generation."""
        data = generate_operational_data(n_samples=100, anomaly_rate=0.1)
        
        assert len(data['features']) == 100
        assert len(data['labels']) == 100
        assert data['features'].shape[1] >= 1
        assert data['labels'].sum() == 10  # 10% of 100
    
    def test_generate_operational_data_with_equipment(self):
        """Test data generation with equipment IDs."""
        data = generate_operational_data(
            n_samples=100, 
            anomaly_rate=0.1, 
            include_equipment=True
        )
        
        assert 'equipment_id' in data['data'].columns
        assert len(data['feature_names']) > 1  # Should have more features with equipment
    
    def test_data_consistency(self):
        """Test data consistency."""
        data = generate_operational_data(n_samples=50, anomaly_rate=0.2)
        
        # Check that features and labels have same length
        assert len(data['features']) == len(data['labels'])
        assert len(data['timestamps']) == len(data['labels'])
        
        # Check that anomaly rate is approximately correct
        actual_rate = data['labels'].mean()
        assert abs(actual_rate - 0.2) < 0.1  # Allow some tolerance


class TestAnomalyDetector:
    """Test anomaly detection models."""
    
    def test_isolation_forest_detector(self):
        """Test Isolation Forest detector."""
        detector = IsolationForestDetector(contamination=0.1, random_state=42)
        
        # Generate test data
        data = generate_operational_data(n_samples=100, anomaly_rate=0.1)
        
        # Fit and predict
        detector.fit(data['features'])
        predictions = detector.predict(data['features'])
        scores = detector.decision_function(data['features'])
        
        assert len(predictions) == len(data['features'])
        assert len(scores) == len(data['features'])
        assert set(predictions).issubset({0, 1})  # Binary predictions
        assert detector.is_fitted
    
    def test_anomaly_detector_wrapper(self):
        """Test the unified AnomalyDetector wrapper."""
        detector = AnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        data = generate_operational_data(n_samples=50, anomaly_rate=0.1)
        
        detector.fit(data['features'])
        predictions = detector.predict(data['features'])
        scores = detector.decision_function(data['features'])
        
        assert len(predictions) == len(data['features'])
        assert len(scores) == len(data['features'])
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        detector = AnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        data = generate_operational_data(n_samples=50, anomaly_rate=0.1)
        detector.fit(data['features'])
        
        # Save model
        detector.save_model('test_model.pkl')
        
        # Load model
        loaded_detector = AnomalyDetector.load_model('test_model.pkl')
        
        # Test that loaded model works
        predictions = loaded_detector.predict(data['features'])
        assert len(predictions) == len(data['features'])
        
        # Clean up
        import os
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')


class TestAnomalyEvaluator:
    """Test evaluation metrics."""
    
    def test_evaluator_basic(self):
        """Test basic evaluation functionality."""
        evaluator = AnomalyEvaluator()
        
        # Create test data
        y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0])
        y_scores = np.array([0.1, 0.8, 0.9, 0.2, 0.3, 0.1, 0.7, 0.95, 0.1, 0.2])
        
        metrics = evaluator.evaluate(y_true, y_pred, y_scores)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'total_cost' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_evaluator_with_scores(self):
        """Test evaluation with anomaly scores."""
        evaluator = AnomalyEvaluator()
        
        y_true = np.array([0, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        y_scores = np.array([0.1, 0.8, 0.9, 0.2, 0.3])
        
        metrics = evaluator.evaluate(y_true, y_pred, y_scores, k_values=[2, 3])
        
        assert 'precision_at_2' in metrics
        assert 'precision_at_3' in metrics
        assert 'aucpr' in metrics
        assert 'roc_auc' in metrics
    
    def test_evaluation_report(self):
        """Test evaluation report generation."""
        evaluator = AnomalyEvaluator()
        
        y_true = np.array([0, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        y_scores = np.array([0.1, 0.8, 0.9, 0.2, 0.3])
        
        report = evaluator.create_evaluation_report(y_true, y_pred, y_scores, "Test Model")
        
        assert isinstance(report, str)
        assert "Test Model" in report
        assert "Precision:" in report
        assert "Recall:" in report


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        data = generate_operational_data(n_samples=100, anomaly_rate=0.1)
        
        # Train model
        detector = AnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        detector.fit(data['features'])
        
        # Make predictions
        predictions = detector.predict(data['features'])
        scores = detector.decision_function(data['features'])
        
        # Evaluate
        evaluator = AnomalyEvaluator()
        metrics = evaluator.evaluate(data['labels'], predictions, scores)
        
        # Check that everything worked
        assert len(predictions) == len(data['labels'])
        assert len(scores) == len(data['labels'])
        assert 'f1_score' in metrics
        assert metrics['f1_score'] >= 0
    
    def test_multiple_algorithms(self):
        """Test multiple algorithms on same data."""
        data = generate_operational_data(n_samples=100, anomaly_rate=0.1)
        
        algorithms = ['isolation_forest', 'one_class_svm', 'lof']
        results = {}
        
        for algo in algorithms:
            detector = AnomalyDetector(algorithm=algo)
            detector.fit(data['features'])
            predictions = detector.predict(data['features'])
            scores = detector.decision_function(data['features'])
            
            evaluator = AnomalyEvaluator()
            metrics = evaluator.evaluate(data['labels'], predictions, scores)
            
            results[algo] = metrics
        
        # Check that all algorithms produced results
        assert len(results) == 3
        for algo in algorithms:
            assert algo in results
            assert 'f1_score' in results[algo]


if __name__ == "__main__":
    pytest.main([__file__])
