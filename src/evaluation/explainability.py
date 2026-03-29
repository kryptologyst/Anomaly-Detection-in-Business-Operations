"""Explainability and interpretability features for anomaly detection.

This module provides SHAP-based explainability and uncertainty quantification
for anomaly detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings

# Suppress SHAP warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)


class AnomalyExplainer:
    """Explainability wrapper for anomaly detection models."""
    
    def __init__(self, model: Any, feature_names: List[str] = None):
        """Initialize the explainer.
        
        Args:
            model: Trained anomaly detection model
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(100)]
        self.explainer = None
        self.shap_values = None
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Limited explainability features.")
    
    def fit_explainer(self, X: np.ndarray, sample_size: int = 100) -> None:
        """Fit the SHAP explainer.
        
        Args:
            X: Training data for background
            sample_size: Number of samples to use for background
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Cannot fit explainer.")
            return
        
        try:
            # Sample background data
            if len(X) > sample_size:
                background_indices = np.random.choice(len(X), sample_size, replace=False)
                background = X[background_indices]
            else:
                background = X
            
            # Create explainer based on model type
            if hasattr(self.model, 'decision_function'):
                # For models with decision_function (Isolation Forest, One-Class SVM)
                self.explainer = shap.Explainer(
                    self.model.decision_function,
                    background,
                    feature_names=self.feature_names[:X.shape[1]]
                )
            else:
                # For other models, use model directly
                self.explainer = shap.Explainer(
                    self.model,
                    background,
                    feature_names=self.feature_names[:X.shape[1]]
                )
            
            logger.info("SHAP explainer fitted successfully")
            
        except Exception as e:
            logger.error(f"Failed to fit SHAP explainer: {e}")
            self.explainer = None
    
    def explain_anomalies(
        self, 
        X: np.ndarray, 
        anomaly_indices: List[int] = None,
        max_explanations: int = 10
    ) -> Dict[str, Any]:
        """Explain anomalies using SHAP values.
        
        Args:
            X: Data to explain
            anomaly_indices: Indices of anomalies to explain
            max_explanations: Maximum number of explanations to return
            
        Returns:
            Dictionary with explanations
        """
        if self.explainer is None:
            logger.warning("Explainer not fitted. Call fit_explainer first.")
            return {}
        
        try:
            # Get SHAP values
            shap_values = self.explainer(X)
            
            # If anomaly indices not provided, find top anomalies
            if anomaly_indices is None:
                anomaly_scores = self.model.decision_function(X)
                anomaly_indices = np.argsort(anomaly_scores)[-max_explanations:].tolist()
            
            explanations = {}
            
            for idx in anomaly_indices[:max_explanations]:
                if hasattr(shap_values, 'values'):
                    # For newer SHAP versions
                    values = shap_values.values[idx]
                    base_value = shap_values.base_values[idx] if hasattr(shap_values, 'base_values') else 0
                else:
                    # For older SHAP versions
                    values = shap_values[idx]
                    base_value = 0
                
                # Get feature contributions
                feature_contributions = []
                for i, (feature_name, contribution) in enumerate(zip(self.feature_names[:len(values)], values)):
                    feature_contributions.append({
                        'feature': feature_name,
                        'contribution': float(contribution),
                        'abs_contribution': float(abs(contribution))
                    })
                
                # Sort by absolute contribution
                feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
                
                explanations[f'anomaly_{idx}'] = {
                    'index': idx,
                    'base_value': float(base_value),
                    'total_contribution': float(sum(values)),
                    'top_features': feature_contributions[:5],  # Top 5 features
                    'all_features': feature_contributions
                }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate explanations: {e}")
            return {}
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get feature importance scores.
        
        Args:
            X: Data to analyze
            
        Returns:
            Dictionary with feature importance scores
        """
        if self.explainer is None:
            logger.warning("Explainer not fitted. Call fit_explainer first.")
            return {}
        
        try:
            # Get SHAP values
            shap_values = self.explainer(X)
            
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values
            
            # Calculate mean absolute SHAP values
            mean_abs_values = np.mean(np.abs(values), axis=0)
            
            # Normalize to sum to 1
            total_importance = np.sum(mean_abs_values)
            if total_importance > 0:
                normalized_importance = mean_abs_values / total_importance
            else:
                normalized_importance = mean_abs_values
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names[:len(normalized_importance)]):
                feature_importance[feature_name] = float(normalized_importance[i])
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return {}
    
    def explain_prediction(self, x: np.ndarray) -> Dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            x: Single data point to explain
            
        Returns:
            Dictionary with explanation
        """
        if self.explainer is None:
            logger.warning("Explainer not fitted. Call fit_explainer first.")
            return {}
        
        try:
            # Reshape if needed
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.explainer(x)
            
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]
                base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0
            else:
                values = shap_values[0]
                base_value = 0
            
            # Get prediction
            prediction = self.model.predict(x)[0]
            score = self.model.decision_function(x)[0]
            
            # Create explanation
            explanation = {
                'prediction': int(prediction),
                'anomaly_score': float(score),
                'base_value': float(base_value),
                'feature_contributions': []
            }
            
            # Add feature contributions
            for i, (feature_name, contribution) in enumerate(zip(self.feature_names[:len(values)], values)):
                explanation['feature_contributions'].append({
                    'feature': feature_name,
                    'contribution': float(contribution),
                    'abs_contribution': float(abs(contribution))
                })
            
            # Sort by absolute contribution
            explanation['feature_contributions'].sort(
                key=lambda x: x['abs_contribution'], 
                reverse=True
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            return {}


class UncertaintyQuantifier:
    """Uncertainty quantification for anomaly detection."""
    
    def __init__(self, model: Any, n_bootstrap: int = 100):
        """Initialize the uncertainty quantifier.
        
        Args:
            model: Trained anomaly detection model
            n_bootstrap: Number of bootstrap samples for uncertainty estimation
        """
        self.model = model
        self.n_bootstrap = n_bootstrap
        self.bootstrap_models = []
    
    def fit_bootstrap_models(self, X: np.ndarray) -> None:
        """Fit bootstrap models for uncertainty estimation.
        
        Args:
            X: Training data
        """
        logger.info(f"Fitting {self.n_bootstrap} bootstrap models...")
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            
            # Create new model instance
            if hasattr(self.model, '__class__'):
                bootstrap_model = self.model.__class__(**self.model.get_params())
            else:
                # For custom models, create a copy
                bootstrap_model = self.model
            
            # Fit bootstrap model
            bootstrap_model.fit(X_bootstrap)
            self.bootstrap_models.append(bootstrap_model)
        
        logger.info("Bootstrap models fitted successfully")
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty estimates.
        
        Args:
            X: Data to predict on
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        if not self.bootstrap_models:
            logger.warning("Bootstrap models not fitted. Call fit_bootstrap_models first.")
            return {}
        
        # Get predictions from all bootstrap models
        all_predictions = []
        all_scores = []
        
        for model in self.bootstrap_models:
            predictions = model.predict(X)
            scores = model.decision_function(X)
            
            all_predictions.append(predictions)
            all_scores.append(scores)
        
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)
        
        # Calculate statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        mean_scores = np.mean(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)
        
        # Calculate prediction intervals
        lower_bound = np.percentile(all_scores, 2.5, axis=0)
        upper_bound = np.percentile(all_scores, 97.5, axis=0)
        
        return {
            'predictions': mean_predictions,
            'prediction_std': std_predictions,
            'scores': mean_scores,
            'score_std': std_scores,
            'confidence_interval_lower': lower_bound,
            'confidence_interval_upper': upper_bound,
            'uncertainty': std_scores  # Use score std as uncertainty measure
        }
    
    def get_uncertainty_summary(self, X: np.ndarray) -> Dict[str, float]:
        """Get summary statistics of uncertainty.
        
        Args:
            X: Data to analyze
            
        Returns:
            Dictionary with uncertainty summary statistics
        """
        results = self.predict_with_uncertainty(X)
        
        if not results:
            return {}
        
        uncertainty = results['uncertainty']
        
        return {
            'mean_uncertainty': float(np.mean(uncertainty)),
            'std_uncertainty': float(np.std(uncertainty)),
            'max_uncertainty': float(np.max(uncertainty)),
            'min_uncertainty': float(np.min(uncertainty)),
            'high_uncertainty_samples': int(np.sum(uncertainty > np.percentile(uncertainty, 90)))
        }


def create_explainability_report(
    explainer: AnomalyExplainer,
    uncertainty_quantifier: UncertaintyQuantifier,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> str:
    """Create a comprehensive explainability report.
    
    Args:
        explainer: Fitted explainer
        uncertainty_quantifier: Fitted uncertainty quantifier
        X: Data
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Formatted explainability report
    """
    report = []
    report.append("EXPLAINABILITY AND UNCERTAINTY REPORT")
    report.append("=" * 50)
    
    # Feature importance
    feature_importance = explainer.get_feature_importance(X)
    if feature_importance:
        report.append("\nFEATURE IMPORTANCE:")
        report.append("-" * 20)
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:  # Top 10 features
            report.append(f"{feature}: {importance:.3f}")
    
    # Uncertainty analysis
    uncertainty_summary = uncertainty_quantifier.get_uncertainty_summary(X)
    if uncertainty_summary:
        report.append("\nUNCERTAINTY ANALYSIS:")
        report.append("-" * 20)
        report.append(f"Mean Uncertainty: {uncertainty_summary['mean_uncertainty']:.3f}")
        report.append(f"Max Uncertainty: {uncertainty_summary['max_uncertainty']:.3f}")
        report.append(f"High Uncertainty Samples: {uncertainty_summary['high_uncertainty_samples']}")
    
    # Anomaly explanations
    anomaly_indices = np.where(y_pred == 1)[0]
    if len(anomaly_indices) > 0:
        explanations = explainer.explain_anomalies(X, anomaly_indices[:5].tolist())
        if explanations:
            report.append("\nTOP ANOMALY EXPLANATIONS:")
            report.append("-" * 25)
            
            for anomaly_id, explanation in explanations.items():
                report.append(f"\n{anomaly_id.upper()}:")
                report.append(f"  Total Contribution: {explanation['total_contribution']:.3f}")
                report.append("  Top Contributing Features:")
                
                for feature_info in explanation['top_features']:
                    report.append(f"    {feature_info['feature']}: {feature_info['contribution']:.3f}")
    
    return "\n".join(report)
