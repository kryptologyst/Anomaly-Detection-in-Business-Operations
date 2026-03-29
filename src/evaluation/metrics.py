"""Evaluation metrics and analysis for anomaly detection.

This module provides comprehensive evaluation metrics specifically designed
for anomaly detection in operational contexts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class AnomalyEvaluator:
    """Comprehensive evaluation for anomaly detection models."""
    
    def __init__(self, cost_false_positive: float = 10.0, cost_false_negative: float = 100.0):
        """Initialize the evaluator.
        
        Args:
            cost_false_positive: Cost of a false positive (unnecessary alert)
            cost_false_negative: Cost of a false negative (missed anomaly)
        """
        self.cost_fp = cost_false_positive
        self.cost_fn = cost_false_negative
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        k_values: List[int] = None,
    ) -> Dict[str, float]:
        """Evaluate anomaly detection performance.
        
        Args:
            y_true: True binary labels (1=anomaly, 0=normal)
            y_pred: Predicted binary labels (1=anomaly, 0=normal)
            y_scores: Anomaly scores (optional, for ranking metrics)
            k_values: K values for precision@K calculation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if k_values is None:
            k_values = [10, 50, 100]
        
        metrics = {}
        
        # Basic classification metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall']
        ) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        
        # Specificity and sensitivity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = metrics['recall']
        
        # Cost-based metrics
        total_cost = fp * self.cost_fp + fn * self.cost_fn
        metrics['total_cost'] = total_cost
        metrics['cost_per_sample'] = total_cost / len(y_true)
        
        # Alert workload metrics
        metrics['alert_rate'] = np.mean(y_pred)
        metrics['true_anomaly_rate'] = np.mean(y_true)
        metrics['alert_efficiency'] = metrics['precision']
        
        # Ranking metrics (if scores provided)
        if y_scores is not None:
            metrics.update(self._calculate_ranking_metrics(y_true, y_scores, k_values))
        
        logger.info(f"Evaluation completed. F1: {metrics['f1_score']:.3f}, "
                   f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        
        return metrics
    
    def _calculate_ranking_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k_values: List[int],
    ) -> Dict[str, float]:
        """Calculate ranking-based metrics.
        
        Args:
            y_true: True binary labels
            y_scores: Anomaly scores
            k_values: K values for precision@K
            
        Returns:
            Dictionary of ranking metrics
        """
        metrics = {}
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        # Precision@K
        for k in k_values:
            if k <= len(sorted_labels):
                precision_at_k = np.mean(sorted_labels[:k])
                metrics[f'precision_at_{k}'] = precision_at_k
        
        # AUCPR (Area Under Precision-Recall Curve)
        if len(np.unique(y_true)) > 1:
            metrics['aucpr'] = average_precision_score(y_true, y_scores)
        else:
            metrics['aucpr'] = 0.0
        
        # ROC AUC
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        else:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def create_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        model_name: str = "Model",
    ) -> str:
        """Create a detailed evaluation report.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Anomaly scores
            model_name: Name of the model for the report
            
        Returns:
            Formatted evaluation report
        """
        metrics = self.evaluate(y_true, y_pred, y_scores)
        
        report = f"""
ANOMALY DETECTION EVALUATION REPORT
===================================
Model: {model_name}

CLASSIFICATION METRICS:
- Accuracy: {metrics['accuracy']:.3f}
- Precision: {metrics['precision']:.3f}
- Recall (Sensitivity): {metrics['recall']:.3f}
- Specificity: {metrics['specificity']:.3f}
- F1-Score: {metrics['f1_score']:.3f}

ALERT WORKLOAD METRICS:
- Alert Rate: {metrics['alert_rate']:.3f}
- True Anomaly Rate: {metrics['true_anomaly_rate']:.3f}
- Alert Efficiency: {metrics['alert_efficiency']:.3f}

COST ANALYSIS:
- Total Cost: ${metrics['total_cost']:.2f}
- Cost per Sample: ${metrics['cost_per_sample']:.3f}
- False Positive Cost: ${self.cost_fp:.2f} per alert
- False Negative Cost: ${self.cost_fn:.2f} per missed anomaly
"""
        
        if y_scores is not None:
            report += f"""
RANKING METRICS:
- AUCPR: {metrics['aucpr']:.3f}
- ROC AUC: {metrics['roc_auc']:.3f}
"""
            
            # Add precision@K metrics
            for key, value in metrics.items():
                if key.startswith('precision_at_'):
                    report += f"- {key.replace('_', '@').title()}: {value:.3f}\n"
        
        return report
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot precision-recall curve.
        
        Args:
            y_true: True binary labels
            y_scores: Anomaly scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        aucpr = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AUCPR = {aucpr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_anomaly_scores_distribution(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot distribution of anomaly scores by true label.
        
        Args:
            y_true: True binary labels
            y_scores: Anomaly scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title(f'Anomaly Score Distribution - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_leaderboard(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = 'f1_score',
    secondary_metric: str = 'precision',
) -> pd.DataFrame:
    """Create a leaderboard from multiple model results.
    
    Args:
        results: Dictionary of model results
        primary_metric: Primary metric for ranking
        secondary_metric: Secondary metric for tie-breaking
        
    Returns:
        DataFrame with ranked results
    """
    df = pd.DataFrame(results).T
    
    # Sort by primary metric, then secondary metric
    df = df.sort_values([primary_metric, secondary_metric], ascending=False)
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ['rank', primary_metric, secondary_metric] + [
        col for col in df.columns if col not in ['rank', primary_metric, secondary_metric]
    ]
    df = df[cols]
    
    return df


def calculate_business_impact(
    metrics: Dict[str, float],
    baseline_cost: float = 1000.0,
    alert_handling_cost: float = 50.0,
) -> Dict[str, float]:
    """Calculate business impact metrics.
    
    Args:
        metrics: Evaluation metrics
        baseline_cost: Baseline operational cost
        alert_handling_cost: Cost per alert to investigate
        
    Returns:
        Dictionary of business impact metrics
    """
    impact = {}
    
    # Cost savings from detecting anomalies
    anomalies_detected = metrics['recall'] * metrics['true_anomaly_rate']
    cost_saved_per_anomaly = baseline_cost * 0.1  # Assume 10% cost reduction per detected anomaly
    impact['cost_saved'] = anomalies_detected * cost_saved_per_anomaly
    
    # Alert handling costs
    total_alerts = metrics['alert_rate']
    impact['alert_handling_cost'] = total_alerts * alert_handling_cost
    
    # Net benefit
    impact['net_benefit'] = impact['cost_saved'] - impact['alert_handling_cost']
    
    # ROI
    if impact['alert_handling_cost'] > 0:
        impact['roi'] = impact['net_benefit'] / impact['alert_handling_cost']
    else:
        impact['roi'] = float('inf')
    
    return impact
