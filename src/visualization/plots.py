"""Visualization utilities for anomaly detection results.

This module provides comprehensive visualization tools for exploring
anomaly detection results and operational data patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AnomalyVisualizer:
    """Comprehensive visualization for anomaly detection results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
    
    def plot_time_series_with_anomalies(
        self,
        df: pd.DataFrame,
        metric_column: str = 'metric_value',
        anomaly_column: str = 'is_anomaly',
        timestamp_column: Optional[str] = None,
        title: str = "Operational Metrics with Anomalies",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot time series data with highlighted anomalies.
        
        Args:
            df: DataFrame with time series data
            metric_column: Name of the metric column
            anomaly_column: Name of the anomaly column
            timestamp_column: Name of the timestamp column (if not index)
            title: Plot title
            save_path: Path to save the plot
        """
        if timestamp_column:
            df = df.set_index(timestamp_column)
        
        plt.figure(figsize=self.figsize)
        
        # Plot normal points
        normal_mask = df[anomaly_column] == 0
        plt.plot(
            df.index[normal_mask],
            df[metric_column][normal_mask],
            'o-',
            color='blue',
            alpha=0.7,
            markersize=4,
            label='Normal'
        )
        
        # Plot anomalies
        anomaly_mask = df[anomaly_column] == 1
        plt.scatter(
            df.index[anomaly_mask],
            df[metric_column][anomaly_mask],
            color='red',
            s=100,
            marker='X',
            label='Anomaly',
            zorder=5
        )
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_time_series(
        self,
        df: pd.DataFrame,
        metric_column: str = 'metric_value',
        anomaly_column: str = 'is_anomaly',
        timestamp_column: Optional[str] = None,
        title: str = "Interactive Operational Metrics",
    ) -> go.Figure:
        """Create interactive time series plot with Plotly.
        
        Args:
            df: DataFrame with time series data
            metric_column: Name of the metric column
            anomaly_column: Name of the anomaly column
            timestamp_column: Name of the timestamp column (if not index)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if timestamp_column:
            df = df.set_index(timestamp_column)
        
        fig = go.Figure()
        
        # Add normal points
        normal_mask = df[anomaly_column] == 0
        fig.add_trace(go.Scatter(
            x=df.index[normal_mask],
            y=df[metric_column][normal_mask],
            mode='lines+markers',
            name='Normal',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Normal</b><br>Time: %{x}<br>Value: %{y}<extra></extra>'
        ))
        
        # Add anomalies
        anomaly_mask = df[anomaly_column] == 1
        fig.add_trace(go.Scatter(
            x=df.index[anomaly_mask],
            y=df[metric_column][anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color='red',
                size=12,
                symbol='x'
            ),
            hovertemplate='<b>Anomaly</b><br>Time: %{x}<br>Value: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Metric Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_anomaly_score_distribution(
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
        plt.figure(figsize=self.figsize)
        
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title(f'Anomaly Score Distribution - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot comparison of multiple models.
        
        Args:
            results: Dictionary of model results
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ['f1_score', 'precision', 'recall', 'aucpr']
        
        df = pd.DataFrame(results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        title: str = "Feature Importance",
        top_k: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            title: Plot title
            top_k: Number of top features to show
            save_path: Path to save the plot
        """
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        top_features = feature_names[sorted_indices[:top_k]]
        top_scores = importance_scores[sorted_indices[:top_k]]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(top_features)), top_scores, color='skyblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            plt.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_anomaly_patterns(
        self,
        df: pd.DataFrame,
        metric_column: str = 'metric_value',
        anomaly_column: str = 'is_anomaly',
        group_column: Optional[str] = None,
        title: str = "Anomaly Patterns Analysis",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot various anomaly patterns and statistics.
        
        Args:
            df: DataFrame with data
            metric_column: Name of the metric column
            anomaly_column: Name of the anomaly column
            group_column: Name of the grouping column (e.g., equipment_id)
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Anomaly rate over time
        if 'timestamp' in df.columns or df.index.name == 'timestamp':
            time_col = df.index if df.index.name == 'timestamp' else df['timestamp']
            df_time = df.copy()
            df_time['time'] = time_col
            df_time['hour'] = df_time['time'].dt.hour
            
            hourly_anomaly_rate = df_time.groupby('hour')[anomaly_column].mean()
            axes[0, 0].plot(hourly_anomaly_rate.index, hourly_anomaly_rate.values, 'o-')
            axes[0, 0].set_title('Anomaly Rate by Hour')
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Anomaly Rate')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Metric distribution by anomaly status
        normal_values = df[df[anomaly_column] == 0][metric_column]
        anomaly_values = df[df[anomaly_column] == 1][metric_column]
        
        axes[0, 1].hist(normal_values, bins=30, alpha=0.7, label='Normal', color='blue')
        axes[0, 1].hist(anomaly_values, bins=30, alpha=0.7, label='Anomaly', color='red')
        axes[0, 1].set_title('Metric Distribution by Status')
        axes[0, 1].set_xlabel('Metric Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Anomaly rate by group (if group column provided)
        if group_column and group_column in df.columns:
            group_anomaly_rate = df.groupby(group_column)[anomaly_column].mean()
            group_anomaly_rate.plot(kind='bar', ax=axes[1, 0], color='orange')
            axes[1, 0].set_title(f'Anomaly Rate by {group_column}')
            axes[1, 0].set_ylabel('Anomaly Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No grouping column provided', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Group Analysis')
        
        # 4. Rolling anomaly rate
        if len(df) > 100:
            window_size = min(50, len(df) // 10)
            rolling_anomaly_rate = df[anomaly_column].rolling(window=window_size).mean()
            axes[1, 1].plot(rolling_anomaly_rate.index, rolling_anomaly_rate.values)
            axes[1, 1].set_title(f'Rolling Anomaly Rate (window={window_size})')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Anomaly Rate')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data for rolling analysis', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Rolling Analysis')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """Create a comprehensive dashboard of results.
        
        Args:
            df: DataFrame with original data
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Anomaly scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Time series with predictions
        if 'metric_value' in df.columns:
            axes[0, 0].plot(df.index, df['metric_value'], 'b-', alpha=0.7, label='Metric Value')
            anomaly_indices = np.where(y_pred == 1)[0]
            axes[0, 0].scatter(df.index[anomaly_indices], df['metric_value'].iloc[anomaly_indices], 
                             color='red', s=50, marker='X', label='Predicted Anomalies')
            axes[0, 0].set_title('Time Series with Predictions')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Score distribution
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        axes[0, 1].hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue')
        axes[0, 1].hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 2], cmap='Blues')
        axes[0, 2].set_title('Confusion Matrix')
        axes[0, 2].set_xlabel('Predicted')
        axes[0, 2].set_ylabel('Actual')
        
        # 4. Precision-Recall curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        axes[1, 0].plot(recall, precision)
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        axes[1, 1].plot(fpr, tpr)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_title('ROC Curve')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Model performance summary
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics_text = f"""
        Model: {model_name}
        
        Precision: {report['1']['precision']:.3f}
        Recall: {report['1']['recall']:.3f}
        F1-Score: {report['1']['f1-score']:.3f}
        
        Accuracy: {report['accuracy']:.3f}
        """
        
        axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes, 
                        fontsize=12, verticalalignment='center')
        axes[1, 2].set_title('Performance Summary')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Anomaly Detection Dashboard - {model_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
