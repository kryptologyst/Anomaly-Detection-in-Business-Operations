"""Anomaly detection algorithms for operational data.

This module provides a unified interface for various anomaly detection algorithms
commonly used in business operations monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import logging
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

logger = logging.getLogger(__name__)


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detection algorithms."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the detector.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.is_fitted = False
        self.scaler = StandardScaler()
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fit the anomaly detection model.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies in the data.
        
        Args:
            X: Data to predict on
            
        Returns:
            Binary array (1=anomaly, 0=normal)
        """
        pass
    
    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state,
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseAnomalyDetector':
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        model = model_data['model']
        
        logger.info(f"Model loaded from {filepath}")
        return model


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[str, int, float] = 'auto',
        random_state: int = 42,
    ):
        """Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
        )
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Isolation Forest model...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info("Isolation Forest model fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using Isolation Forest.
        
        Args:
            X: Data to predict on
            
        Returns:
            Binary array (1=anomaly, 0=normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert from (-1, 1) to (1, 0) where 1=anomaly
        return (predictions == -1).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using Isolation Forest.
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to positive scores (higher = more anomalous)
        return -scores


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: Union[str, float] = 'scale',
        nu: float = 0.1,
        random_state: int = 42,
    ):
        """Initialize One-Class SVM detector.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            gamma: Kernel coefficient
            nu: Upper bound on fraction of training errors
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        
        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
        )
    
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit the One-Class SVM model.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting One-Class SVM model...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info("One-Class SVM model fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using One-Class SVM.
        
        Args:
            X: Data to predict on
            
        Returns:
            Binary array (1=anomaly, 0=normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert from (-1, 1) to (1, 0) where 1=anomaly
        return (predictions == -1).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using One-Class SVM.
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to positive scores (higher = more anomalous)
        return -scores


class LocalOutlierFactorDetector(BaseAnomalyDetector):
    """Local Outlier Factor anomaly detector."""
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        algorithm: str = 'auto',
        random_state: int = 42,
    ):
        """Initialize LOF detector.
        
        Args:
            n_neighbors: Number of neighbors to consider
            contamination: Expected proportion of anomalies
            algorithm: Algorithm for nearest neighbors
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.algorithm = algorithm
        
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            algorithm=algorithm,
            novelty=True,  # Enable novelty detection for predict method
        )
    
    def fit(self, X: np.ndarray) -> 'LocalOutlierFactorDetector':
        """Fit the LOF model.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Local Outlier Factor model...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info("LOF model fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using LOF.
        
        Args:
            X: Data to predict on
            
        Returns:
            Binary array (1=anomaly, 0=normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert from (-1, 1) to (1, 0) where 1=anomaly
        return (predictions == -1).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using LOF.
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to positive scores (higher = more anomalous)
        return -scores


class AnomalyDetector:
    """Unified interface for anomaly detection algorithms."""
    
    def __init__(self, algorithm: str = 'isolation_forest', **kwargs):
        """Initialize the anomaly detector.
        
        Args:
            algorithm: Algorithm to use ('isolation_forest', 'one_class_svm', 'lof')
            **kwargs: Additional parameters for the specific algorithm
        """
        self.algorithm = algorithm.lower()
        self.kwargs = kwargs
        
        if self.algorithm == 'isolation_forest':
            self.detector = IsolationForestDetector(**kwargs)
        elif self.algorithm == 'one_class_svm':
            self.detector = OneClassSVMDetector(**kwargs)
        elif self.algorithm == 'lof':
            self.detector = LocalOutlierFactorDetector(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logger.info(f"Initialized {self.algorithm} detector")
    
    def fit(self, X: np.ndarray) -> 'AnomalyDetector':
        """Fit the anomaly detection model.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        self.detector.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies in the data.
        
        Args:
            X: Data to predict on
            
        Returns:
            Binary array (1=anomaly, 0=normal)
        """
        return self.detector.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        return self.detector.decision_function(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.detector.save_model(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'AnomalyDetector':
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        detector = BaseAnomalyDetector.load_model(filepath)
        
        # Create wrapper instance
        instance = cls.__new__(cls)
        instance.detector = detector
        instance.algorithm = detector.__class__.__name__.lower().replace('detector', '')
        instance.kwargs = {}
        
        return instance
