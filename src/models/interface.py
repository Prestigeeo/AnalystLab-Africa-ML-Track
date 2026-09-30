"""
Model Interface and Adapter Module for FinTrust ML Workflow (Week 3: Part C)
===========================================================================
MENTOR NOTE FOR INTERNS:
In enterprise Machine Learning Engineering, decoupling model training/implementation
from the serving and evaluation pipeline is crucial. 

Why Use an Adapter Pattern?
1. COLLABORATION: A Data Science team or intern colleague might build a model using
   LightGBM, XGBoost, CatBoost, PyTorch, or Scikit-Learn. By agreeing on a standard
   Model Adapter interface, the ML Engineering pipeline can execute inference without
   caring about the underlying framework.
2. STANDARDIZATION: Ensures every model provides:
   - predict(X)
   - predict_proba(X)
   - get_metadata() (version, author, hyperparameters, date, threshold)
   - get_feature_importances()
3. SEPARATION OF CONCERNS: Protects the production API from unexpected model API changes.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
import joblib
import logging

# Ensure project root is in sys.path when running file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("FinTrust.ModelInterface")


class BaseModelAdapter(ABC):
    """
    Abstract Base Class defining the contract for all FinTrust prediction models.
    Any internal model or externally contributed Data Science model must implement
    or be wrapped by this adapter.
    """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate binary predictions (0 = Legitimate, 1 = Risk Review).
        
        Args:
            X (np.ndarray): 2D preprocessed numeric feature matrix.
            
        Returns:
            np.ndarray: 1D array of binary integer predictions.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate risk probability distributions.
        
        Args:
            X (np.ndarray): 2D preprocessed numeric feature matrix.
            
        Returns:
            np.ndarray: 2D array where col 0 = P(Legitimate), col 1 = P(Risk Review).
        """
        pass

    @abstractmethod
    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract feature importance scores for model interpretability.
        
        Args:
            feature_names (Optional[List[str]]): Names of transformed columns.
            
        Returns:
            Dict[str, float]: Mapping of feature names to importance scores.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve standardized model governance metadata.
        
        Returns:
            Dict[str, Any]: Metadata dictionary (author, model_type, version, metrics, etc.).
        """
        pass

    @abstractmethod
    def save(self, filepath: Path) -> Path:
        """Serialize adapter and underlying model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: Path) -> "BaseModelAdapter":
        """Deserialize adapter from disk."""
        pass


class FinTrustModelAdapter(BaseModelAdapter):
    """
    Standard Model Adapter wrapping our internal Scikit-Learn models
    (RandomForest, GradientBoosting, LogisticRegression).
    """

    def __init__(
        self,
        estimator: Any,
        metadata: Optional[Dict[str, Any]] = None,
        classification_threshold: float = 0.35
    ):
        self.estimator = estimator
        self.classification_threshold = classification_threshold
        self.metadata = metadata or {
            "model_family": type(estimator).__name__,
            "author": "FinTrust MLE Track",
            "version": "1.0.0-week3",
            "framework": "scikit-learn",
            "classification_threshold": classification_threshold,
            "description": "Integration-Ready FinTrust Risk Classifier"
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Applies operational probability threshold instead of hardcoded 0.5."""
        if hasattr(self.estimator, "predict_proba"):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= self.classification_threshold).astype(int)
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        elif hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(X)
            # Sigmoid conversion
            p1 = 1.0 / (1.0 + np.exp(-df))
            return np.vstack([1.0 - p1, p1]).T
        else:
            preds = self.estimator.predict(X)
            return np.vstack([1.0 - preds, preds]).T

    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        if hasattr(self.estimator, "feature_importances_"):
            importances = self.estimator.feature_importances_
            if feature_names and len(feature_names) == len(importances):
                return dict(sorted(zip(feature_names, map(float, importances)), key=lambda x: x[1], reverse=True))
            return {f"feature_{i}": float(val) for i, val in enumerate(importances)}
        elif hasattr(self.estimator, "coef_"):
            coefs = np.abs(self.estimator.coef_[0])
            if feature_names and len(feature_names) == len(coefs):
                return dict(sorted(zip(feature_names, map(float, coefs)), key=lambda x: x[1], reverse=True))
            return {f"feature_{i}": float(val) for i, val in enumerate(coefs)}
        return {}

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def save(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved FinTrustModelAdapter to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "FinTrustModelAdapter":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model adapter file not found: {filepath}")
        adapter = joblib.load(filepath)
        if not isinstance(adapter, BaseModelAdapter):
            # If a raw scikit-learn estimator was saved directly, wrap it automatically
            logger.warning("Loaded object is a raw estimator, wrapping in FinTrustModelAdapter.")
            adapter = FinTrustModelAdapter(estimator=adapter)
        logger.info(f"Loaded FinTrustModelAdapter from {filepath}")
        return adapter


class ExternalModelAdapter(BaseModelAdapter):
    """
    Adapter specifically designed to integrate external models contributed by
    Data Science peers or third-party vendors without pipeline modification.
    
    Mentors Note:
    In Week 3 Part C, if an intern is provided with another intern's model,
    they can wrap it in ExternalModelAdapter. If no outside model is provided,
    this adapter documents the exact contract required for peer integration.
    """

    def __init__(
        self,
        external_model: Any,
        author_name: str = "External DS Contributor",
        model_name: str = "External Risk Model",
        version: str = "1.0.0",
        classification_threshold: float = 0.35
    ):
        self.model = external_model
        self.classification_threshold = classification_threshold
        self.metadata = {
            "model_family": type(external_model).__name__,
            "author": author_name,
            "model_name": model_name,
            "version": version,
            "is_external": True,
            "classification_threshold": classification_threshold,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= self.classification_threshold).astype(int)
        return np.array(self.model.predict(X)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.array(self.model.predict_proba(X))
        preds = np.array(self.model.predict(X))
        return np.vstack([1.0 - preds, preds]).T

    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if feature_names and len(feature_names) == len(importances):
                return dict(sorted(zip(feature_names, map(float, importances)), key=lambda x: x[1], reverse=True))
            return {f"feature_{i}": float(v) for i, v in enumerate(importances)}
        return {}

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def save(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "ExternalModelAdapter":
        filepath = Path(filepath)
        adapter = joblib.load(filepath)
        return adapter
