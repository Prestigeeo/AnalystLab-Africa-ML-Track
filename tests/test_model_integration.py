"""
Technical Unit Tests: Model Integration & Adapter (Week 3 - Part C & D)
======================================================================
MENTOR NOTE FOR INTERNS:
Part C & D require testing:
- Model loading
- Prediction generation
- Output format
- Model Adapter contract adherence (BaseModelAdapter / ExternalModelAdapter)
"""

import pytest
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import PATHS, MODEL_CONFIG
from src.models.interface import BaseModelAdapter, FinTrustModelAdapter, ExternalModelAdapter


@pytest.fixture
def sample_feature_matrix():
    """Generates synthetic 2D feature matrix (10 samples, 60 features)."""
    np.random.seed(42)
    return np.random.randn(10, 60)


# ----------------------------------------------------------------------
# 1. Model Loading Test
# ----------------------------------------------------------------------
def test_model_loading():
    """
    Test: Model loading.
    Expected: Loads saved model artifact and verifies it is an instance of BaseModelAdapter.
    """
    assert PATHS.model_artifact_path.exists(), "Model artifact must exist before running tests."
    adapter = FinTrustModelAdapter.load(PATHS.model_artifact_path)
    
    assert isinstance(adapter, BaseModelAdapter)
    assert hasattr(adapter, "predict")
    assert hasattr(adapter, "predict_proba")
    assert hasattr(adapter, "get_metadata")


# ----------------------------------------------------------------------
# 2. Prediction Generation Test
# ----------------------------------------------------------------------
def test_prediction_generation(sample_feature_matrix):
    """
    Test: Prediction generation.
    Expected: Generates binary predictions array matching number of samples.
    """
    adapter = FinTrustModelAdapter.load(PATHS.model_artifact_path)
    preds = adapter.predict(sample_feature_matrix)
    
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(sample_feature_matrix)
    assert set(preds).issubset({0, 1})


# ----------------------------------------------------------------------
# 3. Output Format Test
# ----------------------------------------------------------------------
def test_output_format(sample_feature_matrix):
    """
    Test: Output format.
    Expected: predict_proba returns 2D matrix of shape (N, 2) where rows sum to 1.0.
    """
    adapter = FinTrustModelAdapter.load(PATHS.model_artifact_path)
    probs = adapter.predict_proba(sample_feature_matrix)
    
    assert probs.ndim == 2
    assert probs.shape == (len(sample_feature_matrix), 2)
    # Probabilities in each row must sum to 1.0 (within float tolerance)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
    # Probability values must lie in [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


# ----------------------------------------------------------------------
# 4. External Model Adapter Compatibility Test (Part C)
# ----------------------------------------------------------------------
def test_external_model_adapter_compatibility(sample_feature_matrix):
    """
    Test: Adapter pattern allows seamless integration of external DS models.
    """
    # Create a mock external scikit-learn or custom model
    dummy_model = LogisticRegression()
    dummy_model.fit(sample_feature_matrix, np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
    
    external_adapter = ExternalModelAdapter(
        external_model=dummy_model,
        author_name="DS Intern Jane Doe",
        model_name="Experimental Logistic Classifier",
        version="0.9.1"
    )
    
    assert isinstance(external_adapter, BaseModelAdapter)
    preds = external_adapter.predict(sample_feature_matrix)
    probs = external_adapter.predict_proba(sample_feature_matrix)
    meta = external_adapter.get_metadata()
    
    assert len(preds) == 10
    assert probs.shape == (10, 2)
    assert meta["is_external"] is True
    assert meta["author"] == "DS Intern Jane Doe"


# ----------------------------------------------------------------------
# 5. Metadata Governance Audit Test
# ----------------------------------------------------------------------
def test_model_metadata_audit():
    """
    Test: Model metadata dictionary contains required governance fields.
    """
    adapter = FinTrustModelAdapter.load(PATHS.model_artifact_path)
    metadata = adapter.get_metadata()
    
    required_keys = ["model_family", "version", "classification_threshold"]
    for key in required_keys:
        assert key in metadata, f"Metadata missing required key '{key}'"
