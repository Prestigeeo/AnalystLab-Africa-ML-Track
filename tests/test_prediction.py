"""
Integration Tests: End-to-End Prediction Pipeline (Part D & E)
===============================================================
MENTOR NOTE FOR INTERNS:
Integration tests verify that individual components (Validation, Preprocessing,
Inference, Decision Logic) integrate seamlessly into the 7-stage pipeline:

Data -> Validation -> Preprocessing -> Feature Preparation -> Model -> Prediction -> Output
"""

import pytest
import pandas as pd
import numpy as np

from src.config import PATHS, SCHEMA
from src.models.predict import FinTrustPredictionPipeline
from src.validation.validator import DataValidationError


@pytest.fixture(scope="module")
def prediction_pipeline():
    """Provides an initialized prediction pipeline instance."""
    # Ensure artifacts exist before running tests
    assert PATHS.model_artifact_path.exists(), "Model artifact must exist before running integration tests."
    assert PATHS.preprocessor_artifact_path.exists(), "Preprocessor artifact must exist before running integration tests."
    return FinTrustPredictionPipeline(strict_validation=True)


@pytest.fixture
def sample_valid_payload():
    """Provides a valid transaction dictionary."""
    return {
        "Transaction_ID": "FT-T999999",
        "Customer_ID": "FT-C00888",
        "Transaction_DateTime": 46023.25,
        "Transaction_Type": "Transfer",
        "Amount_NGN": 85000.0,
        "Channel": "Mobile App",
        "Device_Type": "Android",
        "Location": "Lagos",
        "International_Transaction": "No",
        "Transaction_Status": "Successful"
    }


def test_pipeline_valid_single_prediction(prediction_pipeline, sample_valid_payload):
    """Verifies that a valid payload runs through all 7 stages and yields scored output."""
    result_df = prediction_pipeline.run_pipeline(sample_valid_payload, verbose=False)
    
    assert len(result_df) == 1
    assert "Predicted_Risk_Flag" in result_df.columns
    assert "Risk_Probability" in result_df.columns
    assert "Risk_Tier" in result_df.columns
    assert "Operational_Action" in result_df.columns
    
    prob = result_df.loc[0, "Risk_Probability"]
    assert 0.0 <= prob <= 1.0
    assert result_df.loc[0, "Predicted_Risk_Flag"] in ["Yes", "No"]
    assert result_df.loc[0, "Risk_Tier"] in ["Low", "Medium", "High"]


def test_pipeline_rejects_corrupted_data(prediction_pipeline, sample_valid_payload):
    """Verifies that corrupt inputs trigger validation errors in Stage 2 without reaching the model."""
    corrupted_payload = sample_valid_payload.copy()
    corrupted_payload["Amount_NGN"] = -100.0  # Invalid negative amount
    
    with pytest.raises(DataValidationError):
        prediction_pipeline.run_pipeline(corrupted_payload, verbose=False)


def test_pipeline_reproducibility(prediction_pipeline, sample_valid_payload):
    """Verifies determinism: identical inputs generate identical probability scores."""
    out1 = prediction_pipeline.run_pipeline(sample_valid_payload, verbose=False)
    out2 = prediction_pipeline.run_pipeline(sample_valid_payload, verbose=False)
    
    np.testing.assert_almost_equal(
        out1["Risk_Probability"].values,
        out2["Risk_Probability"].values,
        decimal=6
    )
