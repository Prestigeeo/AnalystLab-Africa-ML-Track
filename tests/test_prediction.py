"""
Integration Tests: End-to-End Prediction Pipeline (Week 3 - Part D & F)
========================================================================
MENTOR NOTE FOR INTERNS:
Integration tests verify that the 7-stage pipeline functions seamlessly:
Data -> Validation -> Preprocessing -> Feature Preparation -> Model -> Prediction -> Output

Tested Criteria:
- Valid input execution
- Customer profile auto-enrichment
- Prediction generation & risk tier assignment
- Structured output format
- Deterministic reproducibility
- Batch scoring capability
- Rejection of corrupted inputs
"""

import pytest
import pandas as pd
import numpy as np

from src.config import PATHS, SCHEMA, MODEL_CONFIG
from src.models.predict import FinTrustPredictionPipeline
from src.validation.validator import DataValidationError


@pytest.fixture(scope="module")
def prediction_pipeline():
    """Provides an initialized prediction pipeline instance."""
    assert PATHS.model_artifact_path.exists(), "Model artifact must exist before running tests."
    assert PATHS.preprocessor_artifact_path.exists(), "Preprocessor artifact must exist before running tests."
    return FinTrustPredictionPipeline(strict_validation=True)


@pytest.fixture
def sample_raw_transaction():
    """Provides a raw transaction dictionary that needs customer profile lookup."""
    return {
        "Transaction_ID": "TX_TEST_888",
        "Customer_ID": "FT-C00001",
        "Transaction_DateTime": 46023.25,
        "Transaction_Type": "Transfer",
        "Amount_NGN": 85000.0,
        "Channel": "Mobile App",
        "Device_Type": "Android",
        "Location": "Lagos",
        "International_Transaction": "No",
        "Transaction_Status": "Successful"
    }


# ----------------------------------------------------------------------
# 1. Prediction Generation & End-to-End Execution
# ----------------------------------------------------------------------
def test_end_to_end_valid_prediction(prediction_pipeline, sample_raw_transaction):
    """
    Verifies that a valid transaction flows through all 7 stages and produces prediction.
    """
    result_df = prediction_pipeline.run_pipeline(sample_raw_transaction, verbose=False)
    
    assert len(result_df) == 1
    assert result_df.loc[0, "Predicted_Risk_Flag"] in ["Yes", "No"]
    assert result_df.loc[0, "Risk_Tier"] in ["Low", "Medium", "High"]
    assert result_df.loc[0, "Operational_Action"] in [
        "Auto-Approve",
        "Secondary Verification (SMS/OTP)",
        "Immediate Review / Payment Hold"
    ]


# ----------------------------------------------------------------------
# 2. Output Format Contract
# ----------------------------------------------------------------------
def test_end_to_end_output_format(prediction_pipeline, sample_raw_transaction):
    """
    Verifies the schema of the generated output DataFrame.
    """
    result_df = prediction_pipeline.run_pipeline(sample_raw_transaction, verbose=False)
    
    expected_output_fields = [
        "Transaction_ID",
        "Customer_ID",
        "Predicted_Risk_Flag",
        "Risk_Probability",
        "Risk_Tier",
        "Operational_Action",
        "Model_Version",
        "Scored_At"
    ]
    for field in expected_output_fields:
        assert field in result_df.columns, f"Output missing expected field: {field}"
        
    prob = result_df.loc[0, "Risk_Probability"]
    assert 0.0 <= prob <= 1.0


# ----------------------------------------------------------------------
# 3. Customer Profile Auto-Enrichment
# ----------------------------------------------------------------------
def test_end_to_end_auto_enrichment(prediction_pipeline, sample_raw_transaction):
    """
    Verifies that customer demographics (Age, Customer_Segment) are automatically
    joined from the customer feature store cache when only raw transaction is provided.
    """
    result_df = prediction_pipeline.run_pipeline(sample_raw_transaction, verbose=False)
    
    # Customer profile attributes should have been enriched
    assert "Age" in result_df.columns
    assert "Customer_Segment" in result_df.columns
    assert not pd.isna(result_df.loc[0, "Age"])


# ----------------------------------------------------------------------
# 4. Reproducibility Test
# ----------------------------------------------------------------------
def test_end_to_end_reproducibility(prediction_pipeline, sample_raw_transaction):
    """
    Verifies reproducibility: Scoring the same transaction twice yields identical results.
    """
    out1 = prediction_pipeline.run_pipeline(sample_raw_transaction, verbose=False)
    out2 = prediction_pipeline.run_pipeline(sample_raw_transaction, verbose=False)
    
    np.testing.assert_almost_equal(
        out1["Risk_Probability"].values,
        out2["Risk_Probability"].values,
        decimal=6
    )
    assert out1["Predicted_Risk_Flag"].iloc[0] == out2["Predicted_Risk_Flag"].iloc[0]


# ----------------------------------------------------------------------
# 5. Batch Scoring Capability
# ----------------------------------------------------------------------
def test_end_to_end_batch_prediction(prediction_pipeline, sample_raw_transaction):
    """
    Verifies that multiple transactions can be scored concurrently in a single batch call.
    """
    batch_list = [
        sample_raw_transaction,
        {**sample_raw_transaction, "Transaction_ID": "TX_TEST_889", "Amount_NGN": 500000.0, "Location": "Abuja"},
        {**sample_raw_transaction, "Transaction_ID": "TX_TEST_890", "Amount_NGN": 1200.0, "Channel": "USSD"}
    ]
    
    result_df = prediction_pipeline.run_pipeline(batch_list, verbose=False)
    assert len(result_df) == 3
    assert len(result_df["Predicted_Risk_Flag"]) == 3


# ----------------------------------------------------------------------
# 6. Rejection of Corrupted Data
# ----------------------------------------------------------------------
def test_end_to_end_rejection_of_corrupt_data(prediction_pipeline, sample_raw_transaction):
    """
    Verifies that invalid inputs (e.g. negative amount) are caught by validator
    and rejected before reaching preprocessor or model.
    """
    corrupt = sample_raw_transaction.copy()
    corrupt["Amount_NGN"] = -999.0
    
    with pytest.raises(DataValidationError):
        prediction_pipeline.run_pipeline(corrupt, verbose=False)
