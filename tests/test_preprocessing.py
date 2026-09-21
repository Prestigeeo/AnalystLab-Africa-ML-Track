"""
Unit Tests: Preprocessing Pipeline (Part C & E)
================================================
MENTOR NOTE FOR INTERNS:
Preprocessing pipelines must satisfy three critical engineering properties:
1. Determinism: Identical inputs must yield identical numerical outputs.
2. Leakage Protection: Statistics learned on training must not be recomputed on test data.
3. Fault-Tolerance to Unseen Data: In production, new categorical values appear;
   the preprocessor must handle them without crashing or altering feature dimensionality.
"""

import pytest
import pandas as pd
import numpy as np

from src.config import SCHEMA
from src.preprocessing.pipeline import (
    TemporalFeatureExtractor,
    FinTrustDataPreprocessor,
    build_preprocessing_pipeline
)


@pytest.fixture
def sample_training_df():
    """Generates a small representative training dataframe."""
    return pd.DataFrame([
        {
            "Transaction_ID": "FT-T001",
            "Customer_ID": "FT-C001",
            "Transaction_DateTime": 46023.0,  # 2026-01-01 00:00:00 (Thursday)
            "Transaction_Type": "Transfer",
            "Amount_NGN": 5000.0,
            "Channel": "Mobile App",
            "Device_Type": "Android",
            "Location": "Lagos",
            "International_Transaction": "No",
            "Transaction_Status": "Successful"
        },
        {
            "Transaction_ID": "FT-T002",
            "Customer_ID": "FT-C002",
            "Transaction_DateTime": 46025.5,  # 2026-01-03 12:00:00 (Saturday)
            "Transaction_Type": "Card Purchase",
            "Amount_NGN": 15000.0,
            "Channel": "POS",
            "Device_Type": "POS Terminal",
            "Location": "Abuja",
            "International_Transaction": "Yes",
            "Transaction_Status": "Successful"
        },
        {
            "Transaction_ID": "FT-T003",
            "Customer_ID": "FT-C003",
            "Transaction_DateTime": 46026.75,  # 2026-01-04 18:00:00 (Sunday)
            "Transaction_Type": "Cash Withdrawal",
            "Amount_NGN": 20000.0,
            "Channel": "ATM",
            "Device_Type": "ATM Terminal",
            "Location": "Kano",
            "International_Transaction": "No",
            "Transaction_Status": "Failed"
        }
    ])


def test_temporal_feature_extraction(sample_training_df):
    """Verifies extraction of Transaction_Hour, DayOfWeek, and Is_Weekend."""
    extractor = TemporalFeatureExtractor()
    transformed = extractor.transform(sample_training_df)
    
    assert "Transaction_Hour" in transformed.columns
    assert "Transaction_DayOfWeek" in transformed.columns
    assert "Is_Weekend" in transformed.columns
    
    # Record 1: 00:00 (Thursday -> Day 3, Not weekend)
    assert transformed.loc[0, "Transaction_Hour"] == 0
    assert transformed.loc[0, "Is_Weekend"] == 0
    
    # Record 2: 12:00 (Saturday -> Day 5, Weekend)
    assert transformed.loc[1, "Transaction_Hour"] == 12
    assert transformed.loc[1, "Is_Weekend"] == 1


def test_unseen_categories_handling(sample_training_df):
    """
    Verifies that OneHotEncoder with handle_unknown='ignore' does NOT crash
    or change dimension when encountering an unseen category at test/inference time.
    """
    preprocessor = FinTrustDataPreprocessor(schema=SCHEMA)
    preprocessor.fit(sample_training_df)
    
    # Create test record with an unobserved category
    unseen_df = sample_training_df.iloc[[0]].copy()
    unseen_df.loc[0, "Channel"] = "VirtualReality"  # Never seen during fit!
    unseen_df.loc[0, "Location"] = "Atlantis"       # Never seen during fit!
    
    # Should transform cleanly without error
    X_out = preprocessor.transform(unseen_df)
    
    # Output dimensionality must match training dimensionality exactly
    train_transformed = preprocessor.transform(sample_training_df)
    assert X_out.shape[1] == train_transformed.shape[1]
    assert np.isnan(X_out).sum() == 0  # No NaNs generated


def test_missing_value_imputation(sample_training_df):
    """Verifies that missing numericals and categoricals are imputed cleanly."""
    preprocessor = FinTrustDataPreprocessor(schema=SCHEMA)
    preprocessor.fit(sample_training_df)
    
    # Test record with missing fields
    null_df = sample_training_df.iloc[[0]].copy()
    null_df.loc[0, "Amount_NGN"] = np.nan
    null_df.loc[0, "Device_Type"] = np.nan
    
    X_out = preprocessor.transform(null_df)
    assert not np.isnan(X_out).any(), "Imputer failed: NaNs remain in transformed matrix"
