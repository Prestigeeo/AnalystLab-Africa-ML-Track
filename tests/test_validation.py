"""
Technical Unit Tests: Data Validation (Part E)
==============================================
MENTOR NOTE FOR INTERNS:
In ML Engineering, tests must be deterministic, automated, and cover both
happy paths (valid data) and edge/failure cases (bad data).

This test suite verifies the 5 mandatory scenarios specified in Part E:
1. Valid input
2. Missing values
3. Unexpected category
4. Incorrect data type
5. Empty input
Plus additional boundary constraints (negative amounts, missing columns).
"""

import pytest
import pandas as pd
import numpy as np

from src.config import SCHEMA
from src.validation.validator import DataValidator, DataValidationError


@pytest.fixture
def valid_sample_transaction():
    """Provides a single valid transaction record adhering to schema."""
    return pd.DataFrame([{
        "Transaction_ID": "FT-T000001",
        "Customer_ID": "FT-C00123",
        "Transaction_DateTime": 46023.05,
        "Transaction_Type": "Transfer",
        "Amount_NGN": 15000.50,
        "Channel": "Mobile App",
        "Device_Type": "Android",
        "Location": "Lagos",
        "International_Transaction": "No",
        "Transaction_Status": "Successful",
    }])


# ----------------------------------------------------------------------
# TEST 1: Valid Input (Happy Path)
# ----------------------------------------------------------------------
def test_valid_input(valid_sample_transaction):
    """
    Test Case 1: Valid input.
    Expected: Validation passes without critical errors (is_valid == True, action == 'PROCEED').
    """
    validator = DataValidator(schema=SCHEMA, strict=True)
    is_valid, report, df_clean = validator.validate(valid_sample_transaction)
    
    assert is_valid is True
    assert len(report.errors) == 0
    assert report.action_taken == "PROCEED"
    assert len(df_clean) == 1


# ----------------------------------------------------------------------
# TEST 2: Missing Values
# ----------------------------------------------------------------------
def test_missing_values_warning_and_error(valid_sample_transaction):
    """
    Test Case 2: Missing values.
    Expected:
      - Null feature values (Device_Type, Location) generate WARNINGS for downstream imputation.
      - Null primary key (Transaction_ID) raises a critical validation error.
    """
    # Sub-case A: Missing non-critical feature (should issue warning, allow downstream imputation)
    df_missing_feature = valid_sample_transaction.copy()
    df_missing_feature.loc[0, "Device_Type"] = np.nan
    
    validator_lenient = DataValidator(schema=SCHEMA, strict=False)
    is_valid, report, _ = validator_lenient.validate(df_missing_feature)
    
    assert is_valid is True
    assert any(w.error_type == "MISSING_VALUES" for w in report.warnings)

    # Sub-case B: Missing primary key ID (critical error)
    df_missing_id = valid_sample_transaction.copy()
    df_missing_id.loc[0, "Transaction_ID"] = np.nan
    
    validator_strict = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError) as exc_info:
        validator_strict.validate(df_missing_id)
        
    assert any(e.error_type == "NULL_PRIMARY_KEY" for e in exc_info.value.report.errors)


# ----------------------------------------------------------------------
# TEST 3: Unexpected Category
# ----------------------------------------------------------------------
def test_unexpected_category(valid_sample_transaction):
    """
    Test Case 3: Unexpected category.
    Expected: Rejects unrecognized category (e.g. Channel='Telepathy') with UNEXPECTED_CATEGORY.
    """
    df_bad_cat = valid_sample_transaction.copy()
    df_bad_cat.loc[0, "Channel"] = "Telepathy"  # Not in allowed channels
    
    validator = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate(df_bad_cat)
        
    errors = exc_info.value.report.errors
    assert any(e.error_type == "UNEXPECTED_CATEGORY" for e in errors)
    assert any("Channel" in e.field for e in errors)


# ----------------------------------------------------------------------
# TEST 4: Incorrect Data Type
# ----------------------------------------------------------------------
def test_incorrect_data_type(valid_sample_transaction):
    """
    Test Case 4: Incorrect data type.
    Expected: Fails validation when numeric Amount_NGN is passed as a non-numeric string.
    """
    df_bad_type = valid_sample_transaction.astype({"Amount_NGN": object}).copy()
    df_bad_type.loc[0, "Amount_NGN"] = "twenty_thousand_naira"  # Non-numeric string
    
    validator = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate(df_bad_type)
        
    errors = exc_info.value.report.errors
    assert any(e.error_type == "INCORRECT_DATA_TYPE" for e in errors)


# ----------------------------------------------------------------------
# TEST 5: Empty Input Dataset
# ----------------------------------------------------------------------
def test_empty_dataset():
    """
    Test Case 5: Empty input dataset.
    Expected: Rejects empty dataframe (0 rows) with EMPTY_DATASET error.
    """
    empty_df = pd.DataFrame()
    validator = DataValidator(schema=SCHEMA, strict=True)
    
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate(empty_df)
        
    errors = exc_info.value.report.errors
    assert any(e.error_type == "EMPTY_DATASET" for e in errors)


# ----------------------------------------------------------------------
# TEST 6: Boundary Constraint (Negative / Zero Amount)
# ----------------------------------------------------------------------
def test_negative_amount_input(valid_sample_transaction):
    """
    Test Case 6: Numeric boundary check (Negative transaction amount).
    Expected: Rejects transaction with Amount_NGN <= 0.
    """
    df_negative = valid_sample_transaction.copy()
    df_negative.loc[0, "Amount_NGN"] = -2500.00
    
    validator = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate(df_negative)
        
    errors = exc_info.value.report.errors
    assert any(e.error_type == "OUT_OF_RANGE" for e in errors)
