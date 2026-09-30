"""
Technical Unit Tests: Data Validation (Week 3 - Part D)
======================================================
MENTOR NOTE FOR INTERNS:
In ML Engineering, automated test suites must test:
1. Valid inputs (Happy path)
2. Missing values & imputation thresholds
3. Unexpected categories (Domain boundaries)
4. Invalid data types (Type safety)
5. Empty inputs (Zero-row safeguards)
6. Value range & boundary violations (Business invariants)
7. Enriched multi-table schema validation
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
        "Customer_ID": "CUST-0001",
        "Transaction_DateTime": 46023.05,
        "Transaction_Type": "Transfer",
        "Amount_NGN": 15000.50,
        "Channel": "Mobile App",
        "Device_Type": "Android",
        "Location": "Lagos",
        "International_Transaction": "No",
        "Transaction_Status": "Successful",
    }])


@pytest.fixture
def valid_enriched_transaction():
    """Provides a valid enriched transaction record including customer profile."""
    return pd.DataFrame([{
        "Transaction_ID": "FT-T000001",
        "Customer_ID": "CUST-0001",
        "Transaction_DateTime": 46023.05,
        "Transaction_Type": "Transfer",
        "Amount_NGN": 15000.50,
        "Channel": "Mobile App",
        "Device_Type": "Android",
        "Location": "Lagos",
        "International_Transaction": "No",
        "Transaction_Status": "Successful",
        "Age": 35.0,
        "Tenure_Months": 24.0,
        "Digital_Engagement_Score": 78.5,
        "Gender": "Female",
        "Customer_Segment": "Everyday",
        "Account_Type": "Savings",
        "Monthly_Income_Band": "250k-499k",
        "Preferred_Channel": "Mobile App",
        "Account_Status": "Active"
    }])


# ----------------------------------------------------------------------
# 1. Valid Input Test
# ----------------------------------------------------------------------
def test_valid_input(valid_sample_transaction):
    """
    Test Case: Valid input.
    Expected: Validation passes (is_valid == True, action == 'PROCEED', 0 errors).
    """
    validator = DataValidator(schema=SCHEMA, strict=True)
    is_valid, report, df_clean = validator.validate(valid_sample_transaction, dataset_type="transaction")
    
    assert is_valid is True
    assert len(report.errors) == 0
    assert report.action_taken == "PROCEED"
    assert len(df_clean) == 1


# ----------------------------------------------------------------------
# 2. Missing Values Test
# ----------------------------------------------------------------------
def test_missing_values_warning_and_error(valid_sample_transaction):
    """
    Test Case: Missing values.
    Expected: Missing feature values generate warnings for imputation; missing IDs raise error.
    """
    df_missing_feat = valid_sample_transaction.copy()
    df_missing_feat.loc[0, "Device_Type"] = np.nan
    
    validator_lenient = DataValidator(schema=SCHEMA, strict=False)
    is_valid, report, _ = validator_lenient.validate(df_missing_feat, dataset_type="transaction")
    
    assert is_valid is True
    assert any(w.error_type == "MISSING_VALUES" for w in report.warnings)

    # Missing primary key
    df_missing_id = valid_sample_transaction.copy()
    df_missing_id.loc[0, "Transaction_ID"] = np.nan
    validator_strict = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError) as exc_info:
        validator_strict.validate(df_missing_id, dataset_type="transaction")
    assert any(e.error_type == "NULL_PRIMARY_KEY" for e in exc_info.value.report.errors)


# ----------------------------------------------------------------------
# 3. Unexpected Category Test
# ----------------------------------------------------------------------
def test_unexpected_category(valid_sample_transaction):
    """
    Test Case: Unexpected category.
    Expected: Rejects unrecognized categories (e.g. Channel='Telepathy').
    """
    df_bad_cat = valid_sample_transaction.copy()
    df_bad_cat.loc[0, "Channel"] = "Telepathy"
    
    validator = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate(df_bad_cat, dataset_type="transaction")
        
    errors = exc_info.value.report.errors
    assert any(e.error_type == "UNEXPECTED_CATEGORY" for e in errors)


def test_unexpected_customer_category(valid_enriched_transaction):
    """
    Test Case: Unexpected customer category (Week 3 customer profile validation).
    Expected: Rejects invalid Customer_Segment or Account_Status.
    """
    df_bad_cust = valid_enriched_transaction.copy()
    df_bad_cust.loc[0, "Customer_Segment"] = "VIP_Alien"
    
    validator = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate(df_bad_cust, dataset_type="enriched")
    assert any(e.error_type == "UNEXPECTED_CATEGORY" for e in exc_info.value.report.errors)


# ----------------------------------------------------------------------
# 4. Incorrect Data Type Test
# ----------------------------------------------------------------------
def test_incorrect_data_type(valid_sample_transaction):
    """
    Test Case: Incorrect data type.
    Expected: Fails validation when numeric Amount_NGN contains text.
    """
    df_bad_type = valid_sample_transaction.astype({"Amount_NGN": object}).copy()
    df_bad_type.loc[0, "Amount_NGN"] = "fifty_thousand_naira"
    
    validator = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate(df_bad_type, dataset_type="transaction")
        
    assert any(e.error_type == "INCORRECT_DATA_TYPE" for e in exc_info.value.report.errors)


# ----------------------------------------------------------------------
# 5. Empty Input Dataset Test
# ----------------------------------------------------------------------
def test_empty_dataset():
    """
    Test Case: Empty input dataset.
    Expected: Rejects 0-row dataframe with EMPTY_DATASET error.
    """
    empty_df = pd.DataFrame()
    validator = DataValidator(schema=SCHEMA, strict=True)
    
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate(empty_df)
    assert any(e.error_type == "EMPTY_DATASET" for e in exc_info.value.report.errors)


# ----------------------------------------------------------------------
# 6. Numeric Boundary Violations Test
# ----------------------------------------------------------------------
def test_numeric_boundary_violations(valid_enriched_transaction):
    """
    Test Case: Boundary checks (Negative amount, impossible Age).
    """
    # Negative amount
    df_bad_amt = valid_enriched_transaction.copy()
    df_bad_amt.loc[0, "Amount_NGN"] = -500.0
    validator = DataValidator(schema=SCHEMA, strict=True)
    with pytest.raises(DataValidationError):
        validator.validate(df_bad_amt, dataset_type="enriched")

    # Impossible Age (< 18)
    df_bad_age = valid_enriched_transaction.copy()
    df_bad_age.loc[0, "Age"] = 12.0
    with pytest.raises(DataValidationError):
        validator.validate(df_bad_age, dataset_type="enriched")
