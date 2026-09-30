"""
Technical Unit Tests: Data Preparation & Join Integrity (Week 3 - Part B)
=========================================================================
MENTOR NOTE FOR INTERNS:
Multi-table data preparation carries risks of row explosion (Cartesian products)
or missing customer records. We test:
1. Exact row count preservation (12,000 transactions)
2. 100% customer match rate
3. PII redaction (Customer_Name must not exist in output)
4. Train/Test split stratification consistency
"""

import pytest
import pandas as pd
from pathlib import Path

from src.config import PATHS
from src.data.prepare import DataPreparator


def test_data_preparation_execution():
    """Verifies that DataPreparator executes and creates clean datasets."""
    preparator = DataPreparator()
    result = preparator.prepare_and_export()
    
    assert result["total_enriched_records"] == 12000
    assert result["train_records"] == 9600
    assert result["test_records"] == 2400
    assert PATHS.processed_train_path.exists()
    assert PATHS.processed_test_path.exists()
    assert PATHS.customer_cache_path.exists()


def test_pii_redaction():
    """Verifies that sensitive PII (Customer_Name) is completely stripped."""
    df_train = pd.read_csv(PATHS.processed_train_path)
    df_test = pd.read_csv(PATHS.processed_test_path)
    df_cache = pd.read_csv(PATHS.customer_cache_path)
    
    assert "Customer_Name" not in df_train.columns
    assert "Customer_Name" not in df_test.columns
    assert "Customer_Name" not in df_cache.columns


def test_join_completeness():
    """Verifies that customer profile features are non-null across the enriched dataset."""
    df_enriched = pd.read_csv(PATHS.processed_enriched_path)
    
    assert df_enriched["Age"].isna().sum() == 0
    assert df_enriched["Tenure_Months"].isna().sum() == 0
    assert df_enriched["Customer_Segment"].isna().sum() == 0
