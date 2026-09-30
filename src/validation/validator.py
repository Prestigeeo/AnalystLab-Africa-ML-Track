"""
Data Validator Component for FinTrust ML Pipeline (Week 3: Integration-Ready)
============================================================================
MENTOR NOTE FOR INTERNS:
In an ML pipeline, data validation serves as the first line of defense.
Never assume input data adheres to specification. Upstream schema changes,
database migration bugs, network dropouts, or malicious inputs can compromise
downstream model performance or cause unhandled system crashes.

This module implements a comprehensive DataValidator that executes 6 key checks:
1. Empty dataset detection
2. Column schema validation (missing required columns and unexpected columns)
3. Data type validation
4. Missing value detection and thresholding
5. Categorical domain constraint checks (Transactions + Customer Profile)
6. Numeric boundary and range checks (Amount, Age, Tenure, Digital Engagement)
"""

import sys
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path when running file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SCHEMA, FeatureSchema
from src.validation.schema import ValidationErrorItem, ValidationReport

# Configure module logger
logger = logging.getLogger("FinTrust.Validator")


class DataValidationError(Exception):
    """Custom exception raised when strict data validation fails."""
    def __init__(self, message: str, report: ValidationReport):
        super().__init__(message)
        self.report = report


class DataValidator:
    """
    Validates FinTrust transaction and customer datasets against predefined business contracts.
    
    Attributes:
        schema (FeatureSchema): The expected schema contract.
        strict (bool): If True, validation failures raise DataValidationError.
                       If False, returns report without raising.
    """

    def __init__(self, schema: FeatureSchema = SCHEMA, strict: bool = True):
        self.schema = schema
        self.strict = strict

    def validate(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
        dataset_type: str = "auto"
    ) -> Tuple[bool, ValidationReport, pd.DataFrame]:
        """
        Execute full validation suite on an input DataFrame.
        
        Args:
            df (pd.DataFrame): Raw, enriched, or inference transaction data.
            is_training (bool): If True, requires the target column ('Risk_Review_Flag').
            dataset_type (str): 'auto', 'enriched', 'transaction', or 'customer'.
                                If 'auto', detects based on columns present.
                                
        Returns:
            Tuple[bool, ValidationReport, pd.DataFrame]:
                - bool: True if dataset passed validation, False otherwise.
                - ValidationReport: Detailed error and warning report.
                - pd.DataFrame: Cleaned/filtered data.
        """
        errors: List[ValidationErrorItem] = []
        warnings: List[ValidationErrorItem] = []

        # -------------------------------------------------------------
        # CHECK 1: Empty Dataset Check
        # -------------------------------------------------------------
        if df is None or df.empty:
            errors.append(ValidationErrorItem(
                field="Dataset",
                error_type="EMPTY_DATASET",
                message="The provided dataset is empty (0 rows or None). Processing cannot continue.",
                severity="ERROR",
                invalid_rows_count=0
            ))
            report = ValidationReport(
                is_valid=False,
                total_records=0,
                errors=errors,
                warnings=warnings,
                action_taken="REJECT_ALL"
            )
            logger.error("Validation failed: Dataset is empty.")
            if self.strict:
                raise DataValidationError("Dataset is empty.", report)
            return False, report, df

        total_records = len(df)
        df_clean = df.copy()

        # Determine expected columns based on dataset type
        has_customer_cols = any(col in df_clean.columns for col in self.schema.customer_numerical_features + self.schema.customer_categorical_features)
        
        if dataset_type == "customer":
            expected_num = self.schema.customer_numerical_features
            expected_cat = self.schema.customer_categorical_features
            required_cols = ["Customer_ID"] + expected_num + expected_cat
        elif dataset_type == "transaction" or (dataset_type == "auto" and not has_customer_cols):
            expected_num = self.schema.transaction_numerical_features
            expected_cat = self.schema.transaction_categorical_features
            required_cols = self.schema.id_columns + [self.schema.datetime_column] + expected_num + expected_cat
        else:
            # Enriched dataset (Transactions + Customers)
            expected_num = self.schema.numerical_features
            expected_cat = self.schema.categorical_features
            required_cols = self.schema.id_columns + [self.schema.datetime_column] + expected_num + expected_cat

        if is_training:
            required_cols.append(self.schema.target_column)

        # -------------------------------------------------------------
        # CHECK 2: Column Schema (Missing Required & Unexpected Columns)
        # -------------------------------------------------------------
        present_cols = set(df_clean.columns)
        missing_cols = [col for col in required_cols if col not in present_cols]
        unexpected_cols = [col for col in present_cols if col not in required_cols and col != self.schema.target_column]

        if missing_cols:
            errors.append(ValidationErrorItem(
                field="Columns",
                error_type="MISSING_COLUMNS",
                message=f"Dataset is missing required columns: {missing_cols}",
                severity="ERROR",
                sample_invalid_values=missing_cols
            ))

        if unexpected_cols:
            warnings.append(ValidationErrorItem(
                field="Columns",
                error_type="UNEXPECTED_COLUMNS",
                message=f"Dataset contains unexpected extra columns: {unexpected_cols}. These will be ignored.",
                severity="WARNING",
                sample_invalid_values=unexpected_cols
            ))

        # If mandatory columns are missing, cannot proceed with remaining checks safely
        if missing_cols:
            report = ValidationReport(
                is_valid=False,
                total_records=total_records,
                errors=errors,
                warnings=warnings,
                action_taken="REJECT_ALL"
            )
            if self.strict:
                raise DataValidationError(f"Missing required columns: {missing_cols}", report)
            return False, report, df_clean

        # -------------------------------------------------------------
        # CHECK 3: Data Types & Numeric Integrity
        # -------------------------------------------------------------
        for num_col in expected_num:
            if num_col in df_clean.columns:
                converted = pd.to_numeric(df_clean[num_col], errors='coerce')
                type_mismatches = df_clean[num_col].notna() & converted.isna()
                mismatch_count = int(type_mismatches.sum())
                
                if mismatch_count > 0:
                    sample_vals = df_clean.loc[type_mismatches, num_col].head(5).tolist()
                    errors.append(ValidationErrorItem(
                        field=num_col,
                        error_type="INCORRECT_DATA_TYPE",
                        message=f"Column '{num_col}' contains {mismatch_count} non-numeric values.",
                        severity="ERROR",
                        invalid_rows_count=mismatch_count,
                        sample_invalid_values=sample_vals
                    ))

        # -------------------------------------------------------------
        # CHECK 4: Missing Values (Nulls)
        # -------------------------------------------------------------
        # Primary key checks - null not allowed
        for id_col in self.schema.id_columns:
            if id_col in df_clean.columns:
                null_ids = df_clean[id_col].isna() | (df_clean[id_col].astype(str).str.strip() == "")
                null_count = int(null_ids.sum())
                if null_count > 0:
                    errors.append(ValidationErrorItem(
                        field=id_col,
                        error_type="NULL_PRIMARY_KEY",
                        message=f"Column '{id_col}' contains {null_count} null or blank values. Primary keys must be complete.",
                        severity="ERROR",
                        invalid_rows_count=null_count
                    ))

        # Feature columns - recorded as warnings for imputation
        for col in expected_num + expected_cat:
            if col in df_clean.columns:
                null_count = int(df_clean[col].isna().sum())
                if null_count > 0:
                    pct = (null_count / total_records) * 100
                    warnings.append(ValidationErrorItem(
                        field=col,
                        error_type="MISSING_VALUES",
                        message=f"Column '{col}' has {null_count} ({pct:.2f}%) missing values. Downstream imputation will be applied.",
                        severity="WARNING",
                        invalid_rows_count=null_count
                    ))

        # -------------------------------------------------------------
        # CHECK 5: Unexpected Categories (Domain Constraints)
        # -------------------------------------------------------------
        category_rules = {
            # Transaction categories
            "Transaction_Type": self.schema.valid_transaction_types,
            "Channel": self.schema.valid_channels,
            "Device_Type": self.schema.valid_device_types,
            "Location": self.schema.valid_locations,
            "International_Transaction": self.schema.valid_international,
            "Transaction_Status": self.schema.valid_transaction_statuses,
            # Customer categories (Week 3)
            "Gender": self.schema.valid_genders,
            "Customer_Segment": self.schema.valid_customer_segments,
            "Account_Type": self.schema.valid_account_types,
            "Monthly_Income_Band": self.schema.valid_income_bands,
            "Preferred_Channel": self.schema.valid_preferred_channels,
            "Account_Status": self.schema.valid_account_statuses,
        }

        for cat_col, valid_set in category_rules.items():
            if cat_col in df_clean.columns:
                non_null_vals = df_clean[cat_col].dropna().astype(str)
                unexpected = non_null_vals[~non_null_vals.isin(valid_set)]
                if not unexpected.empty:
                    unexpected_count = len(unexpected)
                    unique_unexpected = unexpected.unique().tolist()
                    errors.append(ValidationErrorItem(
                        field=cat_col,
                        error_type="UNEXPECTED_CATEGORY",
                        message=f"Column '{cat_col}' contains unexpected categories: {unique_unexpected}",
                        severity="ERROR",
                        invalid_rows_count=unexpected_count,
                        sample_invalid_values=unique_unexpected[:5]
                    ))

        # -------------------------------------------------------------
        # CHECK 6: Value Ranges and Invalid Inputs
        # -------------------------------------------------------------
        # 1. Amount_NGN: must be positive and within reasonable max
        if "Amount_NGN" in df_clean.columns:
            numeric_amounts = pd.to_numeric(df_clean["Amount_NGN"], errors='coerce')
            invalid_amounts = (numeric_amounts <= 0) | (numeric_amounts > self.schema.max_amount_ngn)
            invalid_count = int(invalid_amounts.sum())
            if invalid_count > 0:
                sample_vals = numeric_amounts[invalid_amounts].head(5).tolist()
                errors.append(ValidationErrorItem(
                    field="Amount_NGN",
                    error_type="OUT_OF_RANGE",
                    message=f"Found {invalid_count} transactions with invalid Amount_NGN (<= 0 or > {self.schema.max_amount_ngn}).",
                    severity="ERROR",
                    invalid_rows_count=invalid_count,
                    sample_invalid_values=sample_vals
                ))

        # 2. Age: must be between 18 and 100
        if "Age" in df_clean.columns:
            numeric_age = pd.to_numeric(df_clean["Age"], errors='coerce')
            invalid_age = (numeric_age < self.schema.min_age) | (numeric_age > self.schema.max_age)
            invalid_count = int(invalid_age.sum())
            if invalid_count > 0:
                sample_vals = numeric_age[invalid_age].head(5).tolist()
                errors.append(ValidationErrorItem(
                    field="Age",
                    error_type="OUT_OF_RANGE",
                    message=f"Found {invalid_count} customer records with invalid Age (< {self.schema.min_age} or > {self.schema.max_age}).",
                    severity="ERROR",
                    invalid_rows_count=invalid_count,
                    sample_invalid_values=sample_vals
                ))

        # 3. Tenure_Months: must be non-negative
        if "Tenure_Months" in df_clean.columns:
            numeric_tenure = pd.to_numeric(df_clean["Tenure_Months"], errors='coerce')
            invalid_tenure = (numeric_tenure < self.schema.min_tenure_months)
            invalid_count = int(invalid_tenure.sum())
            if invalid_count > 0:
                sample_vals = numeric_tenure[invalid_tenure].head(5).tolist()
                errors.append(ValidationErrorItem(
                    field="Tenure_Months",
                    error_type="OUT_OF_RANGE",
                    message=f"Found {invalid_count} customer records with negative Tenure_Months.",
                    severity="ERROR",
                    invalid_rows_count=invalid_count,
                    sample_invalid_values=sample_vals
                ))

        # 4. Digital_Engagement_Score: must be between 0 and 100
        if "Digital_Engagement_Score" in df_clean.columns:
            numeric_score = pd.to_numeric(df_clean["Digital_Engagement_Score"], errors='coerce')
            invalid_score = (numeric_score < self.schema.min_digital_score) | (numeric_score > self.schema.max_digital_score)
            invalid_count = int(invalid_score.sum())
            if invalid_count > 0:
                sample_vals = numeric_score[invalid_score].head(5).tolist()
                errors.append(ValidationErrorItem(
                    field="Digital_Engagement_Score",
                    error_type="OUT_OF_RANGE",
                    message=f"Found {invalid_count} records with invalid Digital_Engagement_Score (must be 0-100).",
                    severity="ERROR",
                    invalid_rows_count=invalid_count,
                    sample_invalid_values=sample_vals
                ))

        # Determine validation verdict and pipeline action
        is_valid = len(errors) == 0
        if is_valid:
            action_taken = "PROCEED"
        elif self.strict:
            action_taken = "REJECT_ALL"
        else:
            action_taken = "QUARANTINE_RECORDS"

        report = ValidationReport(
            is_valid=is_valid,
            total_records=total_records,
            errors=errors,
            warnings=warnings,
            action_taken=action_taken
        )

        if not is_valid and self.strict:
            raise DataValidationError(
                f"Data validation failed with {len(errors)} critical error(s).",
                report=report
            )

        return is_valid, report, df_clean
