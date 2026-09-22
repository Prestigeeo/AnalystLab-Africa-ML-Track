"""
Data Validator Component for FinTrust ML Pipeline
==================================================
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
5. Categorical domain constraint checks
6. Numeric boundary and range checks
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
    Validates FinTrust transaction datasets against predefined business and statistical contracts.
    
    Attributes:
        schema (FeatureSchema): The expected schema contract.
        strict (bool): If True, validation failures raise DataValidationError or reject entire batch.
                       If False, invalid records are quarantined/flagged while allowing valid records through.
    """

    def __init__(self, schema: FeatureSchema = SCHEMA, strict: bool = True):
        self.schema = schema
        self.strict = strict

    def validate(
        self,
        df: pd.DataFrame,
        is_training: bool = False
    ) -> Tuple[bool, ValidationReport, pd.DataFrame]:
        """
        Execute full validation suite on an input DataFrame.
        
        Args:
            df (pd.DataFrame): Raw or ingested transaction data.
            is_training (bool): If True, requires the target column ('Risk_Review_Flag').
                                If False (inference mode), target column is optional.
                                
        Returns:
            Tuple[bool, ValidationReport, pd.DataFrame]:
                - bool: True if dataset passed validation, False otherwise.
                - ValidationReport: Detailed error and warning report.
                - pd.DataFrame: Cleaned/filtered data (or unchanged original if all valid).
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

        # -------------------------------------------------------------
        # CHECK 2: Column Schema (Missing Required & Unexpected Columns)
        # -------------------------------------------------------------
        required_cols = (
            self.schema.id_columns
            + [self.schema.datetime_column]
            + self.schema.numerical_features
            + self.schema.categorical_features
        )
        if is_training:
            required_cols.append(self.schema.target_column)

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

        # If mandatory columns are missing, we cannot proceed with remaining checks safely
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
        for num_col in self.schema.numerical_features:
            if num_col in df_clean.columns:
                # Attempt conversion to numeric; any non-numeric becomes NaN
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
        # Check critical identifiers (Transaction_ID, Customer_ID) - NULL not allowed
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

        # Check feature columns for nulls - recorded as warnings to be imputed downstream
        for col in self.schema.numerical_features + self.schema.categorical_features:
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
            "Transaction_Type": self.schema.valid_transaction_types,
            "Channel": self.schema.valid_channels,
            "Device_Type": self.schema.valid_device_types,
            "Location": self.schema.valid_locations,
            "International_Transaction": self.schema.valid_international,
            "Transaction_Status": self.schema.valid_transaction_statuses,
        }

        for cat_col, valid_set in category_rules.items():
            if cat_col in df_clean.columns:
                # Disregard nulls here as they are covered by missing-value checks
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
        if "Amount_NGN" in df_clean.columns:
            # Check for non-positive or negative amounts
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
