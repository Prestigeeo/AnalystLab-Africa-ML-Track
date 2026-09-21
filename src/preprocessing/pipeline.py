"""
Preprocessing Pipeline Module for FinTrust ML Workflow
======================================================
MENTOR NOTE FOR INTERNS:
One of the most dangerous, insidious bugs in Machine Learning is DATA LEAKAGE.
Data leakage occurs when information from outside the training dataset (like test
set distribution, future timestamps, or global means/medians) is used to create
the model.

Key Principles Demonstrated Here:
1. FIT ONLY ON TRAIN:
   We compute imputer statistics (medians, modes) and scaling parameters (mean, std)
   strictly on `X_train`. We then call `transform()` on `X_test` and production inputs.
2. HANDLING UNSEEN CATEGORIES:
   In production, a user might submit a channel or location not present in training.
   If using naive pandas `get_dummies`, your feature dimension will change or crash!
   Using `OneHotEncoder(handle_unknown='ignore')` safely zeroes out unseen categories.
3. REPRODUCIBILITY:
   By wrapping transformers inside a Scikit-Learn `Pipeline` and serializing with `joblib`,
   the exact same mathematical transformations execute deterministically anywhere.
"""

import logging
from typing import Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import SCHEMA, PATHS, FeatureSchema

logger = logging.getLogger("FinTrust.Preprocessing")


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to parse transaction timestamps and
    extract cyclical and behavioral temporal signals:
    - Hour of the day (fraud patterns vary between 2 AM and 2 PM)
    - Day of week (weekday vs. weekend spending)
    - Is_Weekend binary indicator
    """
    def __init__(self, datetime_col: str = SCHEMA.datetime_column):
        self.datetime_col = datetime_col
        # Excel epoch reference (days since 1899-12-30)
        self.excel_epoch = pd.Timestamp("1899-12-30")

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless transformer: no parameters need to be estimated from data
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        if self.datetime_col in X_out.columns:
            raw_dates = X_out[self.datetime_col]
            
            # Check if numeric (Excel serial date) or string/datetime
            if pd.api.types.is_numeric_dtype(raw_dates):
                converted_dates = self.excel_epoch + pd.to_timedelta(raw_dates, unit="D")
            else:
                # Fallback to general date parser
                converted_dates = pd.to_datetime(raw_dates, errors="coerce")
            
            # Extract engineered features
            X_out["Transaction_Hour"] = converted_dates.dt.hour.fillna(12).astype(int)
            X_out["Transaction_DayOfWeek"] = converted_dates.dt.dayofweek.fillna(0).astype(int)
            X_out["Is_Weekend"] = X_out["Transaction_DayOfWeek"].isin([5, 6]).astype(int)
        else:
            # If datetime column is missing, provide safe defaults
            X_out["Transaction_Hour"] = 12
            X_out["Transaction_DayOfWeek"] = 0
            X_out["Is_Weekend"] = 0

        return X_out


def build_preprocessing_pipeline(schema: FeatureSchema = SCHEMA) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer integrating numerical and categorical sub-pipelines.
    
    Returns:
        ColumnTransformer: Configured preprocessor ready for fit/transform.
    """
    # 1. Numerical Pipeline: Median Imputation -> Standard Scaling
    all_numerical = schema.numerical_features + schema.engineered_numerical_features
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 2. Categorical Pipeline: Constant Imputation -> One-Hot Encoding
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine into a unified ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, all_numerical),
            ("cat", categorical_pipeline, schema.categorical_features),
        ],
        remainder="drop",  # Drops ID columns like Customer_ID and Transaction_ID
        verbose_feature_names_out=False
    )
    
    return preprocessor


class FinTrustDataPreprocessor:
    """
    High-level preprocessor managing feature extraction, fitting, transformation,
    and artifact persistence.
    """
    def __init__(self, schema: FeatureSchema = SCHEMA):
        self.schema = schema
        self.temporal_extractor = TemporalFeatureExtractor(schema.datetime_column)
        self.column_preprocessor = build_preprocessing_pipeline(schema)
        self.is_fitted = False
        self.feature_names_out_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y=None) -> "FinTrustDataPreprocessor":
        """
        Fit the preprocessor on the training data.
        
        Args:
            X (pd.DataFrame): Training feature DataFrame.
        """
        logger.info("Fitting FinTrust Data Preprocessor...")
        # Step 1: Temporal feature engineering
        X_engineered = self.temporal_extractor.fit_transform(X)
        
        # Step 2: Fit numerical & categorical transformers
        self.column_preprocessor.fit(X_engineered)
        self.is_fitted = True
        
        # Extract resulting feature names for interpretability
        try:
            self.feature_names_out_ = list(self.column_preprocessor.get_feature_names_out())
            logger.info(f"Fitted preprocessor. Total transformed features: {len(self.feature_names_out_)}")
        except Exception:
            self.feature_names_out_ = None
            
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform raw/validated feature data using parameters learned during fit.
        
        Args:
            X (pd.DataFrame): Input DataFrame to transform.
            
        Returns:
            np.ndarray: Transformed numeric feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform can be called.")
            
        X_engineered = self.temporal_extractor.transform(X)
        return self.column_preprocessor.transform(X_engineered)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Convenience method to fit and transform in one step on training data."""
        return self.fit(X, y).transform(X)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serialize preprocessor to disk."""
        target_path = filepath or PATHS.preprocessor_artifact_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        logger.info(f"Saved preprocessor artifact to {target_path}")
        return target_path

    @staticmethod
    def load(filepath: Optional[Path] = None) -> "FinTrustDataPreprocessor":
        """Load preprocessor from disk."""
        target_path = filepath or PATHS.preprocessor_artifact_path
        if not target_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {target_path}")
        preprocessor = joblib.load(target_path)
        logger.info(f"Loaded preprocessor artifact from {target_path}")
        return preprocessor
