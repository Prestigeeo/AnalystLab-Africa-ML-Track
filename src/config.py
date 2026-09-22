"""
Configuration Module for FinTrust ML Pipeline
==============================================
In production Machine Learning systems, hardcoding column names, file paths,
hyperparameters, and business rules across multiple scripts leads to bugs,
silent failures, and configuration drift.

We centralize all project configurations here in `config.py` using Python standard
dataclasses and pathlib.Path for cross-platform compatibility (macOS/Linux/Windows).
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set


# Base project paths (dynamically resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DOCS_DIR = BASE_DIR / "docs"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PathConfig:
    """Paths to raw data files, processed data files, and saved model artifacts."""
    raw_customer_path: Path = RAW_DATA_DIR / "FinTrust_Customer_Data.xlsx"
    raw_transaction_path: Path = RAW_DATA_DIR / "FinTrust_Transaction_Data.xlsx"
    processed_train_path: Path = PROCESSED_DATA_DIR / "fintrust_train_processed.csv"
    processed_test_path: Path = PROCESSED_DATA_DIR / "fintrust_test_processed.csv"
    preprocessor_artifact_path: Path = ARTIFACTS_DIR / "preprocessor.joblib"
    model_artifact_path: Path = ARTIFACTS_DIR / "model.joblib"
    metrics_path: Path = ARTIFACTS_DIR / "metrics.json"


@dataclass(frozen=True)
class FeatureSchema:
    """
    Data contract defining expected columns, types, and categorical levels.
    This acts as the single source of truth for Validation, Preprocessing, and Inference.
    """
    # Identifiers (needed for tracking/auditing, but excluded from model training)
    id_columns: List[str] = field(default_factory=lambda: [
        "Transaction_ID",
        "Customer_ID",
    ])

    # Temporal feature (used for temporal feature engineering)
    datetime_column: str = "Transaction_DateTime"

    # Target variable for fraud/risk review classification
    target_column: str = "Risk_Review_Flag"
    target_mapping: Dict[str, int] = field(default_factory=lambda: {
        "No": 0,
        "Yes": 1
    })

    # Numerical features directly from transaction data
    numerical_features: List[str] = field(default_factory=lambda: [
        "Amount_NGN",
    ])

    # Engineered temporal numerical features
    engineered_numerical_features: List[str] = field(default_factory=lambda: [
        "Transaction_Hour",
        "Transaction_DayOfWeek",
        "Is_Weekend",
    ])

    # Categorical features in transaction data
    categorical_features: List[str] = field(default_factory=lambda: [
        "Transaction_Type",
        "Channel",
        "Device_Type",
        "Location",
        "International_Transaction",
        "Transaction_Status",
    ])

    # Allowed categories for strict domain validation
    valid_transaction_types: Set[str] = field(default_factory=lambda: {
        "Transfer", "Card Purchase", "Bill Payment", "Cash Withdrawal", "Deposit", "Airtime/Data"
    })
    valid_channels: Set[str] = field(default_factory=lambda: {
        "Mobile App", "POS", "Web", "ATM", "USSD"
    })
    valid_device_types: Set[str] = field(default_factory=lambda: {
        "Android", "iOS", "POS Terminal", "Web Browser", "ATM Terminal"
    })
    valid_locations: Set[str] = field(default_factory=lambda: {
        "Lagos", "Abuja", "Port Harcourt", "Kano", "Ibadan", "Enugu", "Kaduna", "Benin City"
    })
    valid_international: Set[str] = field(default_factory=lambda: {
        "Yes", "No"
    })
    valid_transaction_statuses: Set[str] = field(default_factory=lambda: {
        "Successful", "Failed", "Reversed", "Pending"
    })

    # Numeric boundary rules
    min_amount_ngn: float = 0.01
    max_amount_ngn: float = 50_000_000.0  # Reasonable upper boundary for single transaction


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and operational decision thresholds."""
    test_size: float = 0.2
    random_state: int = 42
    
    # In fraud detection, false negatives (missing actual fraud) are usually much
    # more costly than false positives (flagging a safe transaction for manual review).
    # Setting an operational threshold allows tuning precision vs. recall tradeoff.
    classification_threshold: float = 0.35
    
    # Risk tiers for downstream banking ops
    low_risk_threshold: float = 0.30
    high_risk_threshold: float = 0.70


# Global config instances
PATHS = PathConfig()
SCHEMA = FeatureSchema()
MODEL_CONFIG = ModelConfig()
