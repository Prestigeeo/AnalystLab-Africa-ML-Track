"""
Configuration Module for FinTrust ML Pipeline (Week 3: Integration-Ready)
========================================================================
MENTOR NOTE FOR INTERNS:
In Week 3, we transition from transaction-only modeling to an INTEGRATION-READY
multi-table banking architecture. We centralize:
1. File paths for raw, intermediate, and enriched processed data.
2. Unified Feature Schema supporting both transaction attributes AND customer master attributes.
3. Domain boundary rules and valid categorical levels for rigorous validation.
4. Operational decision thresholds for fraud risk triage.
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
    """Paths to raw data files, processed datasets, and serialized model artifacts."""
    raw_customer_path: Path = RAW_DATA_DIR / "FinTrust_Customer_Data.xlsx"
    raw_transaction_path: Path = RAW_DATA_DIR / "FinTrust_Transaction_Data.xlsx"
    
    # Week 3 Enriched Data Paths
    processed_enriched_path: Path = PROCESSED_DATA_DIR / "fintrust_enriched.csv"
    processed_train_path: Path = PROCESSED_DATA_DIR / "fintrust_train.csv"
    processed_test_path: Path = PROCESSED_DATA_DIR / "fintrust_test.csv"
    customer_cache_path: Path = PROCESSED_DATA_DIR / "customers_cleaned.csv"
    
    # Model Artifacts
    preprocessor_artifact_path: Path = ARTIFACTS_DIR / "preprocessor.joblib"
    model_artifact_path: Path = ARTIFACTS_DIR / "model.joblib"
    metrics_path: Path = ARTIFACTS_DIR / "metrics.json"
    metadata_path: Path = ARTIFACTS_DIR / "model_metadata.json"


@dataclass(frozen=True)
class FeatureSchema:
    """
    Unified Data Contract for FinTrust Transactions and Customer Profiles.
    Serves as the single source of truth across Data Preparation, Validation,
    Preprocessing, and Serving.
    """
    # Identifiers
    id_columns: List[str] = field(default_factory=lambda: [
        "Transaction_ID",
        "Customer_ID",
    ])

    # Temporal feature
    datetime_column: str = "Transaction_DateTime"

    # Target variable for fraud / risk review classification
    target_column: str = "Risk_Review_Flag"
    target_mapping: Dict[str, int] = field(default_factory=lambda: {
        "No": 0,
        "Yes": 1
    })

    # Transaction-level numerical features
    transaction_numerical_features: List[str] = field(default_factory=lambda: [
        "Amount_NGN",
    ])

    # Engineered temporal numerical features
    engineered_numerical_features: List[str] = field(default_factory=lambda: [
        "Transaction_Hour",
        "Transaction_DayOfWeek",
        "Is_Weekend",
    ])

    # Customer profile numerical features (Week 3 Enrichment)
    customer_numerical_features: List[str] = field(default_factory=lambda: [
        "Age",
        "Tenure_Months",
        "Digital_Engagement_Score",
    ])

    # Transaction-level categorical features
    transaction_categorical_features: List[str] = field(default_factory=lambda: [
        "Transaction_Type",
        "Channel",
        "Device_Type",
        "Location",
        "International_Transaction",
        "Transaction_Status",
    ])

    # Customer profile categorical features (Week 3 Enrichment)
    customer_categorical_features: List[str] = field(default_factory=lambda: [
        "Gender",
        "Customer_Segment",
        "Account_Type",
        "Monthly_Income_Band",
        "Preferred_Channel",
        "Account_Status",
    ])

    # Full combined feature lists
    @property
    def numerical_features(self) -> List[str]:
        return self.transaction_numerical_features + self.customer_numerical_features

    @property
    def categorical_features(self) -> List[str]:
        return self.transaction_categorical_features + self.customer_categorical_features

    # Allowed transaction categorical domains
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

    # Allowed customer profile categorical domains (Week 3 Enrichment)
    valid_genders: Set[str] = field(default_factory=lambda: {
        "Male", "Female", "Prefer not to say"
    })
    valid_customer_segments: Set[str] = field(default_factory=lambda: {
        "Everyday", "Premium", "Student", "SME"
    })
    valid_account_types: Set[str] = field(default_factory=lambda: {
        "Savings", "Current", "Premium"
    })
    valid_income_bands: Set[str] = field(default_factory=lambda: {
        "Below 100k", "100k-249k", "250k-499k", "500k-999k", "1m+"
    })
    valid_preferred_channels: Set[str] = field(default_factory=lambda: {
        "Mobile App", "Web", "USSD"
    })
    valid_account_statuses: Set[str] = field(default_factory=lambda: {
        "Active", "Dormant", "Restricted"
    })

    # Numeric boundary rules
    min_amount_ngn: float = 0.01
    max_amount_ngn: float = 50_000_000.0
    min_age: float = 18.0
    max_age: float = 100.0
    min_digital_score: float = 0.0
    max_digital_score: float = 100.0
    min_tenure_months: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and operational decision thresholds."""
    test_size: float = 0.2
    random_state: int = 42
    
    # Operational classification threshold tuned for fraud recall
    classification_threshold: float = 0.35
    
    # Banking risk triage tiers
    low_risk_threshold: float = 0.30
    high_risk_threshold: float = 0.70


# Global config instances
PATHS = PathConfig()
SCHEMA = FeatureSchema()
MODEL_CONFIG = ModelConfig()
