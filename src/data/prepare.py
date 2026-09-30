"""
Data Preparation & Multi-Table Relational Enrichment (Week 3)
============================================================
MENTOR NOTE FOR INTERNS:
In Week 2, our model only scored transactions in isolation.
In real FinTech systems (like FinTrust), a ₦400,000 transfer is normal for a
wealthy corporate client, but highly anomalous for a student with an account
opened last week.

This module resolves the single-table weakness by:
1. Loading raw customer master data (1,500 records) and raw transactions (12,000 records).
2. Redacting Personally Identifiable Information (PII) like `Customer_Name` for NDPR/GDPR compliance.
3. Performing a relational left join on `Customer_ID`.
4. Exporting versioned, reproducible datasets to `data/processed/`:
   - `fintrust_enriched.csv`
   - `fintrust_train.csv`
   - `fintrust_test.csv`
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, SCHEMA, MODEL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FinTrust.DataPrepare")


class DataPreparator:
    """
    Handles ingestion, PII redaction, relational joining, and train/test export
    for FinTrust customer and transaction datasets.
    """
    def __init__(self, paths=PATHS, schema=SCHEMA, config=MODEL_CONFIG):
        self.paths = paths
        self.schema = schema
        self.config = config

    def load_raw_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads raw customer and transaction Excel files."""
        logger.info(f"Loading raw transactions from {self.paths.raw_transaction_path}...")
        df_txn = pd.read_excel(self.paths.raw_transaction_path)
        
        logger.info(f"Loading raw customer data from {self.paths.raw_customer_path}...")
        df_cust = pd.read_excel(self.paths.raw_customer_path)
        
        logger.info(f"Loaded {len(df_txn):,} transactions and {len(df_cust):,} customer profiles.")
        return df_txn, df_cust

    def clean_customer_data(self, df_cust: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans customer dataset and redacts PII (Customer_Name).
        """
        df = df_cust.copy()
        
        # Redact PII (Customer_Name)
        if "Customer_Name" in df.columns:
            logger.info("Redacting PII column 'Customer_Name' for data privacy compliance.")
            df = df.drop(columns=["Customer_Name"])
            
        # Clean Customer_ID
        df["Customer_ID"] = df["Customer_ID"].astype(str).str.strip()
        
        # Save cleaned customer reference cache
        df.to_csv(self.paths.customer_cache_path, index=False)
        logger.info(f"Saved cleaned customer cache to {self.paths.customer_cache_path}")
        return df

    def enrich_transactions(self, df_txn: pd.DataFrame, df_cust: pd.DataFrame) -> pd.DataFrame:
        """
        Performs relational left join of transactions with customer profiles on Customer_ID.
        """
        df_txn_clean = df_txn.copy()
        df_txn_clean["Customer_ID"] = df_txn_clean["Customer_ID"].astype(str).str.strip()
        
        # Execute left join
        logger.info("Executing relational left join on Customer_ID...")
        df_enriched = pd.merge(
            df_txn_clean,
            df_cust,
            on="Customer_ID",
            how="left",
            suffixes=("", "_cust")
        )
        
        # Verify row integrity
        if len(df_enriched) != len(df_txn):
            raise ValueError(f"Join integrity error: Expected {len(df_txn)} rows, got {len(df_enriched)}")
            
        match_rate = df_enriched["Account_Type"].notna().mean() * 100
        logger.info(f"Relational merge complete. Total rows: {len(df_enriched):,} | Customer match rate: {match_rate:.2f}%")
        return df_enriched

    def prepare_and_export(self) -> Dict[str, Any]:
        """
        Executes full data preparation workflow and exports versioned CSV files.
        """
        df_txn, df_cust = self.load_raw_datasets()
        df_cust_clean = self.clean_customer_data(df_cust)
        df_enriched = self.enrich_transactions(df_txn, df_cust_clean)
        
        # Save master enriched dataset
        self.paths.processed_enriched_path.parent.mkdir(parents=True, exist_ok=True)
        df_enriched.to_csv(self.paths.processed_enriched_path, index=False)
        logger.info(f"Exported enriched dataset to {self.paths.processed_enriched_path}")
        
        # Stratified train/test split on target variable
        if self.schema.target_column in df_enriched.columns:
            y = df_enriched[self.schema.target_column].map(self.schema.target_mapping).astype(int)
            train_df, test_df = train_test_split(
                df_enriched,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
            train_df.to_csv(self.paths.processed_train_path, index=False)
            test_df.to_csv(self.paths.processed_test_path, index=False)
            logger.info(f"Exported train split ({len(train_df):,} rows) to {self.paths.processed_train_path}")
            logger.info(f"Exported test split ({len(test_df):,} rows) to {self.paths.processed_test_path}")
        else:
            train_df, test_df = df_enriched, pd.DataFrame()
            
        return {
            "total_enriched_records": len(df_enriched),
            "train_records": len(train_df),
            "test_records": len(test_df),
            "columns": list(df_enriched.columns),
            "enriched_file": str(self.paths.processed_enriched_path),
            "train_file": str(self.paths.processed_train_path),
            "test_file": str(self.paths.processed_test_path),
        }


def run_data_preparation() -> Dict[str, Any]:
    """CLI / Function entry point for data preparation."""
    preparator = DataPreparator()
    return preparator.prepare_and_export()


if __name__ == "__main__":
    run_data_preparation()
