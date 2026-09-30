"""
End-to-End Prediction Pipeline for FinTrust ML Workflow (Week 3: Integration-Ready)
==================================================================================
MENTOR NOTE FOR INTERNS:
In Week 3, we implement the complete, traceable 7-stage prediction workflow:

Data -> Validation -> Preprocessing -> Feature Preparation -> Model -> Prediction -> Output

Enterprise Highlights:
1. Feature Store / Customer Profile Lookup: If a raw transaction is provided with
   only a `Customer_ID`, the pipeline automatically joins customer profile features
   from the clean customer database cache.
2. Model Adapter Support: Directly interoperates with `BaseModelAdapter` and
   `FinTrustModelAdapter` (Part C).
3. Risk Tiers & Business Recommendations: Converts raw probabilities into Low,
   Medium, and High risk tiers with actionable operational decisions (Auto-Approve,
   OTP Verification, Immediate Hold).
4. Full Batch Processing: Supports single transactions, lists, and full DataFrames.
"""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path when running file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, SCHEMA, MODEL_CONFIG, FeatureSchema
from src.validation.validator import DataValidator, DataValidationError
from src.preprocessing.pipeline import FinTrustDataPreprocessor
from src.models.interface import BaseModelAdapter, FinTrustModelAdapter

logger = logging.getLogger("FinTrust.Prediction")


class FinTrustPredictionPipeline:
    """
    Executes the 7-stage prediction workflow for FinTrust transaction risk scoring.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
        customer_cache_path: Optional[Union[str, Path]] = None,
        strict_validation: bool = True
    ):
        self.model_path = Path(model_path or PATHS.model_artifact_path)
        self.preprocessor_path = Path(preprocessor_path or PATHS.preprocessor_artifact_path)
        self.customer_cache_path = Path(customer_cache_path or PATHS.customer_cache_path)
        self.strict_validation = strict_validation
        
        self.validator = DataValidator(schema=SCHEMA, strict=strict_validation)
        self.model: Optional[BaseModelAdapter] = None
        self.preprocessor: Optional[FinTrustDataPreprocessor] = None
        self._customer_cache: Optional[pd.DataFrame] = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads serialized model adapter, preprocessor, and customer feature store."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Please run training first (e.g. `python -m src.models.train`)."
            )
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found at {self.preprocessor_path}. "
                "Please run training first (e.g. `python -m src.models.train`)."
            )

        logger.info("Loading preprocessor and model adapter artifacts...")
        self.preprocessor = FinTrustDataPreprocessor.load(self.preprocessor_path)
        self.model = FinTrustModelAdapter.load(self.model_path)
        
        # Load customer profile cache for feature enrichment if available
        if self.customer_cache_path.exists():
            self._customer_cache = pd.read_csv(self.customer_cache_path)
            logger.info(f"Loaded customer feature store cache ({len(self._customer_cache)} profiles).")
        else:
            logger.warning(f"Customer cache not found at {self.customer_cache_path}. Auto-enrichment disabled.")

    def _enrich_with_customer_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If transaction records are missing customer attributes, joins them from the customer store.
        """
        df_out = df.copy()
        cust_cols = SCHEMA.customer_numerical_features + SCHEMA.customer_categorical_features
        missing_cust_cols = [c for c in cust_cols if c not in df_out.columns]

        if missing_cust_cols and self._customer_cache is not None and "Customer_ID" in df_out.columns:
            # Merge on Customer_ID
            cols_to_merge = ["Customer_ID"] + [c for c in cust_cols if c in self._customer_cache.columns]
            cust_subset = self._customer_cache[cols_to_merge].drop_duplicates(subset=["Customer_ID"])
            df_out = df_out.merge(cust_subset, on="Customer_ID", how="left")
            logger.info(f"Auto-enriched {len(df_out)} transaction(s) with customer demographics.")

        return df_out

    def run_pipeline(
        self,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Executes the complete 7-stage workflow:
        Stage 1: Data Ingestion
        Stage 2: Validation
        Stage 3: Preprocessing
        Stage 4: Feature Preparation
        Stage 5: Model Inference
        Stage 6: Prediction & Risk Tiers
        Stage 7: Output Packaging
        """
        if verbose:
            print("\n" + "="*70)
            print("🚀 EXECUTING FINTRUST 7-STAGE INTEGRATION-READY PREDICTION WORKFLOW")
            print("="*70)

        # -------------------------------------------------------------
        # STAGE 1: Data Ingestion & Auto-Enrichment
        # -------------------------------------------------------------
        if isinstance(input_data, dict):
            df_stage1 = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df_stage1 = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df_stage1 = input_data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected dict, list, or DataFrame.")

        df_enriched = self._enrich_with_customer_profiles(df_stage1)

        if verbose:
            print(f"[STAGE 1: DATA INGESTION & ENRICHMENT]")
            print(f"  -> Ingested {len(df_enriched)} transaction record(s).")
            print(f"  -> Total features available: {len(df_enriched.columns)}")

        # -------------------------------------------------------------
        # STAGE 2: Data Validation
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 2: DATA VALIDATION]")
        is_valid, report, df_stage2 = self.validator.validate(df_enriched, is_training=False, dataset_type="auto")
        
        if verbose:
            print(f"  -> Validation Passed: {is_valid}")
            print(f"  -> Action Taken: {report.action_taken}")
            print(f"  -> Errors Detected: {len(report.errors)} | Warnings: {len(report.warnings)}")

        if not is_valid and self.strict_validation:
            error_details = "; ".join([e.message for e in report.errors])
            raise DataValidationError(f"Validation rejected input: {error_details}", report)

        # -------------------------------------------------------------
        # STAGE 3 & 4: Preprocessing & Feature Preparation
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 3 & 4: PREPROCESSING & FEATURE PREPARATION]")
            print("  -> Applying temporal extraction (Hour, DayOfWeek, Is_Weekend)...")
            print("  -> Applying numerical scaling & one-hot encoding...")

        X_features = self.preprocessor.transform(df_stage2)
        if verbose:
            print(f"  -> Transformed feature matrix shape: {X_features.shape}")

        # -------------------------------------------------------------
        # STAGE 5: Model Inference via Model Adapter
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 5: MODEL INFERENCE (ADAPTER)]")
            
        prob_matrix = self.model.predict_proba(X_features)
        probabilities = prob_matrix[:, 1]
        
        if verbose:
            print(f"  -> Model Family: {self.model.get_metadata().get('model_family', 'Adapter')}")
            print(f"  -> Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")

        # -------------------------------------------------------------
        # STAGE 6: Prediction & Risk Tier Decision
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 6: PREDICTION & DECISION LOGIC]")
            print(f"  -> Operational Decision Threshold: {self.model.classification_threshold}")

        predictions = ["Yes" if p >= self.model.classification_threshold else "No" for p in probabilities]

        # Determine risk tiers and operational recommendations
        risk_tiers = []
        action_recommendations = []
        for p in probabilities:
            if p >= MODEL_CONFIG.high_risk_threshold:
                risk_tiers.append("High")
                action_recommendations.append("Immediate Review / Payment Hold")
            elif p >= MODEL_CONFIG.low_risk_threshold:
                risk_tiers.append("Medium")
                action_recommendations.append("Secondary Verification (SMS/OTP)")
            else:
                risk_tiers.append("Low")
                action_recommendations.append("Auto-Approve")

        # -------------------------------------------------------------
        # STAGE 7: Output Generation
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 7: OUTPUT GENERATION]")

        df_output = df_stage2.copy()
        df_output["Predicted_Risk_Flag"] = predictions
        df_output["Risk_Probability"] = np.round(probabilities, 4)
        df_output["Risk_Tier"] = risk_tiers
        df_output["Operational_Action"] = action_recommendations
        df_output["Model_Version"] = self.model.get_metadata().get("version", "2.0.0-week3")
        df_output["Scored_At"] = datetime.now(timezone.utc).isoformat()

        if verbose:
            print(f"  -> Output generated successfully ({len(df_output)} records scored).")
            print("="*70 + "\n")

        return df_output


def predict_single_transaction(transaction_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience helper to predict a single transaction and return clean dict."""
    pipeline = FinTrustPredictionPipeline()
    result_df = pipeline.run_pipeline(transaction_dict, verbose=False)
    return result_df.iloc[0].to_dict()


def predict_batch_dataset(input_csv_path: Union[str, Path], output_csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Convenience helper to score an entire CSV file."""
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df_raw = pd.read_csv(input_path)
    pipeline = FinTrustPredictionPipeline()
    df_scored = pipeline.run_pipeline(df_raw, verbose=True)

    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_scored.to_csv(out_p, index=False)
        logger.info(f"Saved scored batch to {out_p}")

    return df_scored
