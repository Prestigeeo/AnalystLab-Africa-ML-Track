"""
End-to-End Prediction Pipeline for FinTrust ML Workflow
========================================================
MENTOR NOTE FOR INTERNS:
In Part D of the FinTrust project, we must implement an explicit, traceable
7-stage workflow:

Data -> Validation -> Preprocessing -> Feature Preparation -> Model -> Prediction -> Output

Why is this architectural pattern crucial in ML Engineering?
1. Separation of Concerns: Validation is separated from math; preprocessing is separated from inference.
2. Traceability & Debuggability: If an unexpected prediction occurs in production,
   we can inspect the exact intermediate data state at every transition point.
3. Reliability: Downstream model code never has to defend itself against missing
   columns or malformed strings because upstream stages guarantee invariants.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Union, List, Optional
import pandas as pd
import numpy as np
import joblib

from src.config import PATHS, SCHEMA, MODEL_CONFIG, FeatureSchema
from src.validation.validator import DataValidator, DataValidationError
from src.preprocessing.pipeline import FinTrustDataPreprocessor

logger = logging.getLogger("FinTrust.Prediction")


class FinTrustPredictionPipeline:
    """
    Executes the 7-stage prediction workflow for FinTrust transaction risk scoring.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        strict_validation: bool = True
    ):
        self.model_path = model_path or PATHS.model_artifact_path
        self.preprocessor_path = preprocessor_path or PATHS.preprocessor_artifact_path
        self.strict_validation = strict_validation
        
        self.validator = DataValidator(schema=SCHEMA, strict=strict_validation)
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads serialized model and preprocessor artifacts."""
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

        logger.info("Loading preprocessor and model artifacts...")
        self.preprocessor = FinTrustDataPreprocessor.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        logger.info("Artifacts successfully loaded into memory.")

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
        Stage 5: Model
        Stage 6: Prediction
        Stage 7: Output
        
        Args:
            input_data: Single transaction dict, list of transaction dicts, or DataFrame.
            verbose: If True, prints intermediate diagnostic telemetry at each stage.
            
        Returns:
            pd.DataFrame: Augmented DataFrame with predictions, probabilities, and risk tiers.
        """
        if verbose:
            print("\n" + "="*70)
            print("🚀 EXECUTING FINTRUST 7-STAGE PREDICTION WORKFLOW")
            print("="*70)

        # -------------------------------------------------------------
        # STAGE 1: Data Ingestion
        # -------------------------------------------------------------
        if isinstance(input_data, dict):
            df_stage1 = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df_stage1 = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df_stage1 = input_data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected dict, list, or DataFrame.")

        if verbose:
            print(f"[STAGE 1: DATA INGESTION]")
            print(f"  -> Ingested {len(df_stage1)} transaction record(s).")
            print(f"  -> Ingested columns: {list(df_stage1.columns)}")

        # -------------------------------------------------------------
        # STAGE 2: Data Validation
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 2: DATA VALIDATION]")
        is_valid, report, df_stage2 = self.validator.validate(df_stage1, is_training=False)
        
        if verbose:
            print(f"  -> Validation Passed: {is_valid}")
            print(f"  -> Action Taken: {report.action_taken}")
            print(f"  -> Errors Detected: {len(report.errors)} | Warnings: {len(report.warnings)}")

        if not is_valid and self.strict_validation:
            error_details = "; ".join([e.message for e in report.errors])
            raise DataValidationError(f"Validation rejected input: {error_details}", report)

        # -------------------------------------------------------------
        # STAGE 3: Preprocessing (Temporal Engineering & Imputation)
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 3: PREPROCESSING]")
            print("  -> Applying temporal feature extraction (Hour, DayOfWeek, Is_Weekend)...")
            print("  -> Imputing missing values with training medians/modes...")

        # -------------------------------------------------------------
        # STAGE 4: Feature Preparation (Scaling & Categorical Encoding)
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 4: FEATURE PREPARATION]")
        
        X_features = self.preprocessor.transform(df_stage2)
        if verbose:
            print(f"  -> Generated model-ready feature matrix.")
            print(f"  -> Feature matrix shape: {X_features.shape} (Rows x One-Hot Features)")

        # -------------------------------------------------------------
        # STAGE 5: Model Evaluation
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 5: MODEL INFERENCE]")
        probabilities = self.model.predict_proba(X_features)[:, 1]
        if verbose:
            print(f"  -> Evaluated Random Forest Classifier.")
            print(f"  -> Generated raw risk probabilities in range [{probabilities.min():.3f}, {probabilities.max():.3f}].")

        # -------------------------------------------------------------
        # STAGE 6: Prediction & Risk Tier Decision
        # -------------------------------------------------------------
        if verbose:
            print(f"\n[STAGE 6: PREDICTION & DECISION LOGIC]")
            print(f"  -> Applying operational decision threshold: {MODEL_CONFIG.classification_threshold}")

        threshold = MODEL_CONFIG.classification_threshold
        predictions = ["Yes" if p >= threshold else "No" for p in probabilities]

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
            print("  -> Assembling structured response payload...")

        df_output = df_stage2.copy()
        df_output["Predicted_Risk_Flag"] = predictions
        df_output["Risk_Probability"] = np.round(probabilities, 4)
        df_output["Risk_Tier"] = risk_tiers
        df_output["Operational_Action"] = action_recommendations
        df_output["Scored_At"] = datetime.now(timezone.utc).isoformat()

        if verbose:
            print(f"  -> Enriched output complete. Total rows: {len(df_output)}")
            print("="*70 + "\n")

        return df_output


def predict_single_transaction(transaction_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience helper to predict a single transaction and return clean dict."""
    pipeline = FinTrustPredictionPipeline()
    result_df = pipeline.run_pipeline(transaction_dict, verbose=False)
    return result_df.iloc[0].to_dict()
