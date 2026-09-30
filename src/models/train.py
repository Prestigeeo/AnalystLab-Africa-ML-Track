"""
Model Training and Comparison Script for FinTrust ML Workflow (Week 3)
======================================================================
MENTOR NOTE FOR INTERNS:
In Week 3, we train our fraud risk model on the ENRICHED dataset combining:
- Transaction behavioral signals (amount, hour, day, type, channel, location)
- Customer profile demographics (age, tenure, digital score, income band, segment)

Key ML Engineering Concepts Taught Here:
1. Enriched Feature Space: Adding customer signals increases model discrimination.
2. Model Comparison: We compare a Baseline Logistic Regression vs. Random Forest.
3. Model Adapter Integration: We wrap the final estimator in `FinTrustModelAdapter`
   to ensure contract adherence (Part C).
4. Full Audit Metadata: We save a complete `model_metadata.json` for governance.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path when running file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
    brier_score_loss
)

from src.config import PATHS, SCHEMA, MODEL_CONFIG
from src.data.prepare import DataPreparator
from src.validation.validator import DataValidator
from src.preprocessing.pipeline import FinTrustDataPreprocessor
from src.models.interface import FinTrustModelAdapter

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FinTrust.Training")


def load_enriched_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test partitions. If not present, executes DataPreparator automatically.
    """
    if not PATHS.processed_train_path.exists() or not PATHS.processed_test_path.exists():
        logger.info("Processed train/test files not found. Running DataPreparator...")
        preparator = DataPreparator()
        preparator.prepare_and_export()

    logger.info(f"Loading train dataset from {PATHS.processed_train_path}...")
    df_train = pd.read_csv(PATHS.processed_train_path)
    logger.info(f"Loading test dataset from {PATHS.processed_test_path}...")
    df_test = pd.read_csv(PATHS.processed_test_path)
    return df_train, df_test


def evaluate_model_performance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = MODEL_CONFIG.classification_threshold
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics on hold-out data."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)

    preds = (probs >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_test, probs))
    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    brier = float(brier_score_loss(y_test, probs))
    cm = confusion_matrix(y_test, preds).tolist()

    return {
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier, 4),
        "decision_threshold": threshold,
        "confusion_matrix": cm,
    }


def train_model(save_artifacts: bool = True) -> Tuple[FinTrustModelAdapter, FinTrustDataPreprocessor, Dict[str, Any]]:
    """
    Executes the complete Week 3 training pipeline:
    1. Ingest prepared enriched data.
    2. Validate training records.
    3. Fit preprocessing pipeline.
    4. Train & compare baseline vs tuned Random Forest.
    5. Wrap in FinTrustModelAdapter.
    6. Persist artifacts & metadata.
    """
    # -------------------------------------------------------------
    # STAGE 1: Data Ingestion
    # -------------------------------------------------------------
    df_train, df_test = load_enriched_datasets()
    logger.info(f"Train samples: {len(df_train)} | Test samples: {len(df_test)}")

    # -------------------------------------------------------------
    # STAGE 2: Data Validation
    # -------------------------------------------------------------
    logger.info("Validating train and test datasets...")
    validator = DataValidator(schema=SCHEMA, strict=False)
    _, train_report, df_train = validator.validate(df_train, is_training=True, dataset_type="enriched")
    _, test_report, df_test = validator.validate(df_test, is_training=True, dataset_type="enriched")
    
    if not train_report.is_valid and train_report.action_taken == "REJECT_ALL":
        raise RuntimeError("Validation failed on training data.")

    # -------------------------------------------------------------
    # STAGE 3: Feature & Target Extraction
    # -------------------------------------------------------------
    if df_train[SCHEMA.target_column].dtype == object or isinstance(df_train[SCHEMA.target_column].iloc[0], str):
        y_train = df_train[SCHEMA.target_column].map(SCHEMA.target_mapping).fillna(0).astype(int)
        y_test = df_test[SCHEMA.target_column].map(SCHEMA.target_mapping).fillna(0).astype(int)
    else:
        y_train = df_train[SCHEMA.target_column].astype(int)
        y_test = df_test[SCHEMA.target_column].astype(int)

    X_train_raw = df_train.drop(columns=[SCHEMA.target_column])
    X_test_raw = df_test.drop(columns=[SCHEMA.target_column])

    # -------------------------------------------------------------
    # STAGE 4: Preprocessing & Feature Engineering
    # -------------------------------------------------------------
    logger.info("Fitting preprocessing pipeline on enriched X_train...")
    preprocessor = FinTrustDataPreprocessor(schema=SCHEMA)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.feature_names_out_ or [f"feature_{i}" for i in range(X_train.shape[1])]
    logger.info(f"Transformed feature matrix dimension: {X_train.shape[1]} columns.")

    # -------------------------------------------------------------
    # STAGE 5: Model Comparison (Baseline vs Production Candidate)
    # -------------------------------------------------------------
    logger.info("Training Baseline Model (Logistic Regression)...")
    baseline_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=MODEL_CONFIG.random_state)
    baseline_clf.fit(X_train, y_train)
    baseline_metrics = evaluate_model_performance(baseline_clf, X_test, y_test)
    logger.info(f"Baseline Logistic Regression ROC-AUC: {baseline_metrics['roc_auc']:.4f} | Recall: {baseline_metrics['recall']:.4f}")

    logger.info("Training Production Model (Random Forest Classifier)...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=MODEL_CONFIG.random_state,
        n_jobs=1
    )
    rf_clf.fit(X_train, y_train)
    rf_metrics = evaluate_model_performance(rf_clf, X_test, y_test)
    logger.info(f"Random Forest ROC-AUC: {rf_metrics['roc_auc']:.4f} | Recall: {rf_metrics['recall']:.4f}")

    # -------------------------------------------------------------
    # STAGE 6: Model Adapter Packaging (Part C)
    # -------------------------------------------------------------
    metadata = {
        "model_name": "FinTrust Enriched Random Forest Risk Classifier",
        "model_family": "RandomForestClassifier",
        "version": "2.0.0-week3",
        "author": "FinTrust MLE Track",
        "dataset_version": "enriched_v1_customer_joined",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": len(feature_names),
        "classification_threshold": MODEL_CONFIG.classification_threshold,
        "metrics": rf_metrics,
        "baseline_comparison": {
            "baseline_model": "LogisticRegression",
            "baseline_roc_auc": baseline_metrics["roc_auc"],
            "baseline_f1": baseline_metrics["f1_score"],
            "improvement_roc_auc": round(rf_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4)
        }
    }

    adapter = FinTrustModelAdapter(
        estimator=rf_clf,
        metadata=metadata,
        classification_threshold=MODEL_CONFIG.classification_threshold
    )

    # -------------------------------------------------------------
    # STAGE 7: Summary & Artifact Persistence
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("FINTRUST ML WEEK 3: MODEL TRAINING & EVALUATION REPORT")
    print("="*60)
    print(f"Dataset:              Enriched Transactions + Customer Profiles")
    print(f"Total Features:       {len(feature_names)}")
    print(f"Production Model:     RandomForest (100 trees, max_depth=12)")
    print(f"ROC-AUC Score:        {rf_metrics['roc_auc']:.4f}  (Baseline: {baseline_metrics['roc_auc']:.4f})")
    print(f"Precision:            {rf_metrics['precision']:.4f}")
    print(f"Recall:               {rf_metrics['recall']:.4f}")
    print(f"F1-Score:             {rf_metrics['f1_score']:.4f}")
    print(f"Brier Score Loss:     {rf_metrics['brier_score']:.4f}")
    print(f"Decision Threshold:   {rf_metrics['decision_threshold']}")
    print(f"Confusion Matrix [TN, FP / FN, TP]:\n{np.array(rf_metrics['confusion_matrix'])}")
    print("="*60 + "\n")

    if save_artifacts:
        logger.info("Persisting artifacts to disk...")
        preprocessor.save(PATHS.preprocessor_artifact_path)
        adapter.save(PATHS.model_artifact_path)

        with open(PATHS.metrics_path, "w") as f:
            json.dump(rf_metrics, f, indent=4)

        with open(PATHS.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"All artifacts saved successfully to {PATHS.ARTIFACTS_DIR if hasattr(PATHS, 'ARTIFACTS_DIR') else PATHS.preprocessor_artifact_path.parent}")

    return adapter, preprocessor, rf_metrics


if __name__ == "__main__":
    train_model()
