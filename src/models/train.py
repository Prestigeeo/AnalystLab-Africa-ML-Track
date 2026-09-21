"""
Model Training Script for FinTrust ML Workflow
===============================================
MENTOR NOTE FOR INTERNS:
In this module, we train a machine learning model to detect transactions
that should be flagged for risk review (binary classification: 'Yes' / 'No').

Key ML Engineering Concepts Taught Here:
1. Stratified Splitting: Because risk/fraud is imbalanced (~19.6% Yes),
   we use StratifiedShuffleSplit or train_test_split(stratify=y) so the
   proportions of fraud are identical in both train and test splits.
2. Class Imbalance Handling: We use `class_weight='balanced'` in RandomForest
   so the algorithm penalizes false negatives appropriately.
3. Metric Selection: Accuracy is misleading for imbalanced problems!
   We track ROC-AUC, Precision, Recall, and F1-Score.
4. Artifact Preservation: Both preprocessor and model are serialized
   so that inference environments reproduce exact training transformations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.config import PATHS, SCHEMA, MODEL_CONFIG
from src.validation.validator import DataValidator
from src.preprocessing.pipeline import FinTrustDataPreprocessor

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FinTrust.Training")


def load_raw_data(data_path: Path = PATHS.raw_transaction_path) -> pd.DataFrame:
    """Load raw transaction dataset from Excel file."""
    logger.info(f"Loading raw dataset from {data_path}...")
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
    df = pd.read_excel(data_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns.")
    return df


def train_model(
    data_path: Path = PATHS.raw_transaction_path,
    save_artifacts: bool = True
) -> Tuple[RandomForestClassifier, FinTrustDataPreprocessor, Dict[str, Any]]:
    """
    Executes the full training workflow:
    Load -> Validate -> Split -> Fit Preprocessor -> Fit Model -> Evaluate -> Save
    
    Returns:
        Tuple of (fitted_model, fitted_preprocessor, metrics_dict)
    """
    # -------------------------------------------------------------
    # STAGE 1: Data Ingestion
    # -------------------------------------------------------------
    df = load_raw_data(data_path)

    # -------------------------------------------------------------
    # STAGE 2: Data Validation (Training Mode)
    # -------------------------------------------------------------
    logger.info("Running pre-training data validation...")
    validator = DataValidator(schema=SCHEMA, strict=False)
    is_valid, report, df = validator.validate(df, is_training=True)
    report.print_report()
    
    if not is_valid and report.action_taken == "REJECT_ALL":
        raise RuntimeError("Validation failed. Training aborted due to critical data corruption.")

    # -------------------------------------------------------------
    # STAGE 3: Target Encoding & Data Splitting
    # -------------------------------------------------------------
    logger.info("Encoding target variable and preparing train/test split...")
    # Drop rows where target is NaN (if any)
    df = df.dropna(subset=[SCHEMA.target_column]).copy()
    
    # Map 'Yes' -> 1, 'No' -> 0
    y = df[SCHEMA.target_column].map(SCHEMA.target_mapping).astype(int)
    X = df.drop(columns=[SCHEMA.target_column])

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CONFIG.test_size,
        random_state=MODEL_CONFIG.random_state,
        stratify=y
    )
    logger.info(f"Train split size: {len(X_train)} | Test split size: {len(X_test)}")
    logger.info(f"Train fraud rate: {y_train.mean():.4f} | Test fraud rate: {y_test.mean():.4f}")

    # -------------------------------------------------------------
    # STAGE 4: Preprocessing & Feature Preparation
    # -------------------------------------------------------------
    logger.info("Fitting preprocessing pipeline strictly on X_train...")
    preprocessor = FinTrustDataPreprocessor(schema=SCHEMA)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    logger.info(f"Transformed feature matrix shape: {X_train_transformed.shape}")

    # -------------------------------------------------------------
    # STAGE 5: Model Training
    # -------------------------------------------------------------
    logger.info("Training Random Forest Classifier with balanced class weights...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=MODEL_CONFIG.random_state,
        n_jobs=1
    )
    model.fit(X_train_transformed, y_train)
    logger.info("Model fitting complete.")

    # -------------------------------------------------------------
    # STAGE 6: Model Evaluation on Hold-Out Test Set
    # -------------------------------------------------------------
    logger.info("Evaluating model performance on test set...")
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    
    # Use operational threshold (0.35) for classification decisions
    threshold = MODEL_CONFIG.classification_threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "decision_threshold": threshold,
        "confusion_matrix": cm,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": X_train_transformed.shape[1],
    }

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS (Test Set)")
    print("="*50)
    print(f"ROC-AUC Score:  {metrics['roc_auc']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix [TN, FP / FN, TP]:\n{np.array(cm)}")
    print("="*50 + "\n")

    # -------------------------------------------------------------
    # STAGE 7: Artifact Persistence
    # -------------------------------------------------------------
    if save_artifacts:
        logger.info("Serializing preprocessor, model, and metrics artifacts...")
        preprocessor.save(PATHS.preprocessor_artifact_path)
        joblib.dump(model, PATHS.model_artifact_path)
        
        with open(PATHS.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Artifacts successfully saved to {PATHS.model_artifact_path.parent}")

    return model, preprocessor, metrics


if __name__ == "__main__":
    train_model()
