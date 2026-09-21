"""
FinTrust ML Workflow — Main CLI Entry Point
==========================================
MENTOR NOTE FOR INTERNS:
A production ML project should have an intuitive, unified CLI entry point.
This file allows engineers to train models, score sample inputs, run technical
tests, or start the REST API directly from the terminal.

Usage:
  python main.py --train            # Train model and export artifacts
  python main.py --predict-sample   # Run 7-stage prediction walkthrough on sample data
  python main.py --validate         # Run data validation on raw transaction data
  python main.py --test             # Run automated pytest test suite
  python main.py --serve            # Launch FastAPI prediction service
"""

import sys
import argparse
import subprocess
import json
import pandas as pd

from src.config import PATHS
from src.validation.validator import DataValidator
from src.models.train import train_model, load_raw_data
from src.models.predict import FinTrustPredictionPipeline


def run_training_command():
    """Trains the model and saves preprocessor and model artifacts."""
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: TRAINING PIPELINE")
    print("="*70)
    model, preprocessor, metrics = train_model(save_artifacts=True)
    print("\nTraining completed successfully! Artifacts saved to 'artifacts/'.")


def run_predict_sample_command():
    """Demonstrates the 7-stage prediction pipeline on sample transactions."""
    sample_transaction = {
        "Transaction_ID": "FT-T888001",
        "Customer_ID": "FT-C00452",
        "Transaction_DateTime": 46023.125,  # 03:00 AM (High fraud risk hour!)
        "Transaction_Type": "Transfer",
        "Amount_NGN": 450000.0,            # High value transfer
        "Channel": "Web",
        "Device_Type": "Web Browser",
        "Location": "Lagos",
        "International_Transaction": "Yes", # International transfer
        "Transaction_Status": "Successful"
    }
    
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: SAMPLE PREDICTION WALKTHROUGH")
    print("="*70)
    pipeline = FinTrustPredictionPipeline(strict_validation=True)
    result = pipeline.run_pipeline(sample_transaction, verbose=True)
    
    print("PREDICTION RESULT:")
    print(result[[
        "Transaction_ID",
        "Predicted_Risk_Flag",
        "Risk_Probability",
        "Risk_Tier",
        "Operational_Action"
    ]].to_string(index=False))


def run_validate_command():
    """Executes data validation on the raw transaction dataset."""
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: DATA VALIDATION SUITE")
    print("="*70)
    df = load_raw_data()
    validator = DataValidator(strict=False)
    is_valid, report, _ = validator.validate(df, is_training=True)
    report.print_report()


def run_tests_command():
    """Executes pytest suite across validation, preprocessing, and prediction."""
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: RUNNING TECHNICAL TEST SUITE")
    print("="*70)
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def run_serve_command(host: str = "127.0.0.1", port: int = 8000):
    """Launches the FastAPI microservice."""
    import uvicorn
    print(f"\nStarting FinTrust Prediction Service on http://{host}:{port}...")
    print(f"Interactive API Documentation: http://{host}:{port}/docs\n")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="FinTrust Machine Learning Engineering Workflow — Week 2 Lab"
    )
    parser.add_argument("--train", action="store_true", help="Train model and save artifacts")
    parser.add_argument("--predict-sample", action="store_true", help="Run 7-stage prediction walkthrough on sample")
    parser.add_argument("--validate", action="store_true", help="Run data validation on raw dataset")
    parser.add_argument("--test", action="store_true", help="Run automated pytest technical tests")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI REST prediction service")

    args = parser.parse_args()

    if args.train:
        run_training_command()
    elif args.predict_sample:
        run_predict_sample_command()
    elif args.validate:
        run_validate_command()
    elif args.test:
        run_tests_command()
    elif args.serve:
        run_serve_command()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
