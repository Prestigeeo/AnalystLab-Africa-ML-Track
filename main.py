"""
FinTrust ML Workflow — Main CLI Entry Point (Week 3: Integration-Ready)
======================================================================
MENTOR NOTE FOR INTERNS:
A production ML project should have an intuitive, unified CLI entry point.
This file allows engineers and interns to:
1. Prepare multi-table data (`--prepare-data`)
2. Train models and save adapters (`--train`)
3. Score sample inputs through the 7-stage pipeline (`--predict-sample`)
4. Score batch CSV datasets (`--predict-batch <input.csv>`)
5. Validate data schemas (`--validate`)
6. Run 30+ automated tests (`--test`)
7. Launch the REST API service (`--serve`)

Usage:
  python main.py --prepare-data
  python main.py --train
  python main.py --predict-sample
  python main.py --predict-batch data/processed/fintrust_test.csv
  python main.py --validate
  python main.py --test
  python main.py --serve
"""

import sys
import argparse
import subprocess
import json
from pathlib import Path
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS
from src.data.prepare import DataPreparator
from src.validation.validator import DataValidator
from src.models.train import train_model
from src.models.predict import FinTrustPredictionPipeline, predict_batch_dataset


def run_prepare_data_command():
    """Executes multi-table data ingestion, PII redaction, and relational enrichment."""
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: DATA PREPARATION & RELATIONAL ENRICHMENT")
    print("="*70)
    preparator = DataPreparator()
    results = preparator.prepare_and_export()
    print(f"\n[OK] Enriched master dataset: {results['total_enriched_records']:,} rows")
    print(f"[OK] Train split:             {results['train_records']:,} rows ({results['train_file']})")
    print(f"[OK] Test split:              {results['test_records']:,} rows ({results['test_file']})")
    print(f"[OK] Clean customer cache:    {PATHS.customer_cache_path}")


def run_training_command():
    """Trains baseline & production models on enriched data, saves adapter artifacts."""
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: ENRICHED MODEL TRAINING & ADAPTER PERSISTENCE")
    print("="*70)
    adapter, preprocessor, metrics = train_model(save_artifacts=True)
    print("\n[OK] Training completed successfully!")
    print(f"[OK] Model Adapter saved to:   {PATHS.model_artifact_path}")
    print(f"[OK] Preprocessor saved to:    {PATHS.preprocessor_artifact_path}")
    print(f"[OK] Governance metadata:     {PATHS.metadata_path}")


def run_predict_sample_command():
    """Demonstrates the 7-stage prediction pipeline on sample transactions with auto-enrichment."""
    sample_transaction = {
        "Transaction_ID": "FT-T888001",
        "Customer_ID": "FT-C00001",
        "Transaction_DateTime": 46023.125,  # 03:00 AM (Anomalous hour)
        "Transaction_Type": "Transfer",
        "Amount_NGN": 450000.0,            # High value transfer
        "Channel": "Web",
        "Device_Type": "Web Browser",
        "Location": "Lagos",
        "International_Transaction": "Yes", # Cross-border transfer
        "Transaction_Status": "Successful"
    }
    
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: SAMPLE PREDICTION & 7-STAGE PIPELINE WALKTHROUGH")
    print("="*70)
    pipeline = FinTrustPredictionPipeline(strict_validation=True)
    result = pipeline.run_pipeline(sample_transaction, verbose=True)
    
    print("PREDICTION RESULT SUMMARY:")
    print(result[[
        "Transaction_ID",
        "Customer_ID",
        "Age",
        "Customer_Segment",
        "Predicted_Risk_Flag",
        "Risk_Probability",
        "Risk_Tier",
        "Operational_Action"
    ]].to_string(index=False))


def run_predict_batch_command(input_file: str, output_file: str = "data/processed/scored_batch_output.csv"):
    """Scores an entire batch CSV file."""
    print("\n" + "="*70)
    print(f"FINTRUST ML WORKFLOW: BATCH PREDICTION ({input_file})")
    print("="*70)
    df_scored = predict_batch_dataset(input_file, output_file)
    print(f"\n[OK] Successfully scored {len(df_scored):,} transactions.")
    print(f"[OK] Output saved to {output_file}")


def run_validate_command():
    """Executes data validation on the enriched dataset."""
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: DATA VALIDATION SUITE")
    print("="*70)
    if not PATHS.processed_enriched_path.exists():
        preparator = DataPreparator()
        preparator.prepare_and_export()
    df = pd.read_csv(PATHS.processed_enriched_path)
    validator = DataValidator(strict=False)
    is_valid, report, _ = validator.validate(df, is_training=True, dataset_type="enriched")
    report.print_report()


def run_tests_command():
    """Executes pytest suite across validation, preprocessing, model adapter, prediction, and API."""
    print("\n" + "="*70)
    print("FINTRUST ML WORKFLOW: RUNNING 30-POINT TECHNICAL TEST SUITE")
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
        description="FinTrust Machine Learning Engineering Workflow — Week 3 Integration-Ready"
    )
    parser.add_argument("--prepare-data", action="store_true", help="Ingest raw files, redact PII, join and export enriched data")
    parser.add_argument("--train", action="store_true", help="Train baseline and production models, export adapters")
    parser.add_argument("--predict-sample", action="store_true", help="Run 7-stage prediction walkthrough on sample")
    parser.add_argument("--predict-batch", type=str, help="Score a batch CSV file (path to file)")
    parser.add_argument("--validate", action="store_true", help="Run data validation on enriched dataset")
    parser.add_argument("--test", action="store_true", help="Run automated pytest technical tests (30 tests)")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI REST prediction service")

    args = parser.parse_args()

    if args.prepare_data:
        run_prepare_data_command()
    elif args.train:
        run_training_command()
    elif args.predict_sample:
        run_predict_sample_command()
    elif args.predict_batch:
        run_predict_batch_command(args.predict_batch)
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
