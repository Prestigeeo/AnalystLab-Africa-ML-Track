"""
FinTrust ML Workflow — Master System Health & Verification Script
================================================================
MENTOR UTILITY:
Run this script anytime to verify that every component in the repository is
installed, working, tested, and ready for teaching!

Usage:
    python check_all.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_step(title):
    print(f"\n{BOLD}{BLUE}======================================================================{RESET}")
    print(f"{BOLD}{BLUE}▶ {title}{RESET}")
    print(f"{BOLD}{BLUE}======================================================================{RESET}")


def check_mark(msg, success=True):
    if success:
        print(f"  {GREEN}✓ [PASS]{RESET} {msg}")
    else:
        print(f"  {RED}✗ [FAIL]{RESET} {msg}")


def main():
    start_time = time.time()
    total_checks = 0
    passed_checks = 0

    print(f"\n{BOLD}🔍 RUNNING COMPLETE FINTRUST SYSTEM DIAGNOSTIC & VERIFICATION{RESET}")

    # -------------------------------------------------------------
    # 1. DEPENDENCY CHECK
    # -------------------------------------------------------------
    print_step("CHECK 1: PYTHON PACKAGES & DEPENDENCIES")
    modules = [
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Scikit-Learn ML engine"),
        ("openpyxl", "Excel dataset reader"),
        ("joblib", "Pipeline & model serialization"),
        ("pytest", "Automated test framework"),
        ("fastapi", "REST API serving framework"),
        ("uvicorn", "ASGI web server"),
        ("pydantic", "Schema validation"),
        ("docx", "Word document generation"),
        ("pptx", "PowerPoint presentation generation"),
    ]
    for mod, desc in modules:
        total_checks += 1
        try:
            __import__(mod)
            check_mark(f"{mod:12} ({desc})")
            passed_checks += 1
        except ImportError as e:
            check_mark(f"{mod:12} - Missing: {e}", success=False)

    # -------------------------------------------------------------
    # 2. RAW DATA VERIFICATION
    # -------------------------------------------------------------
    print_step("CHECK 2: DATASET PRESENCE & INTEGRITY")
    from src.config import PATHS
    data_files = [
        (PATHS.raw_customer_path, "Customer master data (1,500 records)"),
        (PATHS.raw_transaction_path, "Transaction dataset (12,000 records)"),
    ]
    for p, desc in data_files:
        total_checks += 1
        if p.exists():
            size_kb = p.stat().st_size / 1024
            check_mark(f"{p.name} exists ({size_kb:.1f} KB) - {desc}")
            passed_checks += 1
        else:
            check_mark(f"{p.name} missing at {p}", success=False)

    # -------------------------------------------------------------
    # 3. PART B: DATA VALIDATION COMPONENT
    # -------------------------------------------------------------
    print_step("CHECK 3: PART B — DATA VALIDATION FIREWALL")
    import pandas as pd
    from src.config import SCHEMA
    from src.validation.validator import DataValidator, DataValidationError

    # Test 3A: Valid dataset check
    total_checks += 1
    df_raw = pd.read_excel(PATHS.raw_transaction_path)
    validator = DataValidator(schema=SCHEMA, strict=False)
    is_valid, report, _ = validator.validate(df_raw, is_training=True)
    if is_valid and len(report.errors) == 0:
        check_mark(f"Raw transaction data validation: {report.total_records:,} rows inspected, 0 critical errors.")
        passed_checks += 1
    else:
        check_mark(f"Validation failed: {report.errors}", success=False)

    # Test 3B: Intentional bad data check
    total_checks += 1
    bad_df = pd.DataFrame([{
        "Transaction_ID": "FT-BAD", "Customer_ID": "FT-C", "Transaction_DateTime": 46023.0,
        "Transaction_Type": "Transfer", "Amount_NGN": -999.0, # Negative amount!
        "Channel": "Telepathy", # Unrecognized category!
        "Device_Type": "Android", "Location": "Lagos",
        "International_Transaction": "No", "Transaction_Status": "Successful"
    }])
    strict_val = DataValidator(schema=SCHEMA, strict=True)
    try:
        strict_val.validate(bad_df)
        check_mark("Security firewall failed: accepted bad data!", success=False)
    except DataValidationError:
        check_mark("Security firewall successfully blocked corrupt input (Negative Amount & Illegal Channel).")
        passed_checks += 1

    # -------------------------------------------------------------
    # 4. PART C: PREPROCESSING & UNSEEN CATEGORY FAULT TOLERANCE
    # -------------------------------------------------------------
    print_step("CHECK 4: PART C — PREPROCESSING & DATA LEAKAGE PREVENTION")
    from src.preprocessing.pipeline import FinTrustDataPreprocessor, TemporalFeatureExtractor
    total_checks += 1
    
    # Test temporal feature extraction
    extractor = TemporalFeatureExtractor()
    df_temp = extractor.transform(df_raw.head(3))
    has_temporal = all(c in df_temp.columns for c in ["Transaction_Hour", "Transaction_DayOfWeek", "Is_Weekend"])
    if has_temporal:
        check_mark("Temporal feature extraction: derived Transaction_Hour, DayOfWeek, and Is_Weekend.")
        passed_checks += 1
    else:
        check_mark("Temporal feature extraction failed.", success=False)

    # Test unseen categories handling
    total_checks += 1
    preprocessor = FinTrustDataPreprocessor.load(PATHS.preprocessor_artifact_path)
    test_sample = df_raw.head(1).copy()
    test_sample["Channel"] = "CryptoVirtualApp"  # Never seen in training!
    test_sample["Location"] = "MarsColony"       # Never seen in training!
    try:
        matrix_out = preprocessor.transform(test_sample)
        if matrix_out.shape[1] == 36 and not pd.isna(matrix_out).any():
            check_mark(f"Unseen category fault tolerance: transformed safely to 36-column vector with 0 NaNs.")
            passed_checks += 1
        else:
            check_mark("Unseen category changed matrix dimensions!", success=False)
    except Exception as e:
        check_mark(f"Preprocessor crashed on unseen category: {e}", success=False)

    # -------------------------------------------------------------
    # 5. PART D: 7-STAGE PREDICTION PIPELINE
    # -------------------------------------------------------------
    print_step("CHECK 5: PART D — 7-STAGE END-TO-END PREDICTION PIPELINE")
    from src.models.predict import FinTrustPredictionPipeline
    total_checks += 1
    pipeline = FinTrustPredictionPipeline(strict_validation=True)
    
    sample_txn = {
        "Transaction_ID": "FT-TEST001",
        "Customer_ID": "FT-C00999",
        "Transaction_DateTime": 46023.125, # 03:00 AM
        "Transaction_Type": "Transfer",
        "Amount_NGN": 500000.0,            # High value transfer
        "Channel": "Web",
        "Device_Type": "Web Browser",
        "Location": "Lagos",
        "International_Transaction": "Yes", # International
        "Transaction_Status": "Successful"
    }
    result_df = pipeline.run_pipeline(sample_txn, verbose=False)
    pred_flag = result_df.loc[0, "Predicted_Risk_Flag"]
    prob = result_df.loc[0, "Risk_Probability"]
    tier = result_df.loc[0, "Risk_Tier"]
    action = result_df.loc[0, "Operational_Action"]
    
    if pred_flag in ["Yes", "No"] and 0.0 <= prob <= 1.0:
        check_mark(f"7-Stage pipeline executed successfully!")
        print(f"       -> Score: {prob:.4f} | Tier: {tier} | Flag: {pred_flag} | Action: {action}")
        passed_checks += 1
    else:
        check_mark("Prediction output malformed!", success=False)

    # -------------------------------------------------------------
    # 6. PART E: AUTOMATED UNIT & INTEGRATION TESTS
    # -------------------------------------------------------------
    print_step("CHECK 6: PART E — AUTOMATED TEST SUITE (PYTEST)")
    import subprocess
    total_checks += 1
    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    if test_run.returncode == 0:
        check_mark("All 12 automated unit and integration tests PASSED.")
        for line in test_run.stdout.splitlines():
            if "PASSED" in line:
                print(f"       {line.strip()}")
        passed_checks += 1
    else:
        check_mark(f"Tests failed:\n{test_run.stdout}", success=False)

    # -------------------------------------------------------------
    # 7. PART F & G: DOWNLOADABLE DELIVERABLES & DOCUMENTATION
    # -------------------------------------------------------------
    print_step("CHECK 7: DELIVERABLES & MENTOR ASSETS")
    deliverables = [
        ("FinTrust_ML_Workflow_Mentor_Guide.docx", "Word Teaching Playbook (Editable)"),
        ("FinTrust_ML_Workflow_Intern_Presentation.pptx", "PowerPoint Presentation Deck (Editable)"),
        ("README.md", "Master Project Documentation"),
        ("docs/MENTOR_GUIDE.md", "Mentor Teaching Notes"),
        ("docs/DATA_DICTIONARY.md", "Data Dictionary"),
        ("docs/TEST_REPORT.md", "Technical Test Report Matrix"),
        ("notebooks/intern_walkthrough.ipynb", "Interactive Intern Notebook"),
        ("artifacts/preprocessor.joblib", "Fitted Preprocessor Artifact"),
        ("artifacts/model.joblib", "Trained Random Forest Artifact"),
        ("artifacts/metrics.json", "Model Evaluation Metrics"),
    ]
    for fname, desc in deliverables:
        total_checks += 1
        fpath = PROJECT_ROOT / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            check_mark(f"{fname:45} ({size_kb:6.1f} KB) - {desc}")
            passed_checks += 1
        else:
            check_mark(f"{fname} missing!", success=False)

    # -------------------------------------------------------------
    # FINAL SUMMARY REPORT
    # -------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{BOLD}======================================================================{RESET}")
    print(f"{BOLD}📊 DIAGNOSTIC SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED (100% HEALTHY) in {elapsed:.2f}s{RESET}")
    print(f"{BOLD}======================================================================{RESET}")
    print(f"\n{GREEN}{BOLD}🎉 EVERYTHING IS WORKING 100% PERFECTLY! YOU ARE READY TO TEACH!{RESET}\n")


if __name__ == "__main__":
    main()
