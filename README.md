# FinTrust ML Workflow — Initial Implementation (Week 2)
### 
### Machine Learning Engineering Track | FinTrust Banking Case Study

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](tests/)

An end-to-end, production-grade Machine Learning Engineering workflow designed for **FinTrust** to detect and score transactions requiring manual risk review (`Risk_Review_Flag`).

This repository translates the Week 1 architectural design into a fully reproducible, modular Python implementation covering **Data Validation**, **Reproducible Preprocessing**, **Model Inference**, **Automated Technical Testing**, and **Microservice Serving**.

---

## 📑 Table of Contents
1. [Project Overview & Architecture](#-project-overview--architecture)
2. [Repository Structure (Part A)](#-repository-structure-part-a)
3. [Data Validation Component (Part B)](#-data-validation-component-part-b)
4. [Preprocessing Workflow (Part C)](#-preprocessing-workflow-part-c)
5. [The 7-Stage Prediction Pipeline (Part D)](#-the-7-stage-prediction-pipeline-part-d)
6. [Technical Testing Suite (Part E)](#-technical-testing-suite-part-e)
7. [Installation & Execution (Part F)](#-installation--execution-part-f)
8. [Assumptions & Known Limitations](#-assumptions--known-limitations)
9. [REST API Microservice (Advanced Component)](#-rest-api-microservice-advanced-component)
10. [Git / GitHub Management (Part G)](#-git--github-management-part-g)


---

## 🏛 Project Overview & Architecture

FinTrust processes thousands of transactions daily. To mitigate fraud and protect customer accounts, suspicious transactions must be flagged for manual compliance review.

```mermaid
flowchart LR
    Data[1. Ingest Data] --> Val[2. Validate Schema]
    Val --> Pre[3. Temporal & Impute]
    Pre --> Feat[4. Encode & Scale]
    Feat --> Model[5. Model Inference]
    Model --> Dec[6. Decision Logic]
    Dec --> Out[7. Structured Output]
```

### Key Engineering Principles:
- **Zero Monolithic Scripts:** Modular architecture separates configuration, validation, preprocessing, and inference.
- **Defensive Data Contracts:** All incoming payloads are validated at system boundaries.
- **Strict Leakage Prevention:** Feature transformers are fitted *only* on training data and frozen via `joblib`.
- **Traceability:** Intermediate data structures can be inspected at every stage.

---

## 📂 Repository Structure (Part A)

```
ANALYSTLAB ML WEEK 2 LAB/
├── .gitignore                      # Excludes venvs, caches, checkpoints, and OS files
├── README.md                       # Master documentation & execution guide
├── requirements.txt                # Frozen dependency specifications
├── main.py                         # Unified CLI runner (--train, --predict-sample, --test, --serve)
│
├── data/                           # Data directory (Raw and Processed)
│   ├── raw/
│   │   ├── FinTrust_Customer_Data.xlsx     # 1,500 customer records
│   │   └── FinTrust_Transaction_Data.xlsx  # 12,000 transaction records
│   └── processed/                  # Cached processed datasets
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── config.py                   # Centralized paths, schemas, and hyperparameters
│   ├── validation/                 # Part B: Data validation
│   │   ├── __init__.py
│   │   ├── schema.py               # Validation result dataclasses and error types
│   │   └── validator.py            # DataValidator rule engine
│   ├── preprocessing/              # Part C: Preprocessing workflow
│   │   ├── __init__.py
│   │   └── pipeline.py             # Temporal extractor & Scikit-Learn ColumnTransformer
│   ├── models/                     # Part D: Model & inference
│   │   ├── __init__.py
│   │   ├── train.py                # Model training, evaluation, and artifact saving
│   │   └── predict.py              # 7-stage prediction engine
│   └── api/                        # Optional Advanced Component: REST Service
│       ├── __init__.py
│       ├── schemas.py              # Pydantic request/response models
│       └── app.py                  # FastAPI microservice
│
├── tests/                          # Part E: Technical tests (pytest)
│   ├── __init__.py
│   ├── test_validation.py          # 6 tests for validation rules (Part B)
│   ├── test_preprocessing.py       # 3 tests for feature extraction and leakage prevention (Part C)
│   └── test_prediction.py          # 3 tests for end-to-end prediction & determinism (Part D)
│
├── artifacts/                      # Serialized ML artifacts
│   ├── preprocessor.joblib         # Fitted Scikit-Learn ColumnTransformer
│   ├── model.joblib                # Fitted Random Forest Classifier
│   └── metrics.json                # Model test performance metrics
│
├── notebooks/                      # Mentorship & Walkthrough
│   └── intern_walkthrough.ipynb    # Interactive step-by-step notebook for intern mentoring
│
└── docs/                           # Detailed Technical Documentation
    ├── DATA_DICTIONARY.md          # Full attribute definitions and schemas
    └── TEST_REPORT.md              # Test -> Expected -> Actual -> Status matrix
```

---

## 🛡 Data Validation Component (Part B)

Located in `src/validation/validator.py`, the `DataValidator` tests for:
1. **Empty Datasets:** Rejects 0-row DataFrames immediately with `EMPTY_DATASET`.
2. **Missing Expected Columns & Unexpected Columns:** Checks for required attributes; flags missing mandatory fields as critical errors and extra fields as non-blocking warnings.
3. **Data Types & Numeric Integrity:** Ensures numeric columns like `Amount_NGN` are strictly valid floats/ints, catching string inputs.
4. **Missing Values (Nulls):** Differentiates between critical primary keys (`Transaction_ID`, `Customer_ID`), which are rejected, and feature columns (`Device_Type`, `Location`), which are flagged for downstream imputation.
5. **Unexpected Categories:** Validates categorical fields (`Channel`, `Transaction_Type`, `Location`, etc.) against allowed domain values.
6. **Numeric Boundaries:** Catches negative or non-positive amounts (`Amount_NGN <= 0`) and extreme outliers.

### What Happens When Invalid Data is Detected:
- **Strict Mode (`strict=True`):** Used during online API inference. The validator raises a `DataValidationError` with a complete `ValidationReport`, returning an HTTP 422 payload to the client.
- **Lenient Mode (`strict=False`):** Used during batch processing. Non-critical warnings (e.g. missing `Device_Type`) are logged and forwarded to imputers, while corrupt records are separated into a quarantine report.

---

## ⚙ Preprocessing Workflow (Part C)

Located in `src/preprocessing/pipeline.py`, the preprocessing workflow guarantees **mathematical reproducibility** and **leakage prevention**:

1. **Temporal Feature Extraction:** Custom transformer `TemporalFeatureExtractor` converts timestamps into:
   - `Transaction_Hour`: Cyclical fraud signal (0 to 23).
   - `Transaction_DayOfWeek`: Day indicator (0 to 6).
   - `Is_Weekend`: Binary flag (0 or 1).
2. **Missing-Value Handling:**
   - Numerical columns: Median imputation (`SimpleImputer(strategy='median')`).
   - Categorical columns: Constant imputation (`SimpleImputer(strategy='constant', fill_value='Missing')`).
3. **Categorical Encoding:**
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Protects production systems against crashing when unobserved categories appear.
4. **Numerical Scaling:**
   - `StandardScaler()` centers and scales transaction amounts and engineered features.
5. **Serialization:**
   - Preprocessor is fitted strictly on `X_train` and saved to `artifacts/preprocessor.joblib`.

---

## 🚀 The 7-Stage Prediction Pipeline (Part D)

Located in `src/models/predict.py`, the `FinTrustPredictionPipeline` orchestrates data movement:

$$\text{Data} \longrightarrow \text{Validation} \longrightarrow \text{Preprocessing} \longrightarrow \text{Feature Preparation} \longrightarrow \text{Model} \longrightarrow \text{Prediction} \longrightarrow \text{Output}$$

### Stage Breakdown:
- **Stage 1 (Data Ingestion):** Accepts JSON, Dict, or Pandas DataFrame.
- **Stage 2 (Data Validation):** Enforces data contract via `DataValidator`.
- **Stage 3 (Preprocessing):** Extracts temporal signals and handles missing values.
- **Stage 4 (Feature Preparation):** Transforms inputs into the fixed 36-dimensional feature vector.
- **Stage 5 (Model Inference):** Evaluates `RandomForestClassifier.predict_proba()` to obtain continuous fraud probabilities.
- **Stage 6 (Prediction & Decision Logic):** Applies tuned decision threshold (`0.35`) and assigns risk tiers:
  - **Low Risk** ($< 0.30$): Auto-approve.
  - **Medium Risk** ($0.30 - 0.70$): Secondary verification (SMS OTP).
  - **High Risk** ($\ge 0.70$): Immediate hold for compliance review.
- **Stage 7 (Output Generation):** Generates structured payload with audit timestamp and operational recommendation.

---

## 🧪 Technical Testing Suite (Part E)

The test suite contains **12 automated tests** executed via `pytest`:

```bash
./venv/bin/pytest tests/ -v
```

### Test Summary Matrix:
| Test ID | Test Category | Description | Expected Result | Status |
| :--- | :--- | :--- | :--- | :-: |
| `test_valid_input` | Validation | Valid transaction payload | Passes with `action='PROCEED'` | **PASS** |
| `test_missing_values` | Validation | Missing Device_Type & missing ID | Warning for Device, Error for ID | **PASS** |
| `test_unexpected_category` | Validation | Channel = 'Telepathy' | Rejects with `UNEXPECTED_CATEGORY` | **PASS** |
| `test_incorrect_data_type` | Validation | Amount = 'twenty_thousand_naira' | Rejects with `INCORRECT_DATA_TYPE` | **PASS** |
| `test_empty_dataset` | Validation | 0-row DataFrame | Rejects with `EMPTY_DATASET` | **PASS** |
| `test_negative_amount` | Validation | Amount = -2500.00 | Rejects with `OUT_OF_RANGE` | **PASS** |
| `test_temporal_features` | Preprocessing | Datetime conversion & hours | Derives hour, day of week, weekend | **PASS** |
| `test_unseen_categories` | Preprocessing | Channel = 'VirtualReality' | Transforms safely with zero NaNs | **PASS** |
| `test_missing_imputation` | Preprocessing | Nulls in numeric & categorical | Imputes all NaNs cleanly | **PASS** |
| `test_pipeline_prediction` | Integration | End-to-end single transaction | Generates risk probability & tier | **PASS** |
| `test_pipeline_rejection` | Integration | Corrupt input to pipeline | Validation fails at Stage 2 | **PASS** |
| `test_reproducibility` | Integration | Repeated scoring on same data | Identical probabilities ($\ge 6$ d.p.) | **PASS** |

*For complete details, see [docs/TEST_REPORT.md](docs/TEST_REPORT.md).*

---

## 💻 Installation & Execution (Part F)

### Prerequisites:
- Python 3.10+ (Tested on Python 3.14 macOS arm64)
- Virtual environment tool (`venv`)

### 1. Setup Virtual Environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Model & Export Artifacts:
```bash
python main.py --train
```
*Output: Serializes `preprocessor.joblib`, `model.joblib`, and `metrics.json` into `artifacts/`.*

### 3. Run Sample 7-Stage Prediction Walkthrough:
```bash
python main.py --predict-sample
```

### 4. Run Technical Test Suite:
```bash
python main.py --test
# OR
pytest tests/ -v
```

### 5. Validate Raw Dataset:
```bash
python main.py --validate
```

---

## 🌐 REST API Microservice (Advanced Component)

To serve real-time predictions, launch the FastAPI microservice:
```bash
python main.py --serve
```
- **Service URL:** `http://127.0.0.1:8000`
- **Interactive OpenAPI/Swagger Docs:** `http://127.0.0.1:8000/docs`

### Example Request (`POST /predict`):
```json
{
  "Transaction_ID": "FT-T100001",
  "Customer_ID": "FT-C00234",
  "Transaction_DateTime": 46023.15,
  "Transaction_Type": "Transfer",
  "Amount_NGN": 450000.0,
  "Channel": "Web",
  "Device_Type": "Web Browser",
  "Location": "Lagos",
  "International_Transaction": "Yes",
  "Transaction_Status": "Successful"
}
```

### Example Response:
```json
{
  "Transaction_ID": "FT-T100001",
  "Customer_ID": "FT-C00234",
  "Predicted_Risk_Flag": "Yes",
  "Risk_Probability": 0.7622,
  "Risk_Tier": "High",
  "Operational_Action": "Immediate Review / Payment Hold",
  "Scored_At": "2026-09-22T00:40:28.342008+00:00"
}
```

---

## 📌 Assumptions & Known Limitations

### Assumptions:
1. **Excel Serial Timestamps:** Raw transaction timestamps are formatted as standard Excel serial dates (days since 1899-12-30). The pipeline automatically converts these to UTC datetime.
2. **Operational Threshold:** In fraud detection, missing fraud (False Negative) is significantly more expensive than investigating a false alarm (False Positive). We use an operational threshold of **0.35** rather than the arbitrary 0.50 default.
3. **Missing Categoricals:** Missing values in `Device_Type` and `Location` (0.8% of records) are treated as an informative missing state (`'Missing'`).

### Known Limitations:
1. **Single-Table Baseline:** The initial model is trained on transaction-level attributes. Customer profile attributes (`Age`, `Income_Band`, `Digital_Engagement_Score`) from `FinTrust_Customer_Data.xlsx` are ready to be joined in Week 3.
2. **Model Complexity:** We utilized a tuned `RandomForestClassifier` with balanced class weights as an interpretable baseline. Future iterations may explore gradient boosting (XGBoost/LightGBM) with hyperparameter tuning.
3. **Data Drift:** The current implementation validates schemas and categories, but does not yet monitor continuous distribution drift (e.g. Kolmogorov-Smirnov / Wasserstein tests).

---

## 👥 Mentor Walkthrough Guide

A dedicated teaching playbook for ML mentors is available at [`docs/MENTOR_GUIDE.md`](docs/MENTOR_GUIDE.md). It contains:
- Step-by-step teaching scripts for each section
- Socratic discussion prompts for interns
- Common intern pitfalls (data leakage, silent type coercion, unseen categories)
- Hands-on coding exercises and challenges
