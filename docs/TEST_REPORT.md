# FinTrust ML Workflow — Technical Test Report (Part E)

**Execution Date:** 2026-09-22  
**Environment:** Python 3.14 (macOS arm64), pytest 9.1.1, scikit-learn 1.9.1, pandas 3.0.6  
**Test Suite:** `tests/test_validation.py`, `tests/test_preprocessing.py`, `tests/test_prediction.py`  
**Overall Result:** 12 Passed, 0 Failed (100% Pass Rate)

---

## 1. Test Execution Matrix (Test → Expected Result → Actual Result → Status)

| # | Test Identifier | Test Category | Description & Input | Expected Result | Actual Result | Status |
| :- | :--- | :--- | :--- | :--- | :--- | :-: |
| **1** | `test_valid_input` | Validation | Valid single transaction payload with complete attributes (`FT-T000001`, `Transfer`, `Amount=15000.50`, etc.). | `is_valid == True`, `report.errors == []`, `action_taken == 'PROCEED'`. | `is_valid: True`, `errors: 0`, `action_taken: PROCEED`. | **PASS** |
| **2** | `test_missing_values_warning_and_error` | Validation | Subcase A: Missing non-critical feature (`Device_Type = NaN`).<br>Subcase B: Missing primary key (`Transaction_ID = NaN`). | Subcase A: Generates non-blocking WARNING (`MISSING_VALUES`).<br>Subcase B: Raises `DataValidationError` with `NULL_PRIMARY_KEY`. | Subcase A: Warning issued, downstream imputation applied.<br>Subcase B: `DataValidationError` raised with `NULL_PRIMARY_KEY`. | **PASS** |
| **3** | `test_unexpected_category` | Validation | Categorical column has illegal domain value (`Channel = 'Telepathy'`). | Raises `DataValidationError` with error code `UNEXPECTED_CATEGORY`. | Detected unrecognized category `'Telepathy'`. Blocked with `UNEXPECTED_CATEGORY`. | **PASS** |
| **4** | `test_incorrect_data_type` | Validation | Numeric field receives alphanumeric string (`Amount_NGN = 'twenty_thousand_naira'`). | Raises `DataValidationError` with error code `INCORRECT_DATA_TYPE`. | Type conversion failed. Caught with `INCORRECT_DATA_TYPE` error. | **PASS** |
| **5** | `test_empty_dataset` | Validation | Empty DataFrame (`0` rows). | Raises `DataValidationError` with error code `EMPTY_DATASET`. | Empty dataset detected. Pipeline aborted immediately with `EMPTY_DATASET`. | **PASS** |
| **6** | `test_negative_amount_input` | Validation | Out-of-bounds numeric rule: `Amount_NGN = -2500.00`. | Raises `DataValidationError` with error code `OUT_OF_RANGE`. | Value violates rule `Amount_NGN > 0`. Flagged with `OUT_OF_RANGE`. | **PASS** |
| **7** | `test_temporal_feature_extraction` | Preprocessing | Excel serial dates (e.g. `46023.0` and `46025.5`). | Correctly derives `Transaction_Hour` (0, 12), `DayOfWeek`, and `Is_Weekend` (0, 1). | Transformed DataFrame contains all 3 temporal features with expected values. | **PASS** |
| **8** | `test_unseen_categories_handling` | Preprocessing | Test data contains unseen categories (`Channel='VirtualReality'`, `Location='Atlantis'`). | Preprocessor uses `handle_unknown='ignore'`, producing identical column dimensionality without errors. | Output matrix shape matches training shape `(N, 36)` with zeros in unobserved columns and zero NaNs. | **PASS** |
| **9** | `test_missing_value_imputation` | Preprocessing | Input contains `NaN` values in numeric and categorical fields. | Numerical median and categorical constant imputers fill nulls cleanly. | Transformed matrix contains zero NaNs (`np.isnan().sum() == 0`). | **PASS** |
| **10**| `test_pipeline_valid_single_prediction` | Integration | Full 7-stage prediction on sample transaction. | Returns enriched record with `Risk_Probability` $\in [0, 1]$, `Predicted_Risk_Flag`, and `Risk_Tier`. | Output record enriched with score, probability, risk tier, and action recommendation. | **PASS** |
| **11**| `test_pipeline_rejects_corrupted_data` | Integration | Corrupted input (`Amount_NGN = -100`) passed to prediction pipeline. | Fails at Stage 2 (Validation); does not reach Stage 5 (Model). | Execution halted at Stage 2 with `DataValidationError`. Model inference protected. | **PASS** |
| **12**| `test_pipeline_reproducibility` | Integration | Repeated scoring calls on the exact same input payload. | Deterministic scoring output: probabilities match to $\ge 6$ decimal places. | `Risk_Probability` outputs match identically across invocations. | **PASS** |

---

## 2. Summary of Failure Handling Policies

| Condition Detected | Pipeline Action | HTTP API Status Code | Downstream Effect |
| :--- | :--- | :--- | :--- |
| **Missing Primary Key** | Rejected | `422 Unprocessable Entity` | Transaction aborted; logged for fraud audit. |
| **Non-Numeric Amount** | Rejected | `422 Unprocessable Entity` | Transaction aborted; client instructed to fix schema. |
| **Negative Amount** | Rejected | `422 Unprocessable Entity` | Transaction aborted; domain violation reported. |
| **Unexpected Category (Online)** | Rejected | `422 Unprocessable Entity` | Prevents arbitrary input injection. |
| **Unexpected Category (Batch)** | Gracefully Zeroed | N/A | Handled via OneHotEncoder `handle_unknown='ignore'`. |
| **Missing Feature (Device/City)** | Warning + Imputed | `200 OK` | Imputed with training median/constant (`'Missing'`). |
| **Empty Input** | Rejected | `400 Bad Request` | Execution halted before resource allocation. |
