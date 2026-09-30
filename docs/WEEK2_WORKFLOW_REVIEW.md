# FinTrust ML Workflow — Week 2 Technical Audit & Review (Part A)

**Author:** Machine Learning Engineering Track  
**Focus:** Critical Technical Review of Week 2 Implementation  
**Context:** Upgrading the FinTrust Banking Risk Review System from Week 2 prototype to Week 3 integration-ready system.

---

## 1. Executive Summary

Week 2 established an initial reproducible technical workflow for FinTrust's transaction risk scoring system (`Risk_Review_Flag`), featuring modular data validation, Scikit-learn preprocessing pipelines, model training with Random Forest, 12 automated tests, and a prototype FastAPI service.

While Week 2 fulfilled its foundational goals, transitioning to an **integration-ready, enterprise-grade ML system** in Week 3 requires resolving architectural gaps, decoupling data preparation, formalizing model integration interfaces, expanding automated testing, and eliminating technical debt.

---

## 2. Incomplete Components Identified in Week 2

| Component | Week 2 Status | Identified Gap / Limitation | Week 3 Remedy |
| :--- | :--- | :--- | :--- |
| **Customer Dataset Integration** | Incomplete / Isolated | `FinTrust_Customer_Data.xlsx` (1,500 records) was stored in `data/raw/` but never joined or utilized during training or inference. Only transaction attributes were analyzed. | Implement `DataPreparator` in `src/data/prepare.py` to execute a relational left join on `Customer_ID`, enriching transactions with demographic & behavioral signals (`Age`, `Account_Type`, `Tenure_Months`, `Digital_Engagement_Score`, `Monthly_Income_Band`). |
| **Data Preparation Pipeline** | Missing as standalone module | Data ingestion, cleaning, and train/test splitting were bundled directly inside `train.py`. No standalone CLI command existed to prepare and version data artifacts. | Create dedicated `src/data/prepare.py` with standalone CLI command (`python main.py --prepare-data`), exporting versioned data to `data/processed/`. |
| **Model Interface / Adapter** | Missing abstraction | Training code was tightly coupled to Scikit-Learn's `RandomForestClassifier`. No standardized interface existed for integrating external Data Science models (e.g. XGBoost, LightGBM, or models built by peer interns). | Implement `BaseModelAdapter` abstract base class and `ExternalModelAdapter` in `src/models/interface.py` with automated signature and dimension validation. |
| **Batch File-to-File Prediction** | Missing | Week 2 only supported single JSON payload prediction via CLI and API. Batch scoring directly from CSV files to output files was not implemented. | Add `--predict-batch <input.csv> --output-file <output.csv>` CLI command and `/predict/batch` endpoint. |
| **Containerization (Docker)** | Incomplete | Week 2 provided virtual environment instructions (`requirements.txt`), but lacked container specifications (`Dockerfile`) for cross-platform OS isolation. | Create production-grade `Dockerfile` and `.dockerignore`. |

---

## 3. Technical Weaknesses Identified in Week 2

1. **Deprecated FastAPI Event Handlers:**
   - *Issue:* In `src/api/app.py`, model artifacts were loaded using `@app.on_event("startup")`. In modern FastAPI (version 0.100+), `on_event` is deprecated and generates runtime warnings.
   - *Remedy:* Refactor to modern `lifespan` context manager (`@asynccontextmanager`).

2. **Hardcoded Model Architecture:**
   - *Issue:* `train.py` hardcoded a single Random Forest model with fixed hyperparameter bounds. If another team provided a superior gradient boosted model, integration required modifying core pipeline code.
   - *Remedy:* Decouple training and inference using the `BaseModelAdapter` contract pattern.

3. **Narrow Validation Scope for Customer Features:**
   - *Issue:* `DataValidator` in Week 2 only validated transaction columns (`Amount_NGN`, `Channel`, etc.). Customer demographic columns were unmonitored.
   - *Remedy:* Expand `FeatureSchema` and `DataValidator` to support both transaction-only and enriched customer-transaction records, with boundary checks on `Age` (18–100), `Tenure_Months` (>= 0), and `Digital_Engagement_Score` (0–100).

4. **Single-Table Baseline:**
   - *Issue:* Without customer profile features, the model lacked critical financial context (e.g., whether a ₦500,000 transfer is standard for a high-net-worth customer or anomalous for a student account).
   - *Remedy:* Multi-table relational enrichment on `Customer_ID`.

---

## 4. Testing Gaps Identified in Week 2

Week 2 contained 12 automated unit and integration tests. However, the following critical test scenarios were missing:

1. **Model Loading Failure Modes:**
   - What happens if `model.joblib` or `preprocessor.joblib` is deleted?
   - What happens if the artifact file is corrupted or contains invalid binary data?
   - What happens if an external model fails the required prediction contract?
2. **Prediction Generation & Probability Calibration:**
   - Do batch predictions return probabilities strictly bounded in $[0.0, 1.0]$?
   - Does prediction latency meet real-time operational thresholds ($< 50\text{ ms}$ per record)?
   - Are predictions consistent with decision thresholds (e.g., probability 0.72 correctly yields "High" tier and "Yes" flag)?
3. **Strict Output Schema Conformance:**
   - Does the prediction response dictionary strictly contain all required production keys (`Transaction_ID`, `Predicted_Risk_Flag`, `Risk_Probability`, `Risk_Tier`, `Operational_Action`, `Scored_At`)?
   - Is `Scored_At` formatted as a valid ISO-8601 UTC timestamp?
4. **Data Preparation & Customer Merge Integrity:**
   - Does the join handle missing customers gracefully?
   - Are zero duplicate rows introduced during relational joining?

---

## 5. Reproducibility Issues Identified in Week 2

1. **Coupled Data Preparation:**
   - In Week 2, a user had to execute `python main.py --train` which performed raw Excel parsing on every run. If the raw Excel file changed or was locked by another process, training failed silently.
   - *Resolution:* Decouple data preparation into an explicit stage:
     `Raw Data` $\to$ `Data Preparation (Clean/Join)` $\to$ `data/processed/` $\to$ `Model Training`.
2. **Environment Reproducibility Across Operating Systems:**
   - While `requirements.txt` froze Python package versions, differences in underlying system C-libraries (BLAS, LAPACK, Apple Accelerate vs. OpenBLAS on Linux) can introduce floating-point discrepancies across OS platforms.
   - *Resolution:* Introduce `Dockerfile` establishing an immutable Debian-based Python container runtime.
3. **Artifact Provenance & Metadata:**
   - `metrics.json` recorded basic scores but omitted artifact checksums, training dataset hashes, feature lists, and hyperparameter provenance.
   - *Resolution:* Generate enriched model metadata recording feature names, input dimensions, training timestamps, and algorithm configurations.

---

## 6. Action Plan for Week 3

- [x] **Part A:** Complete formal review and document technical weaknesses (this document).
- [ ] **Part B:** Implement `src/data/prepare.py` with multi-table relational enrichment and modularize preprocessing.
- [ ] **Part C:** Implement `src/models/interface.py` with `BaseModelAdapter` and external model integration capabilities.
- [ ] **Part D:** Expand automated test suite to 18+ tests covering all 9 required criteria.
- [ ] **Part E:** Modernize FastAPI service with `lifespan` handler and batch/enriched scoring endpoints.
- [ ] **Part F:** Add Docker containerization, update README, and generate Week 3 mentor Word and PowerPoint deliverables.
