# FinTrust Data Dictionary & Schema Contract

This document provides a comprehensive data dictionary for the **FinTrust ML Workflow**. It establishes the formal data contract between data producers (core banking systems) and data consumers (ML engineering models).

---

## 1. Transaction Dataset (`data/raw/FinTrust_Transaction_Data.xlsx`)

- **Total Records:** 12,000
- **Total Attributes:** 11
- **Domain:** Banking transaction processing & fraud risk scoring

| Column Name | Data Type | Missing Count | Valid Range / Categories | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Transaction_ID` | String (Alphanumeric) | 0 (0.0%) | Regex: `^FT-T\d{6}$` | Primary identifier for each financial transaction. Cannot be null. |
| `Customer_ID` | String (Alphanumeric) | 0 (0.0%) | Regex: `^FT-C\d{5}$` | Foreign key referencing the customer entity. |
| `Transaction_DateTime` | Numeric / Datetime | 0 (0.0%) | Excel serial date float (e.g. 46023.00) or ISO timestamp | Date and time when the transaction took place. Converted to UTC datetime in pipeline. |
| `Transaction_Type` | String (Categorical) | 0 (0.0%) | `Transfer`, `Card Purchase`, `Bill Payment`, `Cash Withdrawal`, `Deposit`, `Airtime/Data` | Channel-level nature of the financial event. |
| `Amount_NGN` | Numeric (Float) | 0 (0.0%) | `> 0.00` and `<= 50,000,000.00` | Monetary value of the transaction in Nigerian Naira. Negative or zero values violate domain rules. |
| `Channel` | String (Categorical) | 0 (0.0%) | `Mobile App`, `POS`, `Web`, `ATM`, `USSD` | Client banking interface utilized. |
| `Device_Type` | String (Categorical) | 96 (0.8%) | `Android`, `iOS`, `POS Terminal`, `Web Browser`, `ATM Terminal` | Hardware device or terminal triggering the request. Missing values are imputed with `'Missing'`. |
| `Location` | String (Categorical) | 96 (0.8%) | `Lagos`, `Abuja`, `Port Harcourt`, `Kano`, `Ibadan`, `Enugu`, `Kaduna`, `Benin City` | Geographic origin of the request. Missing values are imputed with `'Missing'`. |
| `International_Transaction`| String (Binary) | 0 (0.0%) | `Yes`, `No` | Indicator whether transaction involves a foreign beneficiary or card network. |
| `Transaction_Status` | String (Categorical) | 0 (0.0%) | `Successful`, `Failed`, `Reversed`, `Pending` | Operational status of the transaction settlement. |
| `Risk_Review_Flag` | String (Binary) | 0 (0.0%) | `Yes` (2,352), `No` (9,648) | **Target Variable**: Indicates whether transaction was flagged for manual risk compliance review. |

---

## 2. Customer Dataset (`data/raw/FinTrust_Customer_Data.xlsx`)

- **Total Records:** 1,500
- **Total Attributes:** 12
- **Domain:** Customer Master Data / Demographics

| Column Name | Data Type | Missing Count | Valid Categories / Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Customer_ID` | String (Alphanumeric) | 0 (0.0%) | Regex: `^FT-C\d{5}$` | Primary identifier for customer entity. |
| `Customer_Name` | String | 0 (0.0%) | PII String (e.g., Ibrahim Adeyemi) | Customer full legal name (redacted / dropped before model training). |
| `Age` | Numeric (Float) | 0 (0.0%) | `18.0` to `100.0` | Age of customer. |
| `Gender` | String (Categorical) | 0 (0.0%) | `Male`, `Female`, `Prefer not to say` | Customer gender identity. |
| `City` | String (Categorical) | 0 (0.0%) | Nigerian cities (Lagos, Abuja, etc.) | Primary residential city of customer. |
| `Customer_Segment` | String (Categorical) | 0 (0.0%) | `Everyday`, `Premium`, `Student`, `SME` | Marketing and business tier segmentation. |
| `Account_Type` | String (Categorical) | 0 (0.0%) | `Savings`, `Current`, `Premium` | Tier and nature of the deposit account. |
| `Tenure_Months` | Numeric (Float) | 0 (0.0%) | `>= 0.0` | Number of months the customer has held an active relationship. |
| `Digital_Engagement_Score` | Numeric (Float) | 0 (0.0%) | `0.0` to `100.0` | Proprietary behavioral index measuring mobile and online banking frequency. |
| `Monthly_Income_Band` | String (Ordinal) | 0 (0.0%) | `Below 100k`, `100k-249k`, `250k-499k`, `500k-999k`, `1m+` | Self-reported monthly earnings bracket. |
| `Preferred_Channel` | String (Categorical) | 0 (0.0%) | `Mobile App`, `Web`, `USSD` | Channel with the highest customer login frequency. |
| `Account_Status` | String (Categorical) | 0 (0.0%) | `Active`, `Dormant`, `Restricted` | Current administrative standing of customer account. |

---

## 3. Data Transformation & Engineered Features

The preprocessing pipeline automatically derives the following temporal signals from `Transaction_DateTime`:

1. **`Transaction_Hour`** (Integer, 0-23): Extracted hour of transaction. Crucial for detecting unusual nocturnal activity (e.g., 2 AM - 4 AM).
2. **`Transaction_DayOfWeek`** (Integer, 0-6): 0 = Monday, 6 = Sunday.
3. **`Is_Weekend`** (Binary, 0 or 1): 1 if Saturday or Sunday, else 0.
