# End-to-End Loan Risk Prediction Pipeline (Czech Bank Dataset)

## Overview
This project delivers an **end-to-end loan risk prediction pipeline** using the **Czech Bank Financial Dataset**, combining **SQL-based feature engineering (ETL)** with **business-focused EDA** and **machine learning classification** to predict loan repayment risk.

The solution supports real-world banking use cases such as:
- **Risk-based underwriting**
- **Early warning signals for delinquency**
- **Portfolio risk monitoring** by customer segment and region

---

## Business Objective
Loan defaults directly impact profitability and portfolio stability. The objective is to predict **loan risk outcomes** using historical behavioral patterns, account activity, and regional demographics.

### Target Variable
`status ∈ {A, B, C, D}`

| Status | Meaning | Business Meaning |
|-------|---------|------------------|
| A | Contract finished, loan paid | ✅ Low Risk |
| B | Contract finished, loan not paid | ❌ High Risk |
| C | Contract running, loan being paid | ✅ Low Risk |
| D | Contract running, loan in debt | ❌ High Risk |

### Optional Binary Risk Mapping (for operational scoring)
To enable a single risk probability score:

- **Good (Non-default):** `{A, C}`
- **Risky (Default):** `{B, D}`

This enables:
✅ approval thresholding  
✅ manual review routing  
✅ collection prioritization  

---

## Data Sources (Raw Tables)
This project builds a loan-level dataset using the following normalized banking tables:

- `loan` — loan details + target label  
- `account` — account metadata (frequency, open date)  
- `client` — customer attributes  
- `disp` — account-client relationship mapping  
- `transaction_data` — transaction history and balances  
- `orders` — payment instructions  
- `card` — card ownership + type  
- `district` — socioeconomic regional indicators  

---

## Master Table (SQL ETL)

### Grain
✅ **1 row per Loan_ID (loan-level observation)**

### Why a Master Table?
Banking datasets are often normalized across entities. For modeling and analytics, we require a **denormalized dataset** containing:

- loan-level information  
- account behavior aggregates  
- customer linkage  
- regional demographics & risk indicators  

### ETL Design (High Level)
The SQL pipeline:
- uses `loan` as the **base table**
- joins account + customer context through `account_id` and `disp`
- aggregates transactions into behavioral features
- aggregates orders and card features
- enriches records with district demographics
- prevents join explosion by enforcing correct grain

✅ Outcome: a clean analytics-ready **loan_master table**

---

## Exploratory Data Analysis (Business-Focused)
EDA is structured to answer three real decision-making questions:

### 1) What is happening in the data?
- dataset shape, feature types  
- numeric and categorical distributions  
- target distribution and class imbalance awareness  

### 2) What is wrong in the data?
- missing values (imputation + missing flags)
- duplicates and grain validation (Loan_ID)
- outlier validation (high-risk/high-value behavior)
- datatype consistency (dates, numeric types)
- leakage checks (ensuring production-valid modeling)

### 3) What matters in the data?
- feature vs target relationships (risk drivers)
- correlation and multicollinearity checks
- region-level segmentation using district indicators
- time patterns (where applicable)

---

## Modeling Approach

### Problem Type
- **Primary:** Multi-class Classification (`A/B/C/D`)  
- **Business Deployment Option:** Binary Classification (Default vs Non-default)

### Baseline Models
- Logistic Regression (interpretable benchmark)
- Tree-based models (Random Forest / Gradient Boosting / XGBoost) for non-linear patterns

### Evaluation Strategy
Due to class imbalance, evaluation focuses on:
- **Macro / Weighted F1** (multi-class)
- **Recall on risky classes (B/D)** to avoid missing high-risk loans
- **PR-AUC** (binary mapping)
- Confusion Matrix to control false negatives
- Threshold tuning recommended for deployment decisioning

---

## Production-Ready Preprocessing (Scikit-learn Pipeline)
A Scikit-learn Pipeline is implemented for reproducibility and deployment readiness:

✅ Numeric Features  
- median imputation  
- scaling  

✅ Categorical Features  
- frequent imputation  
- `OneHotEncoder(handle_unknown="ignore")`

✅ Output  
- preprocessing + model training bundled into a **single pipeline artifact**
- consistent transformations across training and inference

---

## Key Outputs
- ✅ Loan-level master dataset (`loan_master`) using SQL ETL  
- ✅ Risk segmentation insights (EDA)  
- ✅ Multi-class + binary model training  
- ✅ Evaluation artifacts (F1, Recall, Confusion Matrix, PR-AUC)  
- ✅ Exportable pipeline ready for deployment/inference use

---

## Results (Add your actual metrics here)


- **Multi-class model:** Macro F1 = `X.XX`  
- **Binary risk model:** PR-AUC = `X.XX`  
- **Risk class recall (B/D):** `X.XX`

---

## Screenshots / Dashboard Preview


