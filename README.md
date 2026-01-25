# End-to-End Loan Risk Prediction Pipeline (Czech Bank Dataset)

## Overview
This project implements a **production-style, end-to-end Loan Risk Prediction pipeline** using the **Czech Bank Financial Dataset**.  
It combines **SQL-based ETL + feature engineering**, **business-driven EDA**, and **machine learning classification** to predict **loan repayment risk** at loan level.

The solution supports real banking use cases such as:
- **Risk-based underwriting and approval automation**
- **Early warning signals for delinquency**
- **Portfolio risk monitoring by region and customer segment**
- **Risk-driven collection and manual review prioritization**

---

## Business Objective
Loan defaults negatively impact both **profitability** and **credit portfolio stability**.  
The goal of this project is to **predict loan risk outcomes** using historical patterns in:

- customer transactional behavior  
- account activity and financial stability  
- demographic and regional economic indicators  

---

## Target Variable
The dataset contains a multi-class loan status variable:

`Status ∈ {A, B, C, D}`

| Status | Meaning | Business Interpretation |
|------:|---------|--------------------------|
| A | Contract finished, loan paid | ✅ Low Risk |
| B | Contract finished, loan not paid | ❌ High Risk |
| C | Contract running, being paid | ✅ Low Risk |
| D | Contract running, in debt | ❌ High Risk |

---

## Binary Risk Mapping (Deployment-Friendly)
To support operational scoring (single risk probability), the problem can also be mapped to binary classification:

- **Non-default (Good)** → {A, C}  
- **Default (Risky)** → {B, D}

This enables:
- ✅ approval thresholding  
- ✅ manual review routing  
- ✅ collection prioritization  

---

## Data Sources (Raw Tables)
The project creates a loan-level analytical dataset by integrating the following normalized tables:

- `loan` — loan details + target label  
- `account` — account metadata (frequency, open date)  
- `client` — client attributes  
- `disp` — account-client relationship mapping  
- `transaction_data` — transaction history and balances  
- `orders` — payment instructions  
- `card` — card ownership + type  
- `district` — socioeconomic regional indicators  

---

## SQL ETL Output: Master Table (`loan_master`)

### Grain
✅ **1 row per Loan_ID** (loan-level observation)

### Why a Master Table?
Banking datasets are typically normalized across multiple entities.  
For modeling and analytics, we require a **denormalized loan-level dataset** containing:

- loan-level details  
- account activity and behavior aggregates  
- customer/account linking features  
- regional demographics and risk indicators  

### ETL Design (High Level)
The SQL pipeline is designed to:
- use `loan` as the base table  
- join account + client context using `account_id` and `disp`  
- aggregate transaction history into behavioral features (counts, balances, volatility)  
- generate card ownership flags and order-based aggregates  
- enrich with district-level socioeconomic features  
- prevent join explosion by enforcing strict loan-level grain  

✅ Outcome: a clean, analytics-ready **loan_master** table for EDA + ML.

---

## Exploratory Data Analysis (Business-Focused)
EDA is built to support decisions, not just plots. It answers:

### 1) What is happening in the data?
- dataset shape and feature types  
- categorical / numeric distribution analysis  
- target distribution and class imbalance awareness  

### 2) What is wrong in the data?
- missing value analysis and imputation strategy  
- duplicate checks and grain validation (Loan_ID)  
- outlier validation and extreme behavior detection  
- datatype consistency (dates, numerics, categories)  
- leakage checks to ensure production-valid model features  

### 3) What matters for loan risk?
- feature vs target comparisons (risk drivers)  
- correlation and multicollinearity checks  
- regional segmentation using district indicators  
- behavioral patterns from transaction aggregates  

---

## Modeling Approach

### Problem Type
- **Primary:** Multi-class Classification (A/B/C/D)  
- **Deployment Option:** Binary Classification (Default vs Non-default)  

### Models Evaluated
A baseline-to-advanced modeling approach is applied:
- Logistic Regression (interpretable benchmark)  
- Tree-based ensembles (RandomForest, ExtraTrees)  
- Boosting models (AdaBoost, GradientBoosting)  
- Advanced boosting libraries (XGBoost, LightGBM)  
- SVM, KNN, Naive Bayes (for comparison)  

---

## Evaluation Strategy (Imbalance-Aware)
Loan default prediction is an **imbalanced classification problem**, so accuracy alone is not reliable.

The evaluation focuses on:
- **F1-score for risky/default class** (binary)  
- **Macro/Weighted F1-score** (multi-class)  
- **Recall on risky classes (B/D)** to reduce missed defaulters  
- **ROC-AUC / PR-AUC** for ranking and probability quality  
- **Confusion matrix** to manage false-negative risk  
- **Threshold tuning** for real-world approval decisioning  

✅ Threshold selection is implemented using a **Train / Validation / Test** strategy to avoid leakage:
- threshold tuned on validation set  
- final unbiased reporting on test set  

---

## Production-Ready Preprocessing (Scikit-learn Pipeline)
A full **Scikit-learn Pipeline** is used to ensure reproducibility and deployment readiness.

### Numeric Features
- median imputation  
- standard scaling  

### Categorical Features
- most frequent imputation  
- `OneHotEncoder(handle_unknown="ignore")`  

✅ Output:
- preprocessing + model training combined into a single pipeline artifact  
- consistent transformations for both training and inference  

---

## Key Outputs
This project produces:

- ✅ SQL-generated loan-level master dataset (**loan_master**)  
- ✅ Business insights and risk drivers from EDA  
- ✅ Multi-model benchmarking and model selection  
- ✅ Threshold-tuned best model selection (validation-based)  
- ✅ Saved artifacts:
  - trained pipeline model (`.pkl`)  
  - evaluation metrics (`.json`)  
  - best threshold for deployment  

---

## Results (Replace with your final metrics)
> Add your final numbers after running the full pipeline.

- Best Binary Model: **AdaBoost**
- Validation F1 (default class): **0.5**
- Test F1 (default class): **0.489**
- Test ROC-AUC: **0.88**
- Default threshold: **0.42**
- Recall on risky loans (B/D): **0.8**
- Test_Accuray: **0.81**
---

## How to Run the Project

### 1) Load raw dataset into MySQL
- CSVs stored under: `data/raw/`
- ingestion script loads CSV → MySQL tables

### 2) Run SQL ETL pipeline
- SQL scripts apply schema fixes, cleaning, feature engineering
- generates the final master table: `loan_master`

### 3) Run training pipeline
- fetches `loan_master`
- trains multiple models
- selects best model with validation threshold tuning
- saves final artifacts

---

## Screenshots / Dashboard Preview
- dashboard creation in processing  
