# 📊 Loan Risk Analytics – Power BI Dashboard Suite

## Overview
This Power BI solution provides an **end-to-end loan risk monitoring and decision-support system** built on top of machine learning outputs.  
The dashboards are designed for **credit risk teams, policy makers, and collections teams** to monitor portfolio health, identify high-risk loans, and take data-driven actions.

The focus is on **business interpretability, risk concentration, and operational prioritization**, not just visualization.

---

## Data Source
- **Dataset:** Czech Bank Loan Dataset  
- **Input to Power BI:** Model output tables containing:
  - Loan-level default probability
  - Expected loss
  - Risk bucket classification
  - Actual loan status
  - Regional and tenure attributes

---

## Dashboard Structure

### 1️⃣ Loan Risk Monitoring – Current Portfolio Status
**Objective:** Provide an executive snapshot of current portfolio health.

**Key KPIs:**
- Total Loans
- Total Loan Amount
- Default Loans (B/D)
- Default Rate (%)
- Average Loan Amount
- Average Loan Duration

**Key Visuals:**
- Loan status distribution (Paid / Running / Default / In Debt)
- Default rate by region
- Defaulted loan amount by status

**Business Value:**
- Identifies realized credit risk
- Highlights regional performance variation
- Reveals default concentration within the portfolio

---

### 2️⃣ Portfolio Risk & Exposure Monitoring
**Objective:** Detect forward-looking risk using model predictions.

**Key KPIs:**
- Risky Loan Count
- % Loans Flagged for Review
- Total Expected Loss
- False Positive Cost
- Loss Avoided (model impact)

**Key Visuals:**
- Running loans at risk by status
- Predicted default rate by region
- Expected loss by loan status
- Top high-risk loans (loan-level table)

**Business Value:**
- Enables proactive intervention before default
- Quantifies risk in monetary terms
- Supports risk-based portfolio segmentation

---

### 3️⃣ Credit Policy Threshold & Model Justification
**Objective:** Justify model usage and credit policy thresholds.

**Key KPIs:**
- Model Used
- Selected Probability Threshold
- Recall (Actual Defaults Captured)
- ROC-AUC

**Key Visuals:**
- Default probability distribution
- Risk probability vs loan exposure
- Expected loss vs policy threshold
- Model-based risk classification

**Business Value:**
- Demonstrates transparent, explainable decision logic
- Shows trade-off between recall and operational cost
- Supports governance and audit requirements

---

### 4️⃣ Operational Risk Actions & Collections Prioritization
**Objective:** Convert risk insights into operational actions.

**Key KPIs:**
- High-Risk Running Loans
- High-Risk Exposure
- Expected Loss (Running High Risk)
- Average Tenure of Risky Loans

**Key Visuals:**
- High-risk loans by region and risk bucket
- Expected loss concentration by loan tenure
- Priority action table for collections teams

**Business Value:**
- Directs collections efforts to high-impact accounts
- Reduces wasted operational effort
- Improves recovery effectiveness

---

## Key Measures & Logic
- **Default Rate (%)** = Default Loans / Total Loans  
- **Expected Loss** = Default Probability × Loan Amount  
- **Risk Buckets** defined using optimized probability thresholds  
- **Loss Avoided** estimated based on proactive risk identification

---

## Design Principles
- Business-first KPI design
- Clear risk segmentation
- Minimal clutter, maximum interpretability
- Decision-oriented storytelling

---

## Intended Users
- Credit Risk Managers
- Portfolio Risk Analysts
- Credit Policy Teams
- Collections & Recovery Teams

---

## Outcome
This Power BI dashboard suite transforms model outputs into **actionable credit risk intelligence**, enabling better policy decisions, improved portfolio monitoring, and focused operational execution.
