import json
import joblib
import pandas as pd
import numpy as np

from src.features.fetch_master import fetch_master_table
from src.config.settings import (
    FEATURE_COLUMNS,
    MODEL_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR
)


def assign_risk(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


def assign_action(risk):
    if risk == "Low Risk":
        return "Auto Approve"
    elif risk == "Medium Risk":
        return "Manual Review"
    else:
        return "Reject / Monitor"


def run_inference():
    # Load data
    df = fetch_master_table()
    X = df[FEATURE_COLUMNS]

    # Load model
    model = joblib.load(MODEL_DIR / "best_model_pipeline.pkl")

    # Load threshold
    with open(REPORTS_DIR / "best_model_metrics.json") as f:
        metrics = json.load(f)

    threshold = metrics["best_threshold"]

    # Predict
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Build output
    output = df.copy()
    output["default_probability"] = probs.round(4)
    output["predicted_default"] = preds
    output["risk_label"] = output["default_probability"].apply(assign_risk)
    output["suggested_action"] = output["risk_label"].apply(assign_action)

    # Save for Power BI
    output_path = PREDICTIONS_DIR / "loan_scoring_output.csv"
    output.to_csv(output_path, index=False)

    print(f" Inference completed. File saved at: {output_path}")
