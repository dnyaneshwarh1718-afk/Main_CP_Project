import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.pipeline.preprocessing import build_preprocessor
from src.modeling.model_registry import get_models
from src.config.settings import (
    FEATURE_COLUMNS,
    TARGET_COL,
    RANDOM_STATE,
    MODEL_DIR,
    REPORTS_DIR
)

# ✅ Status mapping for multiclass
MULTICLASS_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


def train_multiclass_models(df: pd.DataFrame):
    df = df.copy()

    # ✅ Validate columns
    required_cols = FEATURE_COLUMNS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in loan_master: {missing}")

    df = df[required_cols]
    df = df.dropna(subset=[TARGET_COL])

    # ✅ Target encoding (4 classes)
    df["target_multiclass"] = df[TARGET_COL].map(MULTICLASS_MAP)
    df = df.dropna(subset=["target_multiclass"])
    df["target_multiclass"] = df["target_multiclass"].astype(int)

    X = df[FEATURE_COLUMNS]
    y = df["target_multiclass"]

    # ✅ Train/Val/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    preprocessor, _, _ = build_preprocessor(X_train)
    models = get_models(random_state=RANDOM_STATE)

    # ✅ Remove models that don't support multiclass properly
    # (You can keep more later, but this is safe)
    allowed_models = {}
    for name, model in models.items():
        if name in ["GaussianNB", "RandomForest", "ExtraTrees", "GradientBoosting", "AdaBoost", "XGBoost", "LightGBM", "LogisticRegression"]:
            allowed_models[name] = model

    results = []
    best_model_name = None
    best_pipeline = None
    best_val_f1 = -1

    for name, model in allowed_models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)

        row = {
            "model": name,
            "val_accuracy": accuracy_score(y_val, val_pred),
            "val_f1_macro": f1_score(y_val, val_pred, average="macro"),
            "val_f1_weighted": f1_score(y_val, val_pred, average="weighted"),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_f1_macro": f1_score(y_test, test_pred, average="macro"),
            "test_f1_weighted": f1_score(y_test, test_pred, average="weighted"),
            "Classification_report": classification_report(y_test, test_pred),
        }

        results.append(row)

        #  Select best using VAL macro F1 (fair multiclass metric)
        if row["val_f1_macro"] > best_val_f1:
            best_val_f1 = row["val_f1_macro"]
            best_model_name = name
            best_pipeline = pipeline

    #  Save best multiclass model
    best_model_path = MODEL_DIR / "best_multiclass_pipeline.pkl"
    joblib.dump(best_pipeline, best_model_path)

    #  Save report
    report_path = REPORTS_DIR / "multiclass_model_metrics.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "best_val_macro_f1": best_val_f1,
                "results": results
            },
            f,
            indent=4
        )

    return best_model_name, best_model_path, report_path
