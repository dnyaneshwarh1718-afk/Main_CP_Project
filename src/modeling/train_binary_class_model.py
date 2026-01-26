import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score
)
from src.pipeline.preprocessing import build_preprocessor
from src.modeling.model_registry import get_models
from src.config.settings import (
    FEATURE_COLUMNS,
    TARGET_COL,
    BINARY_STATUS_MAP,
    RANDOM_STATE,
    MODEL_DIR,
    REPORTS_DIR
)


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_f1 = 0.5, -1

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_t = t

    return float(best_t), float(best_f1)


def train_all_models_and_select_best(df: pd.DataFrame):
    df = df.copy()

    #  validate columns
    required_cols = FEATURE_COLUMNS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in loan_master: {missing}")

    df = df[required_cols]

    #  binary target
    df["target_binary"] = df[TARGET_COL].map(BINARY_STATUS_MAP)
    df = df.dropna(subset=["target_binary"])
    df["target_binary"] = df["target_binary"].astype(int)

    X = df[FEATURE_COLUMNS]
    y = df["target_binary"]

    #  Train/Val/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    #  Preprocessor (X only)
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    models = get_models(random_state=RANDOM_STATE)

    all_results = []
    best_model_name = None
    best_pipeline = None
    best_threshold = None
    best_test_f1 = -1

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        row = {"model": name}

        if hasattr(pipeline, "predict_proba"):
            #  tune threshold on VALIDATION
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            best_t, best_val_f1 = find_best_threshold(y_val, val_prob)

            #  evaluate on TEST using val threshold
            test_prob = pipeline.predict_proba(X_test)[:, 1]
            test_pred = (test_prob >= best_t).astype(int)

            #  Metrics
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred)
            test_auc = roc_auc_score(y_test, test_prob)
            risky_recall = recall_score(y_test, test_pred)  # recall for class 1

            row["best_threshold_val"] = best_t
            row["val_f1"] = best_val_f1
            row["test_accuracy"] = test_acc
            row["test_f1"] = test_f1
            row["test_auc"] = test_auc
            row["recall_risky_(B/D=1)"] = risky_recall

        else:
            # fallback (no proba)
            test_pred = pipeline.predict(X_test)

            row["best_threshold_val"] = None
            row["val_f1"] = None
            row["test_accuracy"] = accuracy_score(y_test, test_pred)
            row["test_f1"] = f1_score(y_test, test_pred)
            row["test_auc"] = None
            row["recall_risky_(B/D=1)"] = recall_score(y_test, test_pred)

        all_results.append(row)

        #  Best model selection based on TEST F1 (your current logic)
        if row["test_f1"] > best_test_f1:
            best_test_f1 = row["test_f1"]
            best_model_name = name
            best_pipeline = pipeline
            best_threshold = row["best_threshold_val"]

    #  Save best model pipeline
    best_model_path = MODEL_DIR / "best_model_pipeline.pkl"
    joblib.dump(best_pipeline, best_model_path)

    #  Save all results JSON
    metrics_path = REPORTS_DIR / "best_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "best_threshold": best_threshold,
                "best_test_f1": best_test_f1,
                "all_models": all_results,
                "note": "Threshold tuned on Validation set, evaluated on Test set"
            },
            f,
            indent=4
        )

    return best_model_name, best_model_path, metrics_path
