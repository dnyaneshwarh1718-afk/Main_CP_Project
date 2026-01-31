import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression

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

# =====================================================
# Threshold tuning using F1 (Validation only)
# =====================================================
def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_f1 = -1

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds)

        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return float(best_threshold), float(best_f1)


# =====================================================
# Training & Model Selection Pipeline
# =====================================================
def train_all_models_and_select_best(df: pd.DataFrame):

    df = df.copy()

    # -----------------------------
    # 1. Schema validation
    # -----------------------------
    required_cols = FEATURE_COLUMNS + [TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df[required_cols]

    # -----------------------------
    # 2. Target encoding
    # -----------------------------
    df["target_binary"] = df[TARGET_COL].map(BINARY_STATUS_MAP)
    df = df.dropna(subset=["target_binary"])
    df["target_binary"] = df["target_binary"].astype(int)

    X = df[FEATURE_COLUMNS]
    y = df["target_binary"]

    # -----------------------------
    # 3. Train / Validation / Test split
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # -----------------------------
    # 4. Model registry
    # -----------------------------
    models = get_models(random_state=RANDOM_STATE)

    best_model_name = None
    best_pipeline = None
    best_threshold = None
    best_val_f1 = -1

    all_model_results = []

    # -----------------------------
    # 5. Train + tune models
    # -----------------------------
    for model_name, base_model in models.items():

        preprocessor, _, _ = build_preprocessor(X_train)

        # ----- Logistic Regression (with tuning)
        if model_name == "LogisticRegression":

            lr = LogisticRegression(
                solver="saga",
                max_iter=3000,
                random_state=RANDOM_STATE
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", lr)
            ])

            param_grid = {
                "model__C": [0.01, 0.1, 1, 5, 10],
                "model__l1_ratio": [0, 1]
            }

            search = GridSearchCV(
                pipeline,
                param_grid,
                scoring="f1",
                cv=5,
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            pipeline = search.best_estimator_

        # ----- Other models
        else:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", base_model)
            ])
            pipeline.fit(X_train, y_train)

        # Skip models without probability output
        if not hasattr(pipeline, "predict_proba"):
            continue

        # -----------------------------
        # Validation evaluation
        # -----------------------------
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        threshold, val_f1 = find_best_threshold(y_val, val_prob)

        all_model_results.append({
            "model": model_name,
            "validation_f1": round(val_f1, 4),
            "best_threshold": round(threshold, 2)
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_name = model_name
            best_pipeline = pipeline
            best_threshold = threshold

    # -----------------------------
    # 6. Final TEST evaluation
    # -----------------------------
    test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    test_metrics = {
        "accuracy": round(accuracy_score(y_test, test_pred), 4),
        "f1": round(f1_score(y_test, test_pred), 4),
        "auc": round(roc_auc_score(y_test, test_prob), 4),
        "recall_default_(1)": round(recall_score(y_test, test_pred, pos_label=1), 4),
        "precision_default_(1)": round(precision_score(y_test, test_pred, pos_label=1), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

    # -----------------------------
    # 7. Save artifacts
    # -----------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_model_pipeline.pkl"
    metrics_path = REPORTS_DIR / "best_model_metrics.json"

    joblib.dump(best_pipeline, model_path)

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "best_threshold": round(best_threshold, 2),
                "validation_f1": round(best_val_f1, 4),
                "test_metrics": test_metrics,
                "all_models_validation": all_model_results,
                "note": "Hyperparameters tuned on TRAIN, threshold optimized on VAL, test used once."
            },
            f,
            indent=4
        )

    return best_model_name, model_path, metrics_path
