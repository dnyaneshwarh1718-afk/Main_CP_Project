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
    recall_score
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


# -----------------------------
# Threshold tuning (Validation)
# -----------------------------
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


# -----------------------------
# Training pipeline
# -----------------------------
def train_all_models_and_select_best(df: pd.DataFrame):

    df = df.copy()

    # -----------------------------
    # 1. Schema validation
    # -----------------------------
    required_cols = FEATURE_COLUMNS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_cols]

    # -----------------------------
    # 2. Binary target
    # -----------------------------
    df["target_binary"] = df[TARGET_COL].map(BINARY_STATUS_MAP)
    df = df.dropna(subset=["target_binary"])
    df["target_binary"] = df["target_binary"].astype(int)

    X = df[FEATURE_COLUMNS]
    y = df["target_binary"]

    # -----------------------------
    # 3. Train / Val / Test split
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # -----------------------------
    # 4. Models
    # -----------------------------
    models = get_models(random_state=RANDOM_STATE)

    all_model_results = []
    best_model_name = None
    best_pipeline = None
    best_threshold = None
    best_val_f1_global = -1

    # -----------------------------
    # 5. Train + tune models
    # -----------------------------
    for name, base_model in models.items():

        # fresh preprocessor per model
        preprocessor, _, _ = build_preprocessor(X_train)

        # ---- ONLY LogisticRegression is tuned ----
        if name == "LogisticRegression":

            model = LogisticRegression(
            solver="saga",
            max_iter=3000,
            random_state=RANDOM_STATE
        )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            param_grid = {
                "model__C": [0.01, 0.1, 1, 5, 10],
                "model__l1_ratio": [0, 1]  # 0=L2, 1=L1
            }

            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring="f1",
                cv=5,
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            pipeline = search.best_estimator_

        else:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", base_model)
            ])
            pipeline.fit(X_train, y_train)

        # -----------------------------
        # Validation evaluation
        # -----------------------------
        if not hasattr(pipeline, "predict_proba"):
            continue  # skip unsafe models

        val_prob = pipeline.predict_proba(X_val)[:, 1]
        best_t, val_f1 = find_best_threshold(y_val, val_prob)

        all_model_results.append({
            "model": name,
            "val_f1": val_f1,
            "best_threshold_val": best_t
        })

        # -----------------------------
        # Model selection (VALIDATION ONLY)
        # -----------------------------
        if val_f1 > best_val_f1_global:
            best_val_f1_global = val_f1
            best_model_name = name
            best_pipeline = pipeline
            best_threshold = best_t

    # -----------------------------
    # 6. Final TEST evaluation (ONCE)
    # -----------------------------
    test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    test_metrics = {
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred),
        "test_auc": roc_auc_score(y_test, test_prob),
        "recall_risky_(B/D=1)": recall_score(y_test, test_pred, pos_label=1)
    }

    # -----------------------------
    # 7. Save artifacts
    # -----------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_model_pipeline.pkl"
    joblib.dump(best_pipeline, model_path)

    metrics_path = REPORTS_DIR / "best_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "best_threshold": best_threshold,
                "validation_f1": best_val_f1_global,
                "test_metrics": test_metrics,
                "all_models_validation": all_model_results,
                "note": "Hyperparameters tuned on TRAIN, threshold on VAL, test used once."
            },
            f,
            indent=4
        )

    return best_model_name, model_path, metrics_path
