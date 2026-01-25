import joblib
import json
import mlflow
import mlflow.sklearn

import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.config.settings import(
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    BINARY_STATUS_MAP,
    MODEL_DIR,
    FEATURE_COLUMNS
)
from src.pipeline.preprocessing import build_preprocessor
from src.modeling.model_registry import get_models

def train_all_models_and_select_best(df: pd.DataFrame):
    df = df.copy()

    required_cols = FEATURE_COLUMNS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing colums in loan_master: {missing}")
    
    # selecting only required columns
    df = df[required_cols]

    # creating binary target
    df["target_binary"] = df[TARGET_COL].map(BINARY_STATUS_MAP)
    df = df.dropna(subset=["target_binary"])
    df["target_binary"] = df["target_binary"].astype(int)

    # deriving X features 
    x = df[FEATURE_COLUMNS]
    # deriving y variable
    y = df["target_binary"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,y,
        Test_size = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify = y
    )
        
        
        
    # preprocessing
    preprocessor, num_cols, cat_cols = build_preprocessor(x_train)

    # model
    models = get_models(random_state=RANDOM_STATE)

    best_model_name = None
    best_pipeline = None
    best_score = -1
    results = {}

    mlflow.set_experiment("Loan_Default_Binary_model_Comparison")

    with mlflow.start_run(run_name= "model_comparison"):
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            metrics = {"accuracy": acc, "f1_score": f1}

            try:
                y_prob = pipeline.predict_proba(x_test)[:,1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
            except:
                pass

            results[name] = metrics

            with mlflow.start_run(run_name = name, nested=True):
                mlflow.log_param("model", name)
                mlflow.log_param("test_size", TEST_SIZE)
                mlflow.log_param("random_state", RANDOM_STATE)
                mlflow.log_param("num_cols_count", len(num_cols))
                mlflow.log_param("cat_cols_count",len(cat_cols))

                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                mlflow.sklearn.load_model(pipeline, artifact_path = "model")

            
            if f1 > best_score:
                best_score = f1
                best_model_name = name
                best_pipeline = pipeline

        # saving best model locally
        best_model_path = MODEL_DIR / "best_binary_pipeline.pkl"
        joblib.dump(best_pipeline, best_model_path)

        # saving best model locally
        comparison_path = MODEL_DIR / "model_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(
                {
                "best_model": best_model_name,
                "best_f1_score": best_score,
                "all_results": results
            },
            f,
            indent=4
            )

        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_f1_score",best_score)
        mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")

    return best_model_name, best_model_path, comparison_path, results