import mlflow
import mlflow_tracking

def start_mlflow(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()

def log_params(params: dict):
    for k,v in params.items():
        mlflow.log_param(k,v)

def log_metrics(metrics: dict):
    for k,v in metrics.items():
        mlflow.log_metrics(k,v)

def log_model(model, artifact_path: str = "model"):
    mlflow.sklearn.log_model(model, artifact_path = artifact_path)
