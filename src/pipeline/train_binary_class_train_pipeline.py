from src.features.fetch_master import fetch_master_table
from src.modeling.train_binary_class_model import train_all_models_and_select_best
from src.utils.logger import setup_logger

logger = setup_logger("train_best_pipeline")

def run_best_model_pipeline():
    logger.info("Fetching loan_master data...")
    df = fetch_master_table()
    logger.info(f"loan_master shape: {df.shape}")

    logger.info("Training multiple models and selecting best...")
    best_name, best_path, metrics_path = train_all_models_and_select_best(df)

    logger.info(f"Best Model: {best_name}")
    logger.info(f"Best Model Saved: {best_path}")
    logger.info(f"Metrics Saved: {metrics_path}")
