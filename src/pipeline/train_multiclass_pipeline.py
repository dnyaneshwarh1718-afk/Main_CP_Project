from src.features.fetch_master import fetch_master_table
from src.modeling.train_multiclass_model import train_multiclass_models
from src.utils.logger import setup_logger

logger = setup_logger("train_multiclass_pipeline")

def run_multiclass_pipeline():
    logger.info("Fetching loan_master for multiclass training...")
    df = fetch_master_table()

    logger.info("Training multiclass models (A/B/C/D)...")
    best_name, best_path, report_path = train_multiclass_models(df)

    logger.info(f"Best Multiclass Model: {best_name}")
    logger.info(f"Saved: {best_path}")
    logger.info(f"Report: {report_path}")
