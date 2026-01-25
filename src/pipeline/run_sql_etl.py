from src.config.settings import SQL_FILES
from src.utils.sql_runner import run_sql_file
from src.utils.logger import setup_logger

logger = setup_logger("run_sql_etl")

def run_full_sql_pipeline():
    logger.info("Starting SQl Pipeline")

    for f in SQL_FILES:
        run_sql_file(f)

    logger.info("SQL pipeline completed")

    