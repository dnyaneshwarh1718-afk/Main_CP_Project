import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus

from src.config.settings import RAW_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("load_csv_to_mysql")

load_dotenv()


TABLE_MAP = {
    "account.csv": "account",
    "card.csv": "card",
    "client.csv": "client",
    "disp.csv": "disp",
    "district.csv": "district",
    "loan.csv": "loan",
    "orders.csv": "orders",
    "transaction_data.csv": "transaction_data",
}


def get_sqlalchemy_engine():
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "3306")
    db = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    password = quote_plus(password)

    if not all([host, db, user, password]):
        raise ValueError("Missing DB credentials in .env file")

    #  correct URL (no spaces)
    url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def load_one_csv(file_path: str, table_name: str):
    logger.info(f"Loading CSV -> MySQL | {os.path.basename(file_path)} --> {table_name}")

    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    engine = get_sqlalchemy_engine()

    #  replace = fresh load for pipeline repeatability
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi"
    )

    logger.info(f" Loaded {len(df)} rows into `{table_name}`")


def load_all_raw_csvs():
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {RAW_DATA_DIR}")

    for file_path in csv_files:
        file_name = file_path.name

        if file_name not in TABLE_MAP:
            logger.warning(f" Skipping unknown file: {file_name}")
            continue

        table_name = TABLE_MAP[file_name]
        load_one_csv(str(file_path), table_name)

    logger.info(" All raw CSV files loaded into MySQL successfully")
