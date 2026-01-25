import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from src.config.settings import RAW_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("load_csv_to_mysql")
load_dotenv()

def get_sqlalchemy_engine():
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    url = f"mysql+mysqlconnector: //{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)

def load_one_csv(file_path: str, table_name: str):
    logger.info(f"Loading CSV: {file_path} -> Table: {table_name}")

    df = pd.read_csv(file_path)

    # Standard cleaning
    df.columns = [c.strip().lower() for c in df.columns]

    engine = get_sqlalchemy_engine()

    # Replace = fresh load
    df.to_sql(table_name, con=engine, if_exists="replace", index=False, chunksize= 5000)

    logger.info(f"Loaded {len(df)} rows into `{table_name}`")

def load_all_raw_csvs():
    """
    This assumes raw csv file name = table name
    Ex: account.csv -> account table
    """
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No csv files found inside data/raw")
    
    for f in csv_files:
        table_name = f.stem.lower()
        load_one_csv(str(f), table_name)

    logger.info("All csv files loaded into MySQL")
