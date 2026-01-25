import pandas as pd
from src.db.mysql_connection import get_mysql_connection
from src.config.settings import Master_TABLE_NAME

def fetch_master_table():
    conn = get_mysql_connection()
    df = pd.read_sql(f"SELECT * FROM {Master_TABLE_NAME};",conn)
    conn.close()
    return df
