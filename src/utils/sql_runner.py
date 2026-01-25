from src.db.mysql_connection import get_mysql_connection
from src.utils.logger import setup_logger

logger = setup_logger("sql_runner")

def run_sql_file(sql_path):
    logger.info(f"Running: {sql_path.name}")

    with open(sql_path, "r", encoding = "utf-8") as f:
        script = f.read()

    conn = get_mysql_connection()
    cursor = conn.cursor()

    statements = [s.strip() for s in script.split(";") if s.strip()]

    for stmt in statements:
        cursor.execute(stmt)

    conn.commit()
    cursor.close()
    conn.close()

    logger.info(f"Done: {sql_path.name}")
    