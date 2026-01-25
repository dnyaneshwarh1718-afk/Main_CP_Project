import sqlparse
from src.db.mysql_connection import get_mysql_connection
from src.utils.logger import setup_logger

logger = setup_logger("sql_runner")


def run_sql_file(sql_path):
    logger.info(f"Running: {sql_path.name}")

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    #  sqlparse safely splits SQL statements
    statements = [s.strip() for s in sqlparse.split(sql_script) if s.strip()]

    conn = get_mysql_connection()
    cursor = conn.cursor()

    try:
        for stmt in statements:
            cursor.execute(stmt)

             #  important for SELECT statements
            if cursor.with_rows:
                cursor.fetchall()

        conn.commit()
        logger.info(f" Completed: {sql_path.name}")

    except Exception as e:
        conn.rollback()
        logger.error(f" Failed: {sql_path.name}")
        logger.error(str(e))
        raise

    finally:
        cursor.close()
        conn.close()
