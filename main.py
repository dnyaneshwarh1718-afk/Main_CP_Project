from src.ingestion.load_csv_to_mysql import load_all_raw_csvs
from src.pipeline.run_sql_etl import run_full_sql_pipeline

if __name__ == '__main__':
    print("Loading RAW CSVs into MySQL server..")
    load_all_raw_csvs()

    print("Running SQL ETL Pipeline")
    run_full_sql_pipeline()

    print("Done. Master table ready: loan_master")

    