from src.ingestion.load_csv_to_mysql import load_all_raw_csvs
from src.pipeline.run_sql_etl import run_full_sql_pipeline
from src.pipeline.train_best_pipeline import run_best_model_pipeline

if __name__ == "__main__":
    print("Loading RAW CSVs into MySQL server..")
    load_all_raw_csvs()

    print("Running SQL ETL Pipeline")
    run_full_sql_pipeline()

    print("Running Best Model Training Pipeline")
    run_best_model_pipeline()

    print("Done ")
