from src.ingestion.load_csv_to_mysql import load_all_raw_csvs
from src.pipeline.run_sql_etl import run_full_sql_pipeline
from src.pipeline.train_binary_class_train_pipeline import run_best_model_pipeline
from src.pipeline.train_multiclass_pipeline import run_multiclass_pipeline
from src.inference.run_inference import run_inference

if __name__ == "__main__":
    print("Loading RAW CSVs into MySQL server..")
    load_all_raw_csvs()

    print("Running SQL ETL Pipeline")
    run_full_sql_pipeline()

    print("Running Binary Best Model Training Pipeline")
    run_best_model_pipeline()

    print("Running Multiclass Training Pipeline (A/B/C/D)")
    run_multiclass_pipeline()

    print("Running inference pipeline...")
    run_inference()

    print("Done. Scored data ready for Power BI.")
