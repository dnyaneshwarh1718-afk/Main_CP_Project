from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

SQL_DIR = PROJECT_ROOT / "SQL"
OITPUTS_DIR = PROJECT_ROOT / "outputs"

LOG_DIR = OITPUTS_DIR / "logs"
MODEL_DIR = OITPUTS_DIR / "models"
REPORTS_DIR = OITPUTS_DIR / "reports"
PREDICTIONS_DIR = OITPUTS_DIR / "predictions"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)

for folder in [RAW_DATA_DIR, LOG_DIR, MODEL_DIR, REPORTS_DIR]:
    folder.mkdir(parents = True, exist_ok = True)

Master_TABLE_NAME = "loan_master"

SQL_FILES = [
    SQL_DIR  / "1_raw_schema_fix_and_index.sql",
    SQL_DIR  / "2_data_cleaning_views.sql",
    SQL_DIR / "3_data_validation.sql",
    SQL_DIR / "4_feature_engineering_aggregations.sql",
    SQL_DIR / "5_master_table.sql",
]

# ML Config
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "Loan_Amount", "Duration", "Payments",
    "Region", "Avg_Salary", "Avg_Unemployment_Rate", "Entrepreneurs_Per_1000",
    "Txn_Count", "Avg_Balance", "Balance_Volatility", "Net_Cashflow",
    "Is_Joint_Account", "Has_Card", "Gold_Card_Count",
    "Total_Order_Amount", "Avg_Trans_Amount"
]
TARGET_COL = "Status"

BINARY_STATUS_MAP = {"A":0, "C":0, "B":1, "D": 1}
