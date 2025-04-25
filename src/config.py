from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_FOLDER / "data"

MODEL_FOLDER = PROJECT_FOLDER / "models"

REPORT_FOLDER = PROJECT_FOLDER / "reports"

IMAGE_FOLDER = REPORT_FOLDER / "images"


# Data files
RAW_DATA = DATA_FOLDER / "commodities_raw_data.csv"
CLEAN_DATA = DATA_FOLDER / "commodities_clean_data.parquet"
RESULTS_DATA = DATA_FOLDER / "model_comparison_results.parquet"
FEATURES_DATA = DATA_FOLDER / "features_describe.csv"

# Models files
LINEAR_REGRESSION_MODEL = MODEL_FOLDER / "linear_regression.joblib"
