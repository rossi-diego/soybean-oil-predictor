"""Project-wide path and settings configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_FOLDER = PROJECT_FOLDER / "data"
DATA_BRONZE = DATA_FOLDER / "bronze"
DATA_SILVER = DATA_FOLDER / "silver"
DATA_GOLD = DATA_FOLDER / "gold"

MODEL_FOLDER = PROJECT_FOLDER / "models"
REPORT_FOLDER = PROJECT_FOLDER / "reports"
IMAGE_FOLDER = REPORT_FOLDER / "images"
MONITORING_FOLDER = REPORT_FOLDER / "monitoring"

# ---------------------------------------------------------------------------
# Legacy data files (kept for backward compatibility during migration)
# ---------------------------------------------------------------------------
RAW_DATA = DATA_FOLDER / "commodities_raw_data.csv"
CLEAN_DATA = DATA_FOLDER / "commodities_clean_data.parquet"
RESULTS_DATA = DATA_FOLDER / "model_comparison_results.parquet"
FEATURES_DATA = DATA_FOLDER / "features_describe.csv"

# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------
LINEAR_REGRESSION_MODEL = MODEL_FOLDER / "linear_regression.joblib"
XGBOOST_MODEL = MODEL_FOLDER / "xgboost_baseline.joblib"

# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------
DUCKDB_PATH = DATA_FOLDER / "soybean_oil.duckdb"

# ---------------------------------------------------------------------------
# Ticker symbols — CBOT / ICE / Bursa Malaysia
# ---------------------------------------------------------------------------
TICKERS = {
    "boc1": "ZL=F",       # Soybean oil front-month (CBOT)
    "sc1": "ZS=F",        # Soybean front-month (CBOT)
    "smc1": "ZM=F",       # Soybean meal front-month (CBOT)
    "zc1": "ZC=F",        # Corn front-month (CBOT)
    "lcoc1": "BZ=F",      # Brent crude front-month (ICE)
    "hoc1": "HO=F",       # Heating oil front-month (NYMEX)
    "rsc1": "ZW=F",       # Wheat front-month (CBOT)
}
# NOTE: Palm oil (fcpoc1) removed — PALM.L returns GBp ETF price (~78),
# but training data uses MYR/tonne (~3000-7000). No reliable yfinance source
# for FCPO in the correct unit. Model retrained without it.

TARGET_COLUMN = "boc1"

FEATURE_COLUMNS = [
    "smc1", "sc1", "lcoc1", "hoc1", "rsc1",
]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_log_level: str = "info"

    usda_api_key: str = ""
    fred_api_key: str = ""

    duckdb_path: str = str(DUCKDB_PATH)

    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "soybean-oil-predictor"

    evidently_report_path: str = str(MONITORING_FOLDER)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
