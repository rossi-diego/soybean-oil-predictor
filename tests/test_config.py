"""Unit tests for src.config path definitions."""

import joblib

from src.config import (
    DATA_FOLDER,
    IMAGE_FOLDER,
    LINEAR_REGRESSION_MODEL,
    MODEL_FOLDER,
    REPORT_FOLDER,
    TICKERS,
    TARGET_COLUMN,
    Settings,
)


def test_data_folder_exists():
    """DATA_FOLDER must exist on disk."""
    assert DATA_FOLDER.exists(), f"Missing directory: {DATA_FOLDER}"


def test_model_folder_exists():
    """MODEL_FOLDER must exist on disk."""
    assert MODEL_FOLDER.exists(), f"Missing directory: {MODEL_FOLDER}"


def test_report_folder_exists():
    """REPORT_FOLDER must exist on disk."""
    assert REPORT_FOLDER.exists(), f"Missing directory: {REPORT_FOLDER}"


def test_image_folder_exists():
    """IMAGE_FOLDER must exist on disk."""
    assert IMAGE_FOLDER.exists(), f"Missing directory: {IMAGE_FOLDER}"


def test_model_file_exists():
    """The trained model file must exist at the configured path."""
    assert LINEAR_REGRESSION_MODEL.exists(), (
        f"Model file not found: {LINEAR_REGRESSION_MODEL}. "
        "Run notebooks/02-linear_regression.ipynb to generate it."
    )


def test_model_file_loadable():
    """joblib.load must successfully deserialise the trained pipeline."""
    model = joblib.load(LINEAR_REGRESSION_MODEL)
    assert model is not None


def test_tickers_dict_not_empty():
    """TICKERS must contain at least the core commodity contracts."""
    assert len(TICKERS) >= 6
    assert "boc1" in TICKERS


def test_target_column():
    """Target column must be boc1."""
    assert TARGET_COLUMN == "boc1"


def test_settings_defaults():
    """Settings must have sensible defaults without .env file."""
    settings = Settings()
    assert settings.api_port == 8000
    assert settings.mlflow_experiment_name == "soybean-oil-predictor"
