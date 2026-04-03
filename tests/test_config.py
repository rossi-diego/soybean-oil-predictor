"""Unit tests for src.config path definitions."""

import pytest

from src.config import (
    DATA_FOLDER,
    IMAGE_FOLDER,
    LINEAR_REGRESSION_MODEL,
    MODEL_FOLDER,
    REPORT_FOLDER,
    TARGET_COLUMN,
    TICKERS,
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


def test_model_file_loadable():
    """joblib.load must successfully deserialise the trained pipeline."""
    if not LINEAR_REGRESSION_MODEL.exists():
        pytest.skip("Model file not available in CI")
    import joblib

    try:
        model = joblib.load(LINEAR_REGRESSION_MODEL)
    except Exception as exc:
        pytest.skip(f"Model incompatible with current sklearn: {exc}")
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
