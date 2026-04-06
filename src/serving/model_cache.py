"""Global model cache — loaded once at app startup."""

from __future__ import annotations

import json

import joblib

from src.config import DATA_FOLDER, MODEL_FOLDER
from src.log import get_logger

logger = get_logger(__name__)

_model = None
_model_name: str = "none"


def load_models() -> None:
    """Load the champion model into memory.

    Reads model_metadata.json to determine which model is champion,
    then loads that specific joblib file. Falls back to any available
    model if metadata is missing.
    """
    global _model, _model_name

    # Try to load the champion from metadata
    metadata_path = DATA_FOLDER / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            champion = meta.get("champion", "ridge")
            champion_path = MODEL_FOLDER / f"{champion}.joblib"
            if champion_path.exists():
                _model = joblib.load(champion_path)
                _model_name = champion
                logger.info("champion_loaded", name=champion, path=str(champion_path))
                return
        except Exception:
            logger.exception("champion_load_failed")

    # Fallback: try any model file
    for name in ["ridge", "xgboost", "linear", "lasso", "elasticnet"]:
        path = MODEL_FOLDER / f"{name}.joblib"
        if path.exists():
            try:
                _model = joblib.load(path)
                _model_name = name
                logger.info("fallback_model_loaded", name=name, path=str(path))
                return
            except Exception:
                logger.exception("model_load_failed", name=name)

    logger.warning("no_model_available")


def get_model():
    """Return the cached model and its name."""
    return _model, _model_name
