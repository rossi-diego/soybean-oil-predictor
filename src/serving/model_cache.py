"""Global model cache — loaded once at app startup."""

from __future__ import annotations

import joblib

from src.config import LINEAR_REGRESSION_MODEL, XGBOOST_MODEL
from src.log import get_logger

logger = get_logger(__name__)

_model = None
_model_name: str = "none"


def load_models() -> None:
    """Load the best available model into memory.

    Called once during app lifespan startup. XGBoost preferred,
    linear regression as fallback.
    """
    global _model, _model_name

    if XGBOOST_MODEL.exists():
        try:
            _model = joblib.load(XGBOOST_MODEL)
            _model_name = "xgboost_baseline"
            logger.info("model_loaded", name=_model_name, path=str(XGBOOST_MODEL))
            return
        except Exception:
            logger.exception("xgboost_load_failed")

    if LINEAR_REGRESSION_MODEL.exists():
        try:
            _model = joblib.load(LINEAR_REGRESSION_MODEL)
            _model_name = "linear_regression"
            logger.info("model_loaded", name=_model_name, path=str(LINEAR_REGRESSION_MODEL))
            return
        except Exception:
            logger.exception("linear_model_load_failed")

    logger.warning("no_model_available")


def get_model():
    """Return the cached model and its name."""
    return _model, _model_name
