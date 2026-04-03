"""ML models package — baseline, linear, time-series, evaluation, explainability.

Backward-compatible re-exports from the original models.py via pipeline module.
"""

from src.models.pipeline import (
    build_regression_model_pipeline,
    grid_search_cv_regressor,
    organize_cv_results,
    train_and_validate_regression_model,
)

__all__ = [
    "build_regression_model_pipeline",
    "train_and_validate_regression_model",
    "grid_search_cv_regressor",
    "organize_cv_results",
]
