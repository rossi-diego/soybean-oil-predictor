"""Model evaluation metrics and comparison utilities.

All metrics follow the quant-reference.md standards:
- Always report on out-of-sample data
- Include directional accuracy (critical for trading)
- MAE, RMSE, R², MAPE where appropriate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.log import get_logger

logger = get_logger(__name__)


def evaluate_regression(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard regression metrics for model evaluation.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric name to value.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "directional_accuracy": float(directional_accuracy(y_true, y_pred)),
    }

    return metrics


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Percentage of correct up/down predictions.

    This is the most important metric for trading: even if the magnitude
    is wrong, getting the direction right enables profitable positioning.

    Args:
        y_true: Actual values (at least 2 observations).
        y_pred: Predicted values.

    Returns:
        Fraction of correctly predicted directions (0 to 1).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) < 2:
        return 0.0

    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    correct = (true_direction == pred_direction).sum()
    return correct / len(true_direction)


def compare_models(
    results: dict[str, dict],
) -> pd.DataFrame:
    """Build a comparison table from multiple model results.

    Args:
        results: Mapping of model name to dict containing 'metrics' key.

    Returns:
        DataFrame with models as rows and metrics as columns,
        sorted by RMSE ascending (best first).
    """
    rows = []
    for model_name, result in results.items():
        metrics = result["metrics"].copy()
        metrics["model"] = model_name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("rmse", ascending=True)

    logger.info("model_comparison", best_model=df.index[0], best_rmse=df["rmse"].iloc[0])
    return df
