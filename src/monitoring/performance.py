"""Model performance monitoring.

Tracks prediction quality over time to detect model degradation.
When the model's out-of-sample performance drops below thresholds,
it triggers a retraining signal.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import RegressionPreset
from evidently.report import Report

from src.config import MONITORING_FOLDER, TARGET_COLUMN
from src.log import get_logger

logger = get_logger(__name__)

PERFORMANCE_THRESHOLDS = {
    "mae_threshold": 5.0,
    "rmse_threshold": 7.0,
    "r2_threshold": 0.5,
}


def build_performance_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    prediction_col: str = "prediction",
) -> Report:
    """Generate a regression performance report.

    Args:
        reference_data: Historical predictions with actuals.
        current_data: Recent predictions with actuals.
        target_col: Name of the target column.
        prediction_col: Name of the prediction column.

    Returns:
        Evidently Report object.
    """
    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = prediction_col

    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    logger.info("performance_report_generated")
    return report


def save_performance_report(
    report: Report,
    output_dir: Path | None = None,
    filename: str = "model_performance_report.html",
) -> Path:
    """Save performance report as HTML.

    Args:
        report: Evidently Report object.
        output_dir: Output directory.
        filename: Output filename.

    Returns:
        Path to the saved report.
    """
    output_dir = output_dir or MONITORING_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    report.save_html(str(output_path))
    logger.info("performance_report_saved", path=str(output_path))
    return output_path


def check_model_health(
    y_true: pd.Series,
    y_pred: pd.Series,
    thresholds: dict | None = None,
) -> dict:
    """Check if model performance is within acceptable bounds.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        thresholds: Performance thresholds.

    Returns:
        Dictionary with health status and metrics.
    """
    from src.models.evaluation import evaluate_regression

    thresholds = thresholds or PERFORMANCE_THRESHOLDS
    metrics = evaluate_regression(y_true, y_pred)

    issues = []
    if metrics["mae"] > thresholds["mae_threshold"]:
        issues.append(f"MAE ({metrics['mae']:.2f}) exceeds threshold ({thresholds['mae_threshold']})")
    if metrics["rmse"] > thresholds["rmse_threshold"]:
        issues.append(f"RMSE ({metrics['rmse']:.2f}) exceeds threshold ({thresholds['rmse_threshold']})")
    if metrics["r2"] < thresholds["r2_threshold"]:
        issues.append(f"R² ({metrics['r2']:.2f}) below threshold ({thresholds['r2_threshold']})")

    result = {
        "healthy": len(issues) == 0,
        "metrics": metrics,
        "issues": issues,
        "retrain_recommended": len(issues) >= 2,
    }

    logger.info("model_health_check", **result)
    return result
