"""Data drift detection using Evidently AI.

Monitors whether incoming prediction data has drifted from the
training distribution — an early warning that model performance
may degrade before it shows up in prediction errors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.config import MONITORING_FOLDER, TARGET_COLUMN
from src.log import get_logger

logger = get_logger(__name__)


def build_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    prediction_col: str | None = None,
) -> Report:
    """Generate an Evidently data drift report.

    Args:
        reference_data: Training/reference dataset.
        current_data: New/production data to check for drift.
        target_col: Name of the target column.
        prediction_col: Name of the prediction column (if available).

    Returns:
        Evidently Report object.
    """
    column_mapping = ColumnMapping()
    column_mapping.target = target_col if target_col in reference_data.columns else None
    column_mapping.prediction = prediction_col

    numerical_features = [
        c for c in reference_data.select_dtypes(include="number").columns
        if c not in (target_col, prediction_col)
    ]
    column_mapping.numerical_features = numerical_features

    presets = [DataDriftPreset()]
    if column_mapping.target:
        presets.append(TargetDriftPreset())

    report = Report(metrics=presets)
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    logger.info("drift_report_generated", n_features=len(numerical_features))
    return report


def save_drift_report(
    report: Report,
    output_dir: Path | None = None,
    filename: str = "data_drift_report.html",
) -> Path:
    """Save drift report as HTML.

    Args:
        report: Evidently Report object.
        output_dir: Output directory. Defaults to reports/monitoring/.
        filename: Output filename.

    Returns:
        Path to the saved report.
    """
    output_dir = output_dir or MONITORING_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    report.save_html(str(output_path))
    logger.info("drift_report_saved", path=str(output_path))
    return output_path


def check_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> dict:
    """Quick drift check returning a boolean summary.

    Args:
        reference_data: Training/reference dataset.
        current_data: New/production data.

    Returns:
        Dictionary with drift detection results.
    """
    report = build_drift_report(reference_data, current_data)
    report_dict = report.as_dict()

    metrics = report_dict.get("metrics", [])
    drift_detected = False
    drifted_features = []

    for metric in metrics:
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            for col_name, col_data in result["drift_by_columns"].items():
                if col_data.get("drift_detected", False):
                    drift_detected = True
                    drifted_features.append(col_name)

    summary = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "n_drifted": len(drifted_features),
    }

    logger.info("drift_check", **summary)
    return summary
