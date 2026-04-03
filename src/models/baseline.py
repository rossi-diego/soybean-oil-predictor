"""XGBoost baseline model.

Per stack-reference.md: "Always start with XGBoost baseline. Move to deep
learning only if it meaningfully outperforms on out-of-sample data."

XGBoost is the industry standard for tabular data in quantitative finance.
It handles nonlinear relationships, feature interactions, and missing values
without explicit preprocessing — making it the strongest baseline.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.config import MODEL_FOLDER, Settings
from src.log import get_logger
from src.models.evaluation import evaluate_regression

logger = get_logger(__name__)

RANDOM_STATE = 42

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}


def train_xgboost_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
    log_to_mlflow: bool = True,
) -> dict:
    """Train an XGBoost baseline model with MLflow tracking.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        params: XGBoost hyperparameters. Defaults to :data:`DEFAULT_PARAMS`.
        log_to_mlflow: Whether to log experiment to MLflow.

    Returns:
        Dictionary with keys: model, metrics, feature_importance.
    """
    params = params or DEFAULT_PARAMS
    model = XGBRegressor(**params)

    logger.info("training_xgboost", params=params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)

    importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info("xgboost_trained", **metrics)

    if log_to_mlflow:
        settings = Settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="xgboost_baseline"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, artifact_path="model")

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
    }


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    params: dict | None = None,
) -> pd.DataFrame:
    """Time-series aware walk-forward validation.

    Unlike KFold, this respects temporal ordering — training always
    uses past data to predict future data, preventing look-ahead bias.

    Args:
        X: Feature matrix (must be time-sorted).
        y: Target vector (must be time-sorted).
        n_splits: Number of train/test splits.
        params: XGBoost parameters.

    Returns:
        DataFrame with columns [fold, metric, value] for each split.
    """
    params = params or DEFAULT_PARAMS
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)

        for metric_name, value in metrics.items():
            results.append({"fold": fold, "metric": metric_name, "value": value})

        logger.info("walk_forward_fold", fold=fold, **metrics)

    return pd.DataFrame(results)


def save_baseline_model(model: XGBRegressor, path: Path | None = None) -> Path:
    """Persist the trained XGBoost model to disk.

    Args:
        model: Fitted XGBRegressor.
        path: Output path. Defaults to models/xgboost_baseline.joblib.

    Returns:
        Path where the model was saved.
    """
    path = path or MODEL_FOLDER / "xgboost_baseline.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("model_saved", path=str(path))
    return path
