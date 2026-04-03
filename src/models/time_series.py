"""Time-series specific models using Nixtla's statsforecast.

statsforecast provides fast, production-ready implementations of
classical time-series models (AutoARIMA, ETS, Theta) that respect
temporal structure — unlike cross-sectional models that ignore time ordering.
"""

from __future__ import annotations

import mlflow
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

from src.config import Settings
from src.log import get_logger
from src.models.evaluation import evaluate_regression

logger = get_logger(__name__)


def prepare_statsforecast_df(
    dates: pd.Series,
    values: pd.Series,
    unique_id: str = "boc1",
) -> pd.DataFrame:
    """Prepare data in the format required by statsforecast.

    statsforecast expects columns: unique_id, ds (timestamp), y (value).

    Args:
        dates: Date series.
        values: Target value series.
        unique_id: Identifier for the time series.

    Returns:
        DataFrame formatted for StatsForecast.
    """
    df = pd.DataFrame({
        "unique_id": unique_id,
        "ds": pd.to_datetime(dates),
        "y": values.values,
    })
    return df.sort_values("ds").reset_index(drop=True)


def train_statsforecast_models(
    df: pd.DataFrame,
    horizon: int = 21,
    freq: str = "B",
    log_to_mlflow: bool = True,
) -> dict:
    """Train multiple statistical forecasting models.

    Args:
        df: DataFrame with columns [unique_id, ds, y].
        horizon: Forecast horizon in periods (default 21 = 1 month of trading days).
        freq: Frequency string (B = business days).
        log_to_mlflow: Whether to log results to MLflow.

    Returns:
        Dictionary with keys: forecasts, models, cross_validation.
    """
    models = [
        AutoARIMA(season_length=21),
        AutoETS(season_length=21),
        AutoTheta(season_length=21),
    ]

    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)

    logger.info("training_statsforecast", n_models=len(models), horizon=horizon)

    sf.fit(df)
    forecasts = sf.predict(h=horizon)

    logger.info("statsforecast_trained", forecast_rows=len(forecasts))

    cv_results = sf.cross_validation(df=df, h=horizon, step_size=horizon, n_windows=5)

    if log_to_mlflow:
        settings = Settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        for model_name in ["AutoARIMA", "AutoETS", "AutoTheta"]:
            if model_name in cv_results.columns:
                y_true = cv_results["y"]
                y_pred = cv_results[model_name]
                metrics = evaluate_regression(y_true, y_pred)

                with mlflow.start_run(run_name=f"statsforecast_{model_name}"):
                    mlflow.log_metrics(metrics)
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("horizon", horizon)

                logger.info("statsforecast_cv", model=model_name, **metrics)

    return {
        "forecasts": forecasts,
        "sf": sf,
        "cross_validation": cv_results,
    }
