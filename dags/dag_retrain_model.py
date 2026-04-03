"""Airflow DAG: Weekly model retraining.

Retrains the XGBoost baseline model using the latest data from
the Gold layer. Logs experiment to MLflow and saves the best model.
Runs every Sunday at 20:00 UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import DAG, task


default_args = {
    "owner": "soybean-oil-predictor",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
}


@task
def run_model_retraining():
    """Retrain XGBoost baseline on latest gold-layer data."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.config import CLEAN_DATA, TARGET_COLUMN
    from src.log import setup_logging
    from src.models.baseline import save_baseline_model, train_xgboost_baseline

    setup_logging("INFO")

    df = pd.read_parquet(CLEAN_DATA)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    result = train_xgboost_baseline(X_train, y_train, X_test, y_test)
    save_baseline_model(result["model"])

    return f"Model retrained. RMSE: {result['metrics']['rmse']:.4f}"


@task
def run_drift_check():
    """Check for data drift after retraining."""
    import pandas as pd

    from src.config import CLEAN_DATA
    from src.log import setup_logging
    from src.monitoring.drift import check_drift

    setup_logging("INFO")

    df = pd.read_parquet(CLEAN_DATA)
    n = len(df)
    split = int(n * 0.8)
    reference = df.iloc[:split]
    current = df.iloc[split:]

    result = check_drift(reference, current)
    return f"Drift detected: {result['drift_detected']}, features: {result['drifted_features']}"


with DAG(
    dag_id="retrain_model",
    default_args=default_args,
    description="Weekly model retraining with drift monitoring",
    schedule="0 20 * * 0",  # Sundays at 20:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "mlops", "monitoring"],
):
    run_model_retraining() >> run_drift_check()
