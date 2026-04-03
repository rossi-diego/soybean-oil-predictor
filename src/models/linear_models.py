"""Linear regression model family (Ridge, Lasso, ElasticNet).

These models serve as interpretable benchmarks. While XGBoost typically
outperforms on raw accuracy, linear models provide:
- Coefficient interpretability (which features drive the price)
- Regularization insights (which features are noise vs signal)
- A simpler baseline for comparison
"""

from __future__ import annotations

import mlflow
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.config import Settings
from src.log import get_logger
from src.models.evaluation import evaluate_regression

logger = get_logger(__name__)

RANDOM_STATE = 42

LINEAR_MODELS = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "lasso": Lasso(alpha=0.1, random_state=RANDOM_STATE),
    "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE),
}


def build_linear_pipeline(
    regressor,
    scaler=None,
) -> Pipeline:
    """Build a pipeline with optional scaling + linear regressor.

    Args:
        regressor: scikit-learn linear regressor.
        scaler: Feature scaler. Defaults to RobustScaler.

    Returns:
        Fitted-ready Pipeline.
    """
    scaler = scaler or RobustScaler()
    return Pipeline([("scaler", scaler), ("reg", regressor)])


def train_all_linear_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_to_mlflow: bool = True,
) -> dict[str, dict]:
    """Train all linear model variants and compare results.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        log_to_mlflow: Whether to log to MLflow.

    Returns:
        Mapping of model name to {model, metrics, coefficients}.
    """
    results = {}

    for name, regressor in LINEAR_MODELS.items():
        logger.info("training_linear_model", model=name)
        pipeline = build_linear_pipeline(regressor)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)

        coefs = pipeline.named_steps["reg"].coef_
        coefficients = pd.DataFrame(
            {"feature": X_train.columns, "coefficient": coefs}
        ).sort_values("coefficient", ascending=False)

        results[name] = {
            "model": pipeline,
            "metrics": metrics,
            "coefficients": coefficients,
        }

        logger.info("linear_model_trained", model=name, **metrics)

        if log_to_mlflow:
            settings = Settings()
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)

            with mlflow.start_run(run_name=name):
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return results
