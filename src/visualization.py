"""Visualization utilities for the Soybean Oil Predictor project."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay
from sklearn.utils.validation import check_is_fitted

RANDOM_STATE = 42

sns.set_theme(palette="bright")

PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2


def plot_coefficients(df_coefs: pd.DataFrame, title: str = "Coefficients") -> plt.Figure:
    """Plot a horizontal bar chart of regression model coefficients.

    Args:
        df_coefs: DataFrame with a single column named ``coefficient`` and
            feature names as the index.
        title: Chart title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df_coefs.plot.barh(ax=ax)
    ax.set_title(title)
    ax.axvline(x=0, color=".5")
    ax.set_xlabel("Coefficients")
    ax.get_legend().remove()
    return fig


def plot_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> plt.Figure:
    """Plot residual diagnostics from pre-computed predictions.

    Three panels are produced: residual histogram, residuals vs predicted
    values, and actual vs predicted values.

    Args:
        y_true: Observed target values.
        y_pred: Predicted target values.

    Returns:
        matplotlib Figure with three diagnostic panels.
    """
    residual = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    sns.histplot(residual, kde=True, ax=axs[0])

    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    return fig


def plot_residual_estimator(
    estimator,
    X,
    y,
    eng_formatter: bool = False,
    sample_fraction: float = 0.25,
) -> plt.Figure:
    """Plot residual diagnostics for a fitted regression estimator.

    Args:
        estimator: A fitted scikit-learn estimator (pipeline or regressor).
        X: Feature matrix (DataFrame or array-like).
        y: Target vector (Series, DataFrame, or array-like).
        eng_formatter: If True, apply engineering notation to axis ticks.
        sample_fraction: Fraction of samples to use in scatter plots (0–1].

    Returns:
        matplotlib Figure with three panels: residual histogram,
        residuals vs predicted, and actual vs predicted.
    """
    # Verify the estimator is fitted
    check_is_fitted(estimator)

    # Normalise types and drop NaN rows simultaneously
    X_df = pd.DataFrame(X).reset_index(drop=True)

    if isinstance(y, pd.DataFrame):
        y_sr = y.iloc[:, 0].reset_index(drop=True)
    else:
        y_sr = pd.Series(np.ravel(y)).reset_index(drop=True)

    mask = y_sr.notna() & ~X_df.isna().any(axis=1)
    X_clean = X_df.loc[mask].reset_index(drop=True)
    y_true = y_sr.loc[mask].reset_index(drop=True)

    # Generate predictions and ensure 1-D output
    y_pred = estimator.predict(X_clean)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)
        else:
            raise ValueError(
                f"y_pred is multioutput with shape {y_pred.shape}. "
                f"Select a single output (e.g. y_pred[:, 0]) before plotting."
            )
    else:
        y_pred = y_pred.reshape(-1)

    y_true = np.asarray(y_true).reshape(-1)

    # Build figure with three diagnostic panels
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        ax=axs[1],
        subsample=sample_fraction,
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
    )

    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        ax=axs[2],
        subsample=sample_fraction,
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
    )

    residual = y_true - y_pred
    try:
        sns.histplot(residual, kde=True, ax=axs[0])
    except Exception:
        axs[0].hist(residual, bins=30)
        axs[0].set_title("Residuals distribution")

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    return fig


def plot_model_metrics_comparison(df_results: pd.DataFrame) -> plt.Figure:
    """Plot a 2×2 grid of boxplots comparing model performance metrics.

    Args:
        df_results: Long-form DataFrame produced by
            :func:`src.models.organize_cv_results`, with columns
            ``model``, ``time_seconds``, ``test_r2``,
            ``test_neg_mean_absolute_error``, and
            ``test_neg_root_mean_squared_error``.

    Returns:
        matplotlib Figure with four boxplot panels (Time, R², MAE, RMSE).
    """
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    metric_comparison = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    metric_names = [
        "Time (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metric, name in zip(axs.flatten(), metric_comparison, metric_names):
        sns.boxplot(
            x="model",
            y=metric,
            data=df_results,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(name)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", rotation=90)

    return fig
