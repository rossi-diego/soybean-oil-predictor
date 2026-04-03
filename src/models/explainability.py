"""Model explainability via SHAP (SHapley Additive exPlanations).

Per stack-reference.md: "Explainability — always include in risk models."
SHAP values show how each feature contributes to individual predictions,
which is critical for risk managers who need to understand *why* a model
produces a given forecast.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.log import get_logger

logger = get_logger(__name__)


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> shap.Explanation:
    """Compute SHAP values for a trained model.

    Uses TreeExplainer for tree-based models (XGBoost, LightGBM)
    and KernelExplainer as fallback for other model types.

    Args:
        model: Fitted model or pipeline.
        X: Feature matrix to explain.
        max_samples: Maximum samples to explain (for performance).

    Returns:
        SHAP Explanation object.
    """
    X_sample = X.head(max_samples) if len(X) > max_samples else X

    try:
        explainer = shap.TreeExplainer(model)
        logger.info("using_tree_explainer")
    except Exception:
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        logger.info("using_kernel_explainer")

    shap_values = explainer(X_sample)
    logger.info("shap_computed", n_samples=len(X_sample), n_features=X_sample.shape[1])
    return shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a SHAP summary (beeswarm) plot.

    Shows feature importance and direction of effect for each feature.

    Args:
        shap_values: SHAP Explanation object.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    fig = plt.gcf()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("shap_summary_saved", path=str(save_path))

    return fig


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    sample_idx: int = 0,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a SHAP waterfall plot for a single prediction.

    Shows how each feature pushes the prediction from the base value
    to the final prediction — ideal for explaining individual forecasts
    to traders and risk managers.

    Args:
        shap_values: SHAP Explanation object.
        sample_idx: Index of the sample to explain.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    fig = plt.gcf()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("shap_waterfall_saved", path=str(save_path))

    return fig


def get_feature_importance_from_shap(
    shap_values: shap.Explanation,
) -> pd.DataFrame:
    """Extract global feature importance from SHAP values.

    Args:
        shap_values: SHAP Explanation object.

    Returns:
        DataFrame with columns [feature, importance] sorted descending.
    """
    importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return df.reset_index(drop=True)
