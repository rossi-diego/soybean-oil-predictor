"""Shared utility functions for the Soybean Oil Predictor project."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import LINEAR_REGRESSION_MODEL


def coefficients_dataframe(
    coefs: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """Build a sorted DataFrame of regression coefficients.

    Args:
        coefs: 1-D array of coefficient values.
        columns: Feature names corresponding to each coefficient.

    Returns:
        DataFrame with a single column ``coefficient`` and feature names as
        the index, sorted in ascending order by coefficient value.

    Raises:
        ValueError: If the lengths of ``coefs`` and ``columns`` differ.
    """
    if len(coefs) != len(columns):
        raise ValueError(
            f"Length mismatch: coefs has {len(coefs)} elements but "
            f"columns has {len(columns)} elements."
        )
    return pd.DataFrame(
        data=coefs, index=columns, columns=["coefficient"]
    ).sort_values(by="coefficient")


def load_model() -> Pipeline:
    """Load the trained regression pipeline from disk.

    Returns:
        Fitted scikit-learn Pipeline stored at
        :data:`src.config.LINEAR_REGRESSION_MODEL`.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the model cannot be deserialised (e.g. version
            mismatch between scikit-learn or numpy).
    """
    if not LINEAR_REGRESSION_MODEL.exists():
        raise FileNotFoundError(
            f"Model file not found: {LINEAR_REGRESSION_MODEL}. "
            "Run the training notebook to generate it."
        )
    try:
        return joblib.load(LINEAR_REGRESSION_MODEL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {LINEAR_REGRESSION_MODEL}. "
            "Ensure scikit-learn and numpy versions match those used during "
            f"training. Detail: {type(exc).__name__}: {exc}"
        ) from exc
