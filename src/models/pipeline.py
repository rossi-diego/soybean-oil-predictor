"""Model building, training, and evaluation utilities.

Preserved from the original models.py with full backward compatibility.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def build_regression_model_pipeline(
    regressor,
    preprocessor=None,
    target_transformer=None,
) -> Pipeline | TransformedTargetRegressor:
    """Assemble a scikit-learn regression pipeline.

    Args:
        regressor: Any scikit-learn compatible regressor.
        preprocessor: Optional feature preprocessor (e.g. ColumnTransformer).
        target_transformer: Optional target transformer via
            :class:`~sklearn.compose.TransformedTargetRegressor`.

    Returns:
        A fitted-ready Pipeline (or TransformedTargetRegressor wrapping one).
    """
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])

    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline

    return model


def train_and_validate_regression_model(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Train and cross-validate a regression pipeline.

    Args:
        X: Feature matrix.
        y: Target vector.
        regressor: Any scikit-learn compatible regressor.
        preprocessor: Optional feature preprocessor.
        target_transformer: Optional target transformer.
        n_splits: Number of folds for KFold cross-validation.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary of cross-validation scores.
    """
    model = build_regression_model_pipeline(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    return scores


def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    return_train_score: bool = False,
) -> GridSearchCV:
    """Build a GridSearchCV object for a regression pipeline.

    Args:
        regressor: Any scikit-learn compatible regressor.
        param_grid: Parameter grid for GridSearchCV.
        preprocessor: Optional feature preprocessor.
        target_transformer: Optional target transformer.
        n_splits: Number of KFold splits.
        random_state: Random seed.
        return_train_score: Whether to include training scores.

    Returns:
        Configured (unfitted) GridSearchCV.
    """
    model = build_regression_model_pipeline(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def organize_cv_results(results: dict) -> pd.DataFrame:
    """Flatten cross-validation results into a long-form DataFrame.

    Args:
        results: Mapping of model name to cross-validation scores dict.

    Returns:
        Long-form DataFrame with one row per fold per model.
    """
    for key in results:
        results[key]["time_seconds"] = (
            results[key]["fit_time"] + results[key]["score_time"]
        )

    df_results = (
        pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    )

    df_results_expanded = df_results.explode(
        df_results.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        df_results_expanded = df_results_expanded.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_results_expanded
