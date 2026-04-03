"""Model Results page."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CLEAN_DATA, RESULTS_DATA
from src.utils import load_model
from src.visualization import plot_residual_estimator


def render() -> None:
    """Render the Model Results page.

    Displays model coefficients, cross-validation performance metrics,
    and residual diagnostic plots for the best trained pipeline.
    """
    st.title("Model Results")
    st.write(
        """
        This section summarises the results from various regression models
        trained to predict the price of the front-month soybean oil contract
        (BOC1).

        We compare model performance using cross-validation metrics, inspect
        feature importance via coefficients, and evaluate residuals to
        diagnose model behaviour and reliability.
        """
    )

    # ── Coefficients ──────────────────────────────────────────────────────────
    st.subheader("Model Coefficients")
    st.write(
        "This chart shows the magnitude and direction of each feature's "
        "influence on the target variable (BOC1). Positive values push "
        "predictions up; negative values pull them down. Features with larger "
        "absolute coefficients are more impactful."
    )

    try:
        model = load_model()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    pipe = getattr(model, "regressor", model)
    preproc = pipe.named_steps.get("preprocessor")
    reg = pipe.named_steps.get("reg")

    if reg is None:
        st.error("Pipeline step 'reg' not found. Verify the saved model structure.")
        st.stop()

    coefs = getattr(reg, "coef_", None)
    if coefs is None:
        st.error("Could not read coefficients from pipeline step 'reg'.")
        st.stop()

    coefs = np.asarray(coefs).reshape(-1)

    if preproc is not None and hasattr(preproc, "get_feature_names_out"):
        features = pd.Index(preproc.get_feature_names_out()).str.replace(
            r".*__", "", regex=True
        )
    else:
        df_tmp = pd.read_parquet(CLEAN_DATA)
        features = df_tmp.drop(columns="boc1").columns

    if len(features) != len(coefs):
        st.warning(
            f"Feature count ({len(features)}) does not match coefficient "
            f"count ({len(coefs)}). Truncating to the shorter length."
        )
        min_len = min(len(features), len(coefs))
        features = pd.Index(features[:min_len])
        coefs = coefs[:min_len]

    df_coefs = pd.DataFrame({"feature": features, "coefficient": coefs}).sort_values(
        by="coefficient", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_coefs["feature"], df_coefs["coefficient"])
    ax.axvline(x=0, color="gray", linestyle="--")
    ax.set_title("Model Coefficients")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
    plt.close(fig)

    # ── Performance table ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Performance (Cross-Validation)")
    st.markdown(
        """
        Average cross-validation scores for each candidate model:

        - **R²**: Proportion of variance explained (higher = better).
        - **MAE**: Mean Absolute Error (lower = better).
        - **RMSE**: Root Mean Squared Error (lower = better).

        *Note: scikit-learn returns MAE and RMSE as negative scores; values
        below have been converted to positive.*
        """,
        unsafe_allow_html=True,
    )

    df_results = pd.read_parquet(RESULTS_DATA)
    summary_table = (
        df_results.groupby("model")
        .mean(numeric_only=True)
        .assign(
            test_neg_mean_absolute_error=lambda df: -df[
                "test_neg_mean_absolute_error"
            ],
            test_neg_root_mean_squared_error=lambda df: -df[
                "test_neg_root_mean_squared_error"
            ],
        )
        .sort_values(by="test_neg_root_mean_squared_error", ascending=True)
    )[
        [
            "test_r2",
            "test_neg_mean_absolute_error",
            "test_neg_root_mean_squared_error",
        ]
    ].round(4)

    st.dataframe(summary_table)

    # ── Residual analysis ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Residual Analysis — Best Model")
    st.write(
        """
        These plots assess how well the model predictions align with actual values.

        - **Left**: Histogram of residuals (prediction errors).
        - **Middle**: Residuals vs predicted values.
        - **Right**: Actual vs predicted values.

        A well-calibrated model shows residuals randomly scattered around zero
        and tight clustering along the diagonal.
        """
    )

    df = pd.read_parquet(CLEAN_DATA)
    X = df.drop(columns="boc1")
    y = df["boc1"]

    fig = plot_residual_estimator(model, X, y)
    st.pyplot(fig)
    plt.close(fig)


render()
