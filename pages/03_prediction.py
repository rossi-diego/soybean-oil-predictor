"""Make a Prediction page."""

import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src.config import CLEAN_DATA
from src.utils import load_model


def render() -> None:
    """Render the Make a Prediction page.

    Provides a form for entering commodity price inputs and returns a
    BOC1 price forecast from the trained regression pipeline. The month
    feature is derived automatically from the current calendar date.
    """
    st.title("Make a Prediction")
    st.write(
        """
        Fill in the commodity price inputs below to forecast the price of the
        front-month soybean oil contract (BOC1).

        The **Month** feature is derived automatically from today's date.
        The table below summarises the statistical range of each variable
        observed in the training dataset.
        """
    )

    df_train = pd.read_parquet(CLEAN_DATA)
    feature_cols = [c for c in df_train.columns if c != "boc1"]
    stats = df_train[feature_cols].describe()
    st.dataframe(stats.T.style.format(precision=2))

    st.markdown("### Enter input values")

    def _help(col: str) -> str:
        """Build a tooltip string showing the typical range for a feature.

        Args:
            col: Column name from the training dataset.

        Returns:
            Formatted string with min, max, and mean values.
        """
        if col not in stats.columns:
            return "No statistics available for this field."
        s = stats[col]
        try:
            return (
                f"Typical range: {s['min']:.0f}–{s['max']:.0f} | "
                f"Mean: {s['mean']:.0f}"
            )
        except Exception:
            return f"Min: {s.get('min')} | Max: {s.get('max')}"

    smc1 = st.number_input("Soybean Meal (SMC1)", min_value=0.0, help=_help("smc1"))
    sc1 = st.number_input("Soybean (SC1)", min_value=0.0, help=_help("sc1"))
    lcoc1 = st.number_input("Brent Crude (LCOc1)", min_value=0.0, help=_help("lcoc1"))
    hoc1 = st.number_input("Heating Oil (HOC1)", min_value=0.0, help=_help("hoc1"))
    fcpoc1 = st.number_input("Palm Oil (FCPOc1)", min_value=0.0, help=_help("fcpoc1"))
    rsc1 = st.number_input("Rapeseed Oil (RSC1)", min_value=0.0, help=_help("rsc1"))

    month = datetime.date.today().month
    st.caption(f"Month (auto-detected): **{month}**")

    input_data = pd.DataFrame(
        [
            {
                "smc1": smc1,
                "sc1": sc1,
                "lcoc1": lcoc1,
                "hoc1": hoc1,
                "fcpoc1": fcpoc1,
                "rsc1": rsc1,
                "month": month,
            }
        ]
    )

    if st.button("Predict BOC1"):
        try:
            model = load_model()
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

        pred_val = model.predict(input_data)
        prediction = float(np.ravel(pred_val)[0])
        st.success(f"Predicted BOC1 Price: **{prediction:.2f}**")

        boc1_stats = df_train["boc1"].describe()
        st.caption(
            f"Training data range: {boc1_stats['min']:.2f}–{boc1_stats['max']:.2f}"
            f" | Mean: {boc1_stats['mean']:.2f}"
        )


render()
