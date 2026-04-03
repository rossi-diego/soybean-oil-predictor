"""Soybean Oil Predictor — Streamlit application entry point."""

import streamlit as st

st.set_page_config(
    page_title="Soybean Oil Predictor",
    page_icon="🫘",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page("pages/01_eda.py", title="Exploratory Data Analysis", icon="📊"),
        st.Page("pages/02_model.py", title="Model Results", icon="📈"),
        st.Page("pages/03_prediction.py", title="Make a Prediction", icon="🧮"),
    ]
)
pg.run()
