# ─────────────────────────────────────────────
# Standard libraries
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Third-party libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from PIL import Image

# ─────────────────────────────────────────────
# Paths and environment settings
ROOT_DIR = Path(__file__).resolve().parent
SRC_PATH = ROOT_DIR / "notebooks" / "src"
IMG_PATH = ROOT_DIR / "reports" / "images"


# Garantir que o Python encontre os módulos locais
sys.path.append(str(SRC_PATH))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ─────────────────────────────────────────────
# Local project modules
from src.visualization import (
    PALETTE,
    plot_coefficients,
    plot_model_metrics_comparison,
    plot_residual_estimator, 
    SCATTER_ALPHA
)
from src.utils import coefficients_dataframe
from src.config import LINEAR_REGRESSION_MODEL, CLEAN_DATA, RESULTS_DATA, FEATURES_DATA

# ─────────────────────────────────────────────
# App configuration
import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Soybean Oil Predictor", layout="wide")

# ─────────────────────────────────────────────
# Load model
def load_model():
    """
    Garante que módulos locais estejam importados antes do unpickle
    e dá uma mensagem amigável se houver incompatibilidade de versões/símbolos.
    """
    try:
        # importe módulos do projeto que possam conter funções/classes usadas no pipeline
        mdl = joblib.load(LINEAR_REGRESSION_MODEL)  # aceita Path
        return mdl
    except Exception as e:
        st.error(
            "❌ Falha ao carregar o modelo.\n\n"
            "Verifique se as versões de scikit-learn/numpy são as mesmas usadas no treino **e** "
            "se qualquer transformer/função custom está definido em um módulo Python importável "
            "antes do `joblib.load()`.\n\n"
            f"Detalhe técnico: {type(e).__name__}"
        )
        st.stop()


# ─────────────────────────────────────────────
# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["📊 Exploratory Data Analysis", "📈 Model Results", "🧮 Make a Prediction"],
        icons=["bar-chart", "graph-up", "calculator"],
        menu_icon="cast",
        default_index=0,
    )

# ─────────────────────────────────────────────
# Page 1: Exploratory Data Analysis (EDA)
if selected == "📊 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.write(
        """
        This section presents key visual analyses to better understand the structure, behavior, 
        and relationships within the dataset used to forecast the front-month soybean oil contract (BOC1).

        We focus on identifying feature relevance, variable distributions, correlations, 
        and temporal patterns that support model selection and interpretation.
        """
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    IMG_PATH = Path(__file__).resolve().parent / "reports" / "images"

    image_sections = [
        {
            "file": "pairplot.png",
            "title": "📊 Pairplot of Top Correlated Features",
            "text": "This chart shows pairwise relationships between the target variable (BOC1). "
                    "It helps reveal potential linear or non-linear relationships.",
            "caption": "Pairplot of BOC1 and Correlated Features"
        },
        {
            "file": "heatmap_corr.png",
            "title": "🔥 Correlation Heatmap",
            "text": "This heatmap shows the Pearson correlation between all numeric variables. "
                    "It helps identify multicollinearity and the strongest predictors of BOC1.",
            "caption": "Correlation Matrix Heatmap"
        },
        {
            "file": "boxplot_all_vars.png",
            "title": "📦 Distribution of Variables (Boxplots)",
            "text": "Boxplots display the distribution, skewness, and presence of outliers in each feature. "
                    "The red dot represents the mean, and the line inside the box is the median.",
            "caption": "Boxplot of All Numeric Variables"
        },
        {
            "file": "rolling_avg_boc1.png",
            "title": "📈 BOC1 Price Evolution and Rolling Average",
            "text": "This line chart shows the monthly price of BOC1 over time, along with a 12-month rolling average. "
                    "It helps reveal long-term trends and seasonal patterns.",
            "caption": "BOC1 Monthly Price and Rolling Mean"
        },
    ]

    for section in image_sections:
        st.markdown(f"### {section['title']}")
        st.write(section["text"])

        img_file = IMG_PATH / section["file"]
        if img_file.exists():
            st.image(Image.open(img_file), caption=section["caption"], use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning(f"{section['file']} not found.")

# ─────────────────────────────────────────────
# Page 2: Model Results
elif selected == "📈 Model Results":
    st.subheader("📌 Model Coefficients")
    st.write("""
        This section summarizes the results from various regression models trained to predict the price of the front-month soybean oil contract (BOC1).

        We compare model performance using cross-validation metrics, inspect feature importance via coefficients, and evaluate residuals to diagnose model behavior and reliability.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "This chart shows the magnitude and direction of each feature's influence on the target variable (BOC1). "
        "Positive values push predictions up, negative values pull them down. "
        "Features with larger absolute coefficients are more impactful in the model."
    )

    # --- Load model
    model = model = load_model()


    pipe = getattr(model, "regressor", model)


    preproc = pipe.named_steps.get("preprocessor")
    reg = pipe.named_steps.get("reg")

    if reg is None:
        st.error("Model step 'reg' not found in pipeline.")
    else:
        import numpy as np


        coefs = getattr(reg, "coef_", None)
        if coefs is None:
            st.error("Could not read coefficients from the model step 'reg'.")
        else:
            coefs = np.asarray(coefs).reshape(-1)


            if preproc is not None and hasattr(preproc, "get_feature_names_out"):
                features = preproc.get_feature_names_out()

                features = pd.Index(features).str.replace(r".*__", "", regex=True)
            else:

                df_tmp = pd.read_parquet("data/commodities_clean_data.parquet")
                features = df_tmp.drop(columns="boc1").columns


            if len(features) != len(coefs):
                st.warning(f"Feature length ({len(features)}) != coef length ({len(coefs)}). Attempting to align.")
                min_len = min(len(features), len(coefs))
                features = pd.Index(features[:min_len])
                coefs = coefs[:min_len]

            df_coefs = pd.DataFrame({"feature": features, "coefficient": coefs})
            df_coefs = df_coefs.sort_values(by="coefficient", ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(df_coefs["feature"], df_coefs["coefficient"]) 
            ax.axvline(x=0, color="gray", linestyle="--")
            ax.set_title("Model Coefficients")
            ax.set_xlabel("Coefficient Value")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Model Performance")
    st.markdown("""
        This table presents the **average cross-validation scores** for each model tested:

        - **R² Score**: Proportion of the variance in the target variable explained by the model (higher = better).
        - **MAE**: Mean Absolute Error (lower = better).
        - **RMSE**: Root Mean Squared Error (lower = better).

        ℹ️ *Note: Since sklearn returns MAE and RMSE as negative scores, values below have been converted to positive.*
    """, unsafe_allow_html=True)

    df_results = pd.read_parquet(RESULTS_DATA)
    summary_table = (
        df_results
        .groupby("model")
        .mean(numeric_only=True) 
        .assign(
            test_neg_mean_absolute_error=lambda df: -df["test_neg_mean_absolute_error"],
            test_neg_root_mean_squared_error=lambda df: -df["test_neg_root_mean_squared_error"]
        )
        .sort_values(by="test_neg_root_mean_squared_error", ascending=True)
    )[
        ["test_r2", "test_neg_mean_absolute_error", "test_neg_root_mean_squared_error"]
    ].round(4)

    st.dataframe(summary_table)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📉 Residual Analysis for the Best Model")
    st.write("""
        This plot helps assess how well the model predictions align with the actual values.

        - **Left**: Histogram of residuals (prediction errors)
        - **Middle**: Residuals vs predicted values
        - **Right**: Actual vs predicted values

        A good model shows residuals randomly scattered around zero and tight clustering around the diagonal.
    """)

    df = pd.read_parquet(CLEAN_DATA)
    target_column = "boc1"
    X = df.drop(columns=target_column)
    y = df[target_column]

    plot_residual_estimator(model, X, y)
    st.pyplot(plt.gcf())

# ─────────────────────────────────────────────
# Page 3: Make a Prediction
elif selected == "🧮 Make a Prediction":
    st.subheader("🧮 Make a Prediction")
    st.write("""
        Fill in the required commodity and calendar variables to forecast the price of the front-month soybean oil contract (BOC1).

        The table below summarizes the statistical range of each variable (count, mean, min, max, etc).
    """)

    df_stats = pd.read_csv(FEATURES_DATA, index_col=0)
    requested_cols = ["smc1", "sc1", "lcoc1", "hoc1", "fcpoc1", "rsc1", "so-premp-c1", "brl="]
    available_cols = [c for c in requested_cols if c in df_stats.columns]
    missing_cols = sorted(set(requested_cols) - set(available_cols))

    if missing_cols:
        st.warning(f"As seguintes variáveis não estão no arquivo de stats e serão ignoradas: {missing_cols}")

    if not available_cols:
        st.error("Nenhuma das colunas esperadas está disponível em features_describe.csv.")
        st.stop()

    df_stats = df_stats[available_cols]
    st.dataframe(df_stats.T.style.format(precision=2))

    st.markdown("### Enter input values")

    def build_help(col):
        if col not in df_stats.columns:
            return "Sem estatísticas disponíveis para este campo."
        desc = df_stats[col]
        def _num(x):
            try:
                return float(x)
            except Exception:
                return x
        min_v = _num(desc.get("min", ""))
        max_v = _num(desc.get("max", ""))
        mean_v = _num(desc.get("mean", ""))
        try:
            return f"Typical range: {min_v:.0f}–{max_v:.0f} | Mean: {mean_v:.0f}"
        except Exception:
            return f"Typical range: {min_v}–{max_v} | Mean: {mean_v}"

    smc1 = st.number_input("Soybean Meal (SMC1)", min_value=0.0, help=build_help("smc1"))
    sc1 = st.number_input("Soybean (SC1)", min_value=0.0, help=build_help("sc1"))
    lcoc1 = st.number_input("Brent Crude (LCOc1)", min_value=0.0, help=build_help("lcoc1"))
    hoc1 = st.number_input("Heating Oil (HOC1)", min_value=0.0, help=build_help("hoc1"))
    fcpoc1 = st.number_input("Palm Oil (FCPOc1)", min_value=0.0, help=build_help("fcpoc1"))
    rsc1 = st.number_input("Rapeseed Oil (RSC1)", min_value=0.0, help=build_help("rsc1"))
    so_premp_c1 = st.number_input("Soybean Oil Paranaguá Basis", min_value=-5000, help=build_help("so-premp-c1"))
    brl = st.number_input("Brl x USD", min_value=0.0, help=build_help("brl="))    

    input_data = pd.DataFrame([{
        "smc1": smc1,
        "sc1": sc1,
        "lcoc1": lcoc1,
        "hoc1": hoc1,
        "fcpoc1": fcpoc1,
        "rsc1": rsc1,
        "so-premp-c1": so_premp_c1,
        "brl=": brl
    }])

    if st.button("🔍 Predict BOC1"):
        model = load_model()
        pred_val = model.predict(input_data)
        prediction = float(np.ravel(pred_val)[0])
        st.success(f"📈 Predicted BOC1 Price: **{prediction:.2f}**")

        boc1_stats = pd.read_parquet(CLEAN_DATA)["boc1"].describe()
        st.caption(f"Training data range: {boc1_stats['min']:.2f}–{boc1_stats['max']:.2f} | Mean: {boc1_stats['mean']:.2f}")
