"""Exploratory Data Analysis page."""


import streamlit as st
from PIL import Image

from src.config import IMAGE_FOLDER


def render() -> None:
    """Render the Exploratory Data Analysis page.

    Displays four pre-generated visualisation panels: pairplot,
    correlation heatmap, boxplots, and BOC1 rolling average.
    """
    st.title("Exploratory Data Analysis (EDA)")
    st.write(
        """
        This section presents key visual analyses to better understand the
        structure, behaviour, and relationships within the dataset used to
        forecast the front-month soybean oil contract (BOC1).

        We focus on identifying feature relevance, variable distributions,
        correlations, and temporal patterns that support model selection
        and interpretation.
        """
    )
    st.markdown("<br>", unsafe_allow_html=True)

    image_sections = [
        {
            "file": "pairplot.png",
            "title": "Pairplot of Top Correlated Features",
            "text": (
                "This chart shows pairwise relationships between the target "
                "variable (BOC1). It helps reveal potential linear or "
                "non-linear relationships."
            ),
            "caption": "Pairplot of BOC1 and Correlated Features",
        },
        {
            "file": "heatmap_corr.png",
            "title": "Correlation Heatmap",
            "text": (
                "This heatmap shows the Pearson correlation between all "
                "numeric variables. It helps identify multicollinearity and "
                "the strongest predictors of BOC1."
            ),
            "caption": "Correlation Matrix Heatmap",
        },
        {
            "file": "boxplot_all_vars.png",
            "title": "Distribution of Variables (Boxplots)",
            "text": (
                "Boxplots display the distribution, skewness, and presence "
                "of outliers in each feature. The red dot represents the mean "
                "and the line inside the box is the median."
            ),
            "caption": "Boxplot of All Numeric Variables",
        },
        {
            "file": "rolling_avg_boc1.png",
            "title": "BOC1 Price Evolution and Rolling Average",
            "text": (
                "This line chart shows the monthly price of BOC1 over time, "
                "along with a 12-month rolling average. It helps reveal "
                "long-term trends and seasonal patterns."
            ),
            "caption": "BOC1 Monthly Price and Rolling Mean",
        },
    ]

    for section in image_sections:
        st.markdown(f"### {section['title']}")
        st.write(section["text"])

        img_file = IMAGE_FOLDER / section["file"]
        if img_file.exists():
            st.image(
                Image.open(img_file),
                caption=section["caption"],
                use_container_width=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning(
                f"Image not found: `{section['file']}`. "
                "Re-run the EDA notebook to regenerate it."
            )


render()
