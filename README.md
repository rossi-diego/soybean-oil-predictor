# 🛢️ Soybean Oil Price Predictor

This project forecasts the price of the front-month soybean oil futures contract (BOC1) using absolute prices of related commodities (e.g., palm oil, soybean meal, crude oil). It’s a beginner-friendly project focused on applied data science and regression modeling, deployed with a full-featured Streamlit app.

🔗 **Live App**: [soybean-oil-predictor.streamlit.app](https://soybean-oil-predictor.streamlit.app)

---

## 🚀 Project Overview

**Goal:**  
Predict the price of the front-month soybean oil contract (BOC1) using features derived from absolute prices of key related commodities and temporal information.

**Pipeline Highlights:**
- Clean and preprocess historical price data
- Perform cyclical encoding of the `month` variable
- Use `RobustScaler` to scale numerical features
- Train Linear, Ridge, Lasso, and ElasticNet models
- Compare performance using cross-validation
- Visualize residuals and coefficients
- Build and deploy a Streamlit app with EDA, model diagnostics, and prediction interface

---

## 🌐 App Features

- 🔍 **Exploratory Data Analysis**  
  Visualizations including pairplots, heatmaps, boxplots, and BOC1 trend with rolling average.

- 📈 **Model Results**  
  Cross-validation scores, feature coefficients, and residual diagnostics.

- 🧮 **Prediction Interface**  
  Users can input key commodity prices and month to forecast the BOC1 price.

---

## 📁 Repository Structure

```
soybean-oil-spread-predictor/
├── app.py                    # Streamlit app
├── data/                     # Cleaned datasets and model evaluation data
│   ├── commodities_clean_data.parquet
│   └── model_comparison_results.parquet
├── models/                   # Trained regression model
│   └── linear_regression.joblib
├── notebooks/                # Jupyter notebooks for EDA and modeling
│   ├── 01-eda.ipynb
│   └── 02-linear_regression.ipynb
├── reports/                  # Saved charts and visual outputs
│   └── images/
├── requirements.txt          # Project dependencies
└── src/                      # Core Python modules (config, models, utils, viz)
```

---

## ⚙️ Setup & Installation

Clone the repository:

```bash
git clone https://github.com/SEU_USUARIO/soybean-oil-spread-predictor.git
cd soybean-oil-spread-predictor
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App Locally

```bash
streamlit run app.py
```

---

## 📄 License

MIT License. See `LICENSE` for more information.

---

## 🙋‍♂️ Author

**Diego Rossi**  
[https://soybean-oil-predictor.streamlit.app](https://soybean-oil-predictor.streamlit.app)