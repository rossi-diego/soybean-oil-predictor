# 🛢️ Soybean Oil Price Predictor

This project forecasts the price of the front-month soybean oil futures contract (BOC1) using absolute prices of related commodities (e.g., palm oil, soybean meal, crude oil). It's a beginner-friendly project focused on applied data science and regression modeling.

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

---

## 📁 Repository Structure

```
soybean-oil-spread-predictor/
├── data/                      # Raw and cleaned datasets
│   ├── commodities_raw_data.csv
│   └── commodities_clean_data.parquet
├── models/                    # Saved model file
│   └── linear_regression.joblib  # Best model: Lasso
├── notebooks/                 # Jupyter notebooks
│   ├── 01-eda.ipynb           # Exploratory data analysis
│   └── 02-linear_regression.ipynb # Modeling and evaluation
├── src/                       # Source code
│   ├── config.py              # Project paths
│   ├── models.py              # Model training and validation
│   ├── utils.py               # Helper functions
│   └── visualization.py       # Plots and analysis
├── reports/                   # Generated plots and summaries
└── requirements.txt           # Dependencies
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

## 📊 Usage

### 1. Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/01-eda.ipynb
```

### 2. Train & Evaluate Models

```bash
jupyter notebook notebooks/02-linear_regression.ipynb
```

The best model was **Lasso**, chosen based on lowest RMSE and MAE. The model is saved as:

```
models/linear_regression.joblib
```

### 3. Predict New Values (optional app)

Coming soon via Streamlit app.

---

## 📄 License

MIT License. See `LICENSE` for more information.

---

## 🙋‍♂️ Author

**Diego Rossi**  
For questions or suggestions, please open an issue or reach out directly.