# 🌱 Soybean Oil Spread Predictor

A beginner-friendly data science project to forecast the price of the front-month soybean oil contract (BOC1) using historical spreads against related commodities such as palm oil and soybean meal. No specialized finance background required.

---

## 🚀 Project Overview

**Goal:**  
Predict the price of the front-month soybean oil contract (BOC1) using relative price spreads versus other key commodities (e.g., palm oil, soybean meal).

**Key Steps:**
- **Data Preparation:** Clean raw price data and save as `.parquet`.
- **Exploratory Analysis:** Inspect trends, correlations, and outliers using Jupyter notebooks.
- **Modeling:** Train and evaluate Linear, Lasso, Ridge, and ElasticNet regression models.
- **Visualization & Reporting:** Build charts and export reports for insights.

---

## 🗂️ Project Structure

```
soybean-oil-spread-predictor/
├── .gitignore               # Git ignored files and folders
├── LICENSE                  # MIT License
├── requirements.txt         # Python dependencies
├── data/                    # Raw and cleaned data files
│   ├── commodities_raw_data.csv
│   └── commodities_clean_data.parquet
├── models/                  # Saved regression model files (.joblib)
├── notebooks/
│   └── 01-spreads-eda.ipynb # Exploratory analysis notebook
├── reports/                 # Generated images or HTML reports
├── src/                     # Source code modules
│   ├── config.py            # Project configuration and paths
│   ├── models.py            # Model pipelines, training, CV
│   ├── utils.py             # Helpers (e.g., coefficient formatting)
│   └── visualization.py     # Plotting utilities
```

---

## 💻 Setup & Installation

Clone the repository:

```bash
git clone https://github.com/SEU_USUARIO/soybean-oil-spread-predictor.git
cd soybean-oil-spread-predictor
```

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Or cmd:
.venv\Scripts\activate.bat
# Or macOS/Linux:
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Exploratory Analysis

Launch Jupyter Lab or Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Then open and run: `notebooks/01-spreads-eda.ipynb`

---

### Model Training & Evaluation

From within your notebook or a script:

```python
from src.models import train_and_validate_regression_model
# Follow structure in src/models.py or the notebook example
```

---

### (Optional) Report Generation

```bash
python src/reports_generator.py
```

---

## 🤝 Contributing

Fork the repo and create your feature branch:

```bash
git checkout -b feature/your-feature-name
```

Commit your changes and push:

```bash
git add .
git commit -m "Describe your feature"
git push origin feature/your-feature-name
```

Then open a Pull Request for discussion or review.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Contact

Questions or feedback?  
Open an issue or reach out to **Diego Rossi 94.diegorossi@gmail.com**.