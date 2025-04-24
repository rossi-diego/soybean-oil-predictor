Soybean Oil Spread Predictor

A beginner-friendly data science project that analyzes historical price spreads between soybean oil and related commodities (e.g., palm oil, soybean meal) to forecast the price of the front‑month soybean oil contract (BOC1). No specialized finance background required.

🚀 Project Overview

Goal: Forecast the price of the front‑month soybean oil contract (BOC1) using spreads versus related commodities (palm oil, soybean meal, etc).

Key Steps:

Data Preparation: Clean raw price data and save as Parquet.

Exploratory Analysis: Inspect trends, correlations, and outliers in a Jupyter notebook.

Modeling: Train and validate Linear, Lasso, Ridge, and ElasticNet regression models.

Visualization & Reporting: Generate charts and reports for clear insights.

🗂️ Repository Structure

.
├── .gitignore           # Ignored files and folders (venv, caches)
├── LICENSE              # MIT License
├── requirements.txt     # Python dependencies
├── notebooks/           # Jupyter notebooks for EDA
│   └── 01-spreads-eda.ipynb
├── data/                # Raw and cleaned data files
│   ├── commodities_raw_data.csv
│   └── commodities_clean_data.parquet
├── src/                 # Core Python code modules
│   ├── config.py        # File paths and project settings
│   ├── models.py        # Model pipelines and validation functions
│   ├── utils.py         # Helper functions (e.g., format coefficients)
│   └── visualization.py # Plotting utilities
├── models/              # Serialized model files (.joblib)
└── reports/             # Generated HTML/PDF reports and images

💻 Setup & Installation

Clone the repository

git clone https://github.com/SEU_USUARIO/soybean-oil-spread-predictor.git
cd soybean-oil-spread-predictor

Create and activate a virtual environment

python -m venv .venv
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Windows (cmd.exe):   .venv\Scripts\activate.bat
# macOS/Linux:         source .venv/bin/activate

Install dependencies

pip install -r requirements.txt

▶️ Usage

Exploratory Analysis

Launch Jupyter Lab or Notebook:

jupyter lab
# or
jupyter notebook

Open notebooks/01-spreads-eda.ipynb and run all cells.

Train & Evaluate Models

from src.models import train_and_validate_regression_model
# Follow examples in src/models.py or the notebook

Generate Reports (if applicable)

python src/reports_generator.py

🤝 Contributing

Fork the repository and create a branch:

git checkout -b feature/your-feature-name

Make your changes, then commit and push:

git add .
git commit -m "Describe your changes"
git push origin feature/your-feature-name

Open a Pull Request for review.

📄 License

This project is licensed under the MIT License. See LICENSE for details.

👋 Questions or feedback? Open an issue or contact Diego Rossi.

