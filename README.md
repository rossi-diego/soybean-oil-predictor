# Soybean Oil Price Predictor

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?logo=vercel)](https://soybean-oil-predictor.vercel.app)
[![API Docs](https://img.shields.io/badge/API_Docs-Render-46E3B7?logo=render)](https://soybean-oil-predictor-api.onrender.com/docs)
[![CI](https://github.com/rossi-diego/soybean-oil-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/rossi-diego/soybean-oil-predictor/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Forecasts the front-month soybean oil futures contract (BOC1) using cross-commodity signals, crush economics, and machine learning.

**Live Dashboard:** https://soybean-oil-predictor.vercel.app/
**API Documentation:** https://soybean-oil-predictor-api.onrender.com/docs

---

## Architecture

```
Frontend (Next.js 15 + TypeScript + Tailwind)  ──>  Vercel
Backend  (FastAPI + Ridge/XGBoost + DuckDB)    ──>  Render
Data     (yfinance — daily commodity prices)
```

---

## Key Results

| Metric | Value | Method |
|--------|-------|--------|
| MAE | 2.69 c/lb | Walk-forward (5-fold TimeSeriesSplit) |
| RMSE | 3.53 c/lb | Out-of-sample holdout |
| R2 | 0.81 | Out-of-sample holdout |
| Directional Accuracy | 76.3% | Correct up/down predictions |
| Champion Model | Ridge | Lowest MAE across 5 models |

5 models compared: XGBoost, Ridge, Lasso, ElasticNet, Linear Regression. Ridge selected as champion for lowest MAE on holdout set. Walk-forward validation across 2,100 out-of-sample predictions confirms generalization.

---

## Features

- **Walk-forward validated model** with 95% prediction intervals
- **Real-time commodity prices** from Yahoo Finance (15-min cache)
- **Crush spread and oil/palm spread signals** with 30-day MA and trend interpretation
- **Scenario analysis** — adjust input prices and forecast BOC1
- **Interactive EDA** — price history, correlation matrix, distributions, seasonality, ADF stationarity tests, ACF/PACF
- **5-model comparison** with residual diagnostics, feature importance, and learning curves
- **Model Card** with training metadata, feature explanations, and confidence intervals
- **Feature contribution chart** (XGBoost native SHAP) for every prediction

---

## Stack

| Layer | Technology |
|-------|-----------|
| **ML** | Ridge, XGBoost, scikit-learn, statsforecast |
| **API** | FastAPI, Pydantic v2, uvicorn |
| **Data** | DuckDB, Parquet, pandas, pyarrow, yfinance |
| **Frontend** | Next.js 15, TypeScript, Tailwind CSS, Recharts |
| **MLOps** | Walk-forward validation, model comparison, confidence intervals |
| **Infra** | Docker, GitHub Actions CI, Vercel, Render |
| **Quality** | ruff, pytest, structlog |

---

## Quick Start

### API

```bash
git clone https://github.com/rossi-diego/soybean-oil-predictor.git
cd soybean-oil-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
python scripts/train_model.py
uvicorn src.serving.app:app --reload
```

API at http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend at http://localhost:3000

### Docker (full stack)

```bash
cp .env.example .env
docker compose up --build
```

### Tests

```bash
pip install -r requirements.txt && pip install -e ".[dev]"
pytest tests/ -v
```

---

## Project Structure

```
soybean-oil-predictor/
├── src/
│   ├── config.py                 # Tickers, feature columns, settings
│   ├── log.py                    # Structured logging (structlog)
│   ├── ingestion/                # yfinance, USDA WASDE, FRED ingestion
│   ├── features/                 # Crush spread, technical indicators, calendar
│   ├── models/                   # XGBoost, linear models, evaluation, SHAP
│   ├── serving/                  # FastAPI app, routes, model cache, schemas
│   │   └── routes/
│   │       ├── predict.py        # POST /predict with confidence + contributions
│   │       ├── prices.py         # Live prices, spreads, backtest, history
│   │       ├── eda.py            # Correlations, distributions, seasonality, ADF
│   │       └── models.py         # Comparison, diagnostics, importance, learning curves
│   └── monitoring/               # Evidently AI drift detection (planned)
├── scripts/
│   └── train_model.py            # Train 5 models, walk-forward, learning curves
├── dbt/                          # dbt-core + DuckDB SQL models
├── dags/                         # Airflow DAGs (ingest, retrain)
├── frontend/                     # Next.js 15 + Tailwind + Recharts
│   ├── app/                      # Dashboard, Forecast, Analysis, Models
│   ├── components/               # Navbar, demo banner
│   └── lib/api.ts                # API client with demo mode fallback
├── tests/                        # Unit + integration + API tests
├── data/                         # Parquet training data + JSON diagnostics
├── Dockerfile.api                # Lightweight image for Render
├── docker-compose.yml            # Full stack (API + frontend + MLflow)
├── Makefile                      # make install, api, frontend, lint, test, train
└── .github/workflows/ci.yml     # Lint + test + Docker build
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + model availability |
| `POST` | `/api/v1/predict` | Forecast with confidence interval + feature contributions |
| `GET` | `/api/v1/predict/live` | Forecast using current market prices |
| `GET` | `/api/v1/prices/latest` | Live commodity prices (15-min cache) |
| `GET` | `/api/v1/spreads` | Crush/oil-palm spread signals with 30d MA |
| `GET` | `/api/v1/backtest` | Walk-forward predictions + metrics |
| `GET` | `/api/v1/models/comparison` | 5-model performance table |
| `GET` | `/api/v1/models/champion/diagnostics` | Residuals, predicted vs actual |
| `GET` | `/api/v1/models/champion/feature-importance` | Feature importance (all models) |
| `GET` | `/api/v1/models/learning-curves` | Training vs validation error curves |
| `GET` | `/api/v1/model/info` | Training metadata, feature ranges |
| `GET` | `/api/v1/eda/prices` | Historical multi-commodity prices |
| `GET` | `/api/v1/eda/correlations` | Pearson/Spearman correlation matrix |
| `GET` | `/api/v1/eda/distributions` | Feature histograms + drift comparison |
| `GET` | `/api/v1/eda/seasonality` | Monthly BOC1 box plot data |
| `GET` | `/api/v1/eda/stationarity` | ADF tests, ACF/PACF, returns distribution |

---

## Domain Context

BOC1 is the front-month soybean oil futures contract on the CBOT. Vegetable oil traders use crush spreads (processing margin) and oil/palm spreads (substitution signal) to time hedging decisions. A 1% improvement in hedge timing on a 10,000 MT position is worth ~$50,000.

The model uses 5 cross-commodity features: soybean meal, soybeans, Brent crude, heating oil, and wheat. These capture crush economics, biodiesel demand, and broader grain market sentiment.

---

## License

MIT

---

## Author

**Diego Rossi** — Market Risk & Data Engineering
