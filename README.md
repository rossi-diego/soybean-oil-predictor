# Soybean Oil Predictor

[![CI](https://github.com/rossi-diego/soybean-oil-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/rossi-diego/soybean-oil-predictor/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Commodity price forecasting for the front-month soybean oil futures contract (BOC1) using domain-driven ML. Built with a full medallion-layer data pipeline, XGBoost baseline, and interactive Next.js dashboard.

---

## Business Context

**Who uses this:** Vegetable oil traders, risk managers, and commodity analysts at firms like Cargill, Bunge, Viterra, and ADM.

**What decision it supports:** Timing hedging decisions and identifying relative value opportunities in the BOC1/CPO spread. A 1% improvement in hedge timing on a 10,000 MT position is worth approximately **$50,000**.

**Why it matters:** Soybean oil (BOC1) is the global benchmark for vegetable oil pricing, traded on the Chicago Board of Trade (CBOT). Its price is driven by crush economics, energy markets (biodiesel), and substitution dynamics with palm oil (CPO). This model captures those domain-specific relationships using commodity spreads, technical indicators, and fundamental data.

---

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ DATA SOURCES │     │  BRONZE  │     │  SILVER  │     │   GOLD   │
│              │     │          │     │          │     │          │
│ yfinance     │────>│ Parquet  │────>│ dbt-core │────>│ dbt-core │
│ USDA API     │     │ append   │     │ DuckDB   │     │ DuckDB   │
│ FRED         │     │ only     │     │ cleaned  │     │ features │
└──────────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                            │
┌──────────────┐     ┌──────────┐     ┌──────────┐     ┌────▼─────┐
│  NEXT.JS UI  │<────│ FASTAPI  │<────│  MLFLOW  │<────│  MODELS  │
│              │     │          │     │          │     │          │
│ Dashboard    │     │ /predict │     │ Tracking │     │ XGBoost  │
│ Prediction   │     │ /features│     │ Registry │     │ Ridge    │
│ EDA          │     │ /models  │     │          │     │ ARIMA    │
│ Monitoring   │     │ /health  │     │          │     │ SHAP     │
└──────────────┘     └──────────┘     └──────────┘     └──────────┘
```

---

## Features

- **Medallion data pipeline** — Bronze (raw Parquet) → Silver (cleaned, dbt + DuckDB) → Gold (feature-engineered)
- **Domain-driven features** — Crush spread, BOC1/CPO spread, soy/corn ratio, oil share, rolling volatility
- **XGBoost baseline** — Industry standard for tabular data; always compared against linear and time-series models
- **SHAP explainability** — Feature importance for every prediction (required for risk models)
- **Walk-forward validation** — Time-series aware evaluation, no look-ahead bias
- **FastAPI backend** — Async API with Pydantic v2 validation, auto-generated docs
- **Next.js frontend** — Dark-themed dashboard with prediction interface, EDA, model comparison, monitoring
- **MLflow tracking** — Full experiment versioning and model registry
- **Evidently AI monitoring** — Data drift and model performance degradation alerts
- **Airflow orchestration** — Daily ingestion, monthly WASDE, weekly retraining
- **Docker Compose** — Full stack with one command

---

## Quick Start (< 5 minutes)

### Option 1: Docker (recommended)

```bash
git clone https://github.com/rossi-diego/soybean-oil-predictor.git
cd soybean-oil-predictor
cp .env.example .env
docker compose up --build
```

| Service   | URL                          |
|-----------|------------------------------|
| Frontend  | http://localhost:3000         |
| API Docs  | http://localhost:8000/docs    |
| MLflow    | http://localhost:5000         |

### Option 2: Local development

```bash
git clone https://github.com/rossi-diego/soybean-oil-predictor.git
cd soybean-oil-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"

# Start API
uvicorn src.serving.app:app --reload

# Start frontend (in another terminal)
cd frontend && npm install && npm run dev
```

### Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
soybean-oil-predictor/
├── src/
│   ├── config.py                 # Centralized paths + settings
│   ├── log.py                    # Structured logging (structlog)
│   ├── utils.py                  # Shared utilities
│   ├── visualization.py          # Plotting utilities
│   ├── ingestion/                # Bronze layer: yfinance, USDA, FRED
│   ├── features/                 # Gold layer: spreads, technical, calendar
│   ├── models/                   # ML: XGBoost, linear, time-series, SHAP
│   ├── serving/                  # FastAPI app + routes
│   └── monitoring/               # Evidently AI drift + performance
├── dbt/                          # dbt-core + DuckDB (silver/gold SQL)
├── dags/                         # Airflow DAGs (ingest, retrain)
├── frontend/                     # Next.js 14 + Tailwind + recharts
├── tests/                        # Unit + integration + API tests
├── data/                         # Local data lake (bronze/silver/gold)
├── docker-compose.yml            # Full stack: API + frontend + MLflow
├── Dockerfile                    # Multi-stage (API + Streamlit)
├── Dockerfile.frontend           # Next.js production build
└── .github/workflows/ci.yml     # Lint, test, Docker build
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Ingestion | yfinance, USDA API, FRED | Free public data for commodity futures + fundamentals |
| Storage | Parquet + DuckDB | Delta Lake-compatible, maps to Databricks architecture |
| Transformation | dbt-core + DuckDB | SQL-based, testable, industry standard |
| Modeling | XGBoost, scikit-learn, statsforecast | XGBoost baseline + linear benchmarks + time-series |
| Explainability | SHAP | Required for risk model transparency |
| Tracking | MLflow | Experiment versioning + model registry |
| API | FastAPI + Pydantic v2 | Async, type-safe, auto-documented |
| Frontend | Next.js 14 + Tailwind + shadcn/ui | Interactive, dark-themed, recruiter-ready |
| Monitoring | Evidently AI | Data drift + model performance alerts |
| Orchestration | Airflow | DAG scheduling for data + ML pipelines |
| CI/CD | GitHub Actions | Lint + test + Docker build |
| Container | Docker Compose | Full stack reproducibility |

---

## Models

| Model | Type | Purpose |
|-------|------|---------|
| XGBoost | Gradient boosting | **Primary baseline** — best accuracy on tabular commodity data |
| Ridge / Lasso / ElasticNet | Regularized linear | Interpretable coefficients, feature selection |
| AutoARIMA / AutoETS / AutoTheta | Time-series | Capture temporal patterns and seasonality |

All models are evaluated using:
- **Walk-forward validation** (TimeSeriesSplit — no look-ahead bias)
- **MAE, RMSE, R², MAPE** on out-of-sample data
- **Directional accuracy** — % of correct up/down predictions (critical for trading)

---

## Domain Features

| Feature | What It Captures |
|---------|-----------------|
| Crush spread | Soybean processing margin (oil + meal - beans) |
| BOC1/CPO spread | Substitution dynamics between soy oil and palm oil |
| Soy/corn ratio | Planting decision economics |
| Oil share | Soybean value attributable to oil vs meal |
| Rolling volatility | Volatility regime (5d, 21d, 63d windows) |
| Lagged returns | Momentum signals (1d, 5d, 21d) |
| Z-score | Mean-reversion signal |
| Cyclical month | Seasonality (planting/growing/harvest cycles) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Diego Rossi** — Market Risk Coordinator at Grupo Oleoplan, Porto Alegre, Brazil.
Commodity markets, derivatives, quantitative analysis.
