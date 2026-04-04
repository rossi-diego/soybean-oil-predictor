# Soybean Oil Predictor

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?logo=vercel)](https://soybean-oil-predictor.vercel.app)
[![API Docs](https://img.shields.io/badge/API_Docs-Render-46E3B7?logo=render)](https://soybean-oil-predictor-api.onrender.com/docs)
[![CI](https://github.com/rossi-diego/soybean-oil-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/rossi-diego/soybean-oil-predictor/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Demo](https://soybean-oil-predictor.vercel.app)** | **[API Docs](https://soybean-oil-predictor-api.onrender.com/docs)** | Commodity price forecasting for the front-month soybean oil futures contract (BOC1) using domain-driven ML.

---

## Business Context

**Who uses this:** Vegetable oil traders, risk managers, and commodity analysts at firms like Cargill, Bunge, Viterra, and ADM.

**What decision it supports:** Timing hedging decisions and identifying relative value opportunities in the BOC1/CPO spread. A 1% improvement in hedge timing on a 10,000 MT position is worth approximately **$50,000**.

**Why it matters:** Soybean oil (BOC1) is the global benchmark for vegetable oil pricing, traded on the Chicago Board of Trade (CBOT). Its price is driven by crush economics, energy markets (biodiesel), and substitution dynamics with palm oil (CPO). This model captures those domain-specific relationships using commodity spreads, technical indicators, and fundamental data.

---

## Live Services

| Service | URL | Stack |
|---------|-----|-------|
| **Frontend** | [soybean-oil-predictor.vercel.app](https://soybean-oil-predictor.vercel.app) | Next.js 15 + Tailwind on Vercel |
| **API** | [soybean-oil-predictor-api.onrender.com/docs](https://soybean-oil-predictor-api.onrender.com/docs) | FastAPI + XGBoost on Render |

The API serves live commodity prices from Yahoo Finance, runs predictions with a trained XGBoost model, and exposes feature importance and model metadata endpoints.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API health and model availability |
| `GET` | `/api/v1/prices/latest` | Live commodity prices (15-min cache) |
| `GET` | `/api/v1/prices/history?days=90` | Historical BOC1 prices with model predictions |
| `GET` | `/api/v1/predict/live` | Forecast using current market prices |
| `POST` | `/api/v1/predict` | Forecast with custom input prices |
| `GET` | `/api/v1/features/stats` | Training data feature statistics |
| `GET` | `/api/v1/features/importance` | Feature importance from active model |
| `GET` | `/api/v1/models` | List trained models with metrics |

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
┌──────────────┐     ┌──────────┐                      ┌────▼─────┐
│  NEXT.JS UI  │<────│ FASTAPI  │<─────────────────────│  MODELS  │
│  (Vercel)    │     │ (Render) │                      │          │
│              │     │          │                      │ XGBoost  │
│ Dashboard    │     │ /predict │                      │ Ridge    │
│ Forecast     │     │ /prices  │                      │          │
│ Analysis     │     │ /models  │                      │          │
│ Monitoring   │     │ /health  │                      │          │
└──────────────┘     └──────────┘                      └──────────┘
```

---

## Features

- **Medallion data pipeline** — Bronze (raw Parquet) → Silver (cleaned, dbt + DuckDB) → Gold (feature-engineered)
- **Domain-driven features** — Crush spread, BOC1/CPO spread, soy/corn ratio, oil share, rolling volatility
- **XGBoost baseline** — Industry standard for tabular data; always compared against linear and time-series models
- **Walk-forward validation** — Time-series aware evaluation, no look-ahead bias
- **Live market data** — Real-time commodity prices from Yahoo Finance with 15-minute cache
- **FastAPI backend** — Async API with Pydantic v2 validation, deployed on Render
- **Next.js frontend** — Dark-themed dashboard with forecast interface, deployed on Vercel
- **Actual vs Predicted chart** — 90-day backtest visualization on the dashboard
- **Scenario analysis** — Adjust commodity prices to explore "what if" forecasts
- **Airflow orchestration** — Daily ingestion, monthly WASDE, weekly retraining (local)
- **Docker Compose** — Full stack runnable locally with one command

---

## Quick Start

### Run locally

```bash
git clone https://github.com/rossi-diego/soybean-oil-predictor.git
cd soybean-oil-predictor
```

**Backend API:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
python scripts/train_model.py
uvicorn src.serving.app:app --reload
```

API available at http://localhost:8000/docs

**Frontend** (in another terminal):

```bash
cd frontend && npm install && npm run dev
```

Frontend available at http://localhost:3000

### Docker (full stack)

```bash
cp .env.example .env
docker compose up --build
```

| Service   | URL                          |
|-----------|------------------------------|
| Frontend  | http://localhost:3000         |
| API       | http://localhost:8000/docs    |
| MLflow    | http://localhost:5000         |

### Run tests

```bash
pip install -r requirements.txt && pip install -e ".[dev]"
pytest tests/ -v
```

---

## Project Structure

```
soybean-oil-predictor/
├── src/
│   ├── config.py                 # Paths, tickers, settings
│   ├── log.py                    # Structured logging (structlog)
│   ├── utils.py                  # Shared utilities
│   ├── ingestion/                # Bronze layer: yfinance, USDA, FRED
│   ├── features/                 # Feature engineering: spreads, technical, calendar
│   ├── models/                   # ML: XGBoost, linear, time-series, evaluation, SHAP
│   ├── serving/                  # FastAPI app, routes, model cache, schemas
│   └── monitoring/               # Evidently AI drift + performance
├── scripts/
│   └── train_model.py            # Train XGBoost + Ridge from parquet data
├── dbt/                          # dbt-core + DuckDB (bronze/silver/gold SQL models)
├── dags/                         # Airflow DAGs (ingest, retrain)
├── frontend/                     # Next.js 15 + Tailwind + recharts
├── tests/                        # Unit + integration + API tests
├── data/                         # Training data (parquet)
├── Dockerfile.api                # Lightweight API image for Render
├── Dockerfile.frontend           # Next.js production build
├── docker-compose.yml            # Full stack: API + frontend + MLflow
├── render.yaml                   # Render deployment blueprint
├── requirements-api.txt          # API-only dependencies (lightweight)
├── requirements.txt              # Full development dependencies
├── requirements-orchestration.txt # Airflow + dbt (install separately)
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
| API | FastAPI + Pydantic v2 | Async, type-safe, auto-documented |
| Frontend | Next.js 15 + Tailwind | Interactive dark-themed dashboard |
| Monitoring | Evidently AI | Data drift + model performance alerts |
| Orchestration | Airflow | DAG scheduling for data + ML pipelines |
| CI/CD | GitHub Actions | Lint, test, Docker build |
| Hosting | Vercel (frontend) + Render (API) | Free tier, auto-deploy from GitHub |

---

## Models

| Model | Type | Purpose |
|-------|------|---------|
| XGBoost | Gradient boosting | **Primary baseline** — best accuracy on tabular commodity data |
| Ridge | Regularized linear | Interpretable coefficients, feature selection |

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
