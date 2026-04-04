"""FastAPI application factory for the Soybean Oil Predictor API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings
from src.log import get_logger, setup_logging
from src.serving.model_cache import load_models
from src.serving.routes import eda, features, health, models, predict, prices

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    setup_logging(Settings().api_log_level)
    logger.info("api_starting", version="0.2.0")
    load_models()
    yield
    logger.info("api_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Soybean Oil Predictor API",
        description=(
            "Commodity price forecasting API for the front-month soybean oil "
            "contract (BOC1). Provides live prices, predictions using XGBoost "
            "and linear models, with SHAP explainability."
        ),
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(prices.router, prefix="/api/v1", tags=["Live Prices"])
    app.include_router(features.router, prefix="/api/v1", tags=["Features"])
    app.include_router(models.router, prefix="/api/v1", tags=["Models"])
    app.include_router(eda.router, prefix="/api/v1", tags=["EDA"])

    return app


app = create_app()
