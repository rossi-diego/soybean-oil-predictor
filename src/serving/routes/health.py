"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.config import CLEAN_DATA, LINEAR_REGRESSION_MODEL, XGBOOST_MODEL
from src.serving.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model availability."""
    models_loaded = (
        LINEAR_REGRESSION_MODEL.exists() or XGBOOST_MODEL.exists()
    )
    data_fresh = CLEAN_DATA.exists()

    return HealthResponse(
        status="healthy",
        version="0.2.0",
        models_loaded=models_loaded,
        data_fresh=data_fresh,
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """Kubernetes-style readiness probe."""
    if not (LINEAR_REGRESSION_MODEL.exists() or XGBOOST_MODEL.exists()):
        return {"ready": False, "reason": "No trained model found"}
    return {"ready": True}
