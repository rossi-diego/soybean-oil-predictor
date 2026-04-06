"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.config import CLEAN_DATA, MODEL_FOLDER
from src.serving.schemas import HealthResponse

router = APIRouter()


def _any_model_exists() -> bool:
    """Check if any trained model file exists."""
    return any(MODEL_FOLDER.glob("*.joblib"))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model availability."""
    return HealthResponse(
        status="healthy",
        version="0.2.0",
        models_loaded=_any_model_exists(),
        data_fresh=CLEAN_DATA.exists(),
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """Kubernetes-style readiness probe."""
    if not _any_model_exists():
        return {"ready": False, "reason": "No trained model found"}
    return {"ready": True}
