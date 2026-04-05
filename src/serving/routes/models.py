"""Model metadata and diagnostics endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from src.config import DATA_FOLDER
from src.log import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _load_json(filename: str) -> dict:
    path = DATA_FOLDER / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not available")
    with open(path) as f:
        return json.load(f)


@router.get("/models")
async def list_models() -> dict:
    """List all models with comparison metrics."""
    return _load_json("model_comparison.json")


@router.get("/model/info")
async def get_model_info() -> dict:
    """Return detailed training metadata for the champion model."""
    return _load_json("model_metadata.json")


@router.get("/models/comparison")
async def get_comparison() -> dict:
    """Full model comparison table."""
    return _load_json("model_comparison.json")


@router.get("/models/champion/diagnostics")
async def get_champion_diagnostics() -> dict:
    """Residuals and predicted vs actual for the champion model."""
    return _load_json("champion_diagnostics.json")


@router.get("/models/champion/feature-importance")
async def get_feature_importance() -> dict:
    """Feature importance for all models."""
    return _load_json("feature_importance.json")


@router.get("/models/learning-curves")
async def get_learning_curves() -> dict:
    """Training vs validation error as function of training size."""
    return _load_json("learning_curves.json")


@router.get("/models/walk-forward")
async def get_walk_forward() -> dict:
    """Per-fold walk-forward metrics for all models."""
    return _load_json("walk_forward_folds.json")
