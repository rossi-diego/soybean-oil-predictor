"""Model metadata endpoints."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.config import (
    LINEAR_REGRESSION_MODEL,
    RESULTS_DATA,
    XGBOOST_MODEL,
)
from src.log import get_logger
from src.serving.schemas import ModelInfo, ModelListResponse

logger = get_logger(__name__)

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List all available trained models with their metrics."""
    models_list: list[ModelInfo] = []
    active_model = "none"

    if XGBOOST_MODEL.exists():
        import joblib

        model = joblib.load(XGBOOST_MODEL)
        n_features = getattr(model, "n_features_in_", 0)
        modified = datetime.fromtimestamp(XGBOOST_MODEL.stat().st_mtime)

        models_list.append(ModelInfo(
            name="xgboost_baseline",
            model_type="XGBRegressor",
            metrics={},
            n_features=n_features,
            trained_at=modified.isoformat(),
        ))
        active_model = "xgboost_baseline"

    if LINEAR_REGRESSION_MODEL.exists():
        import joblib

        try:
            model = joblib.load(LINEAR_REGRESSION_MODEL)
        except Exception:
            logger.warning("linear_model_load_failed", path=str(LINEAR_REGRESSION_MODEL))
            model = None

        n_features = 0
        if model is not None:
            pipe = getattr(model, "regressor", model)
            reg = pipe.named_steps.get("reg")
            n_features = len(reg.coef_) if reg and hasattr(reg, "coef_") else 0

        modified = datetime.fromtimestamp(LINEAR_REGRESSION_MODEL.stat().st_mtime)

        metrics = {}
        if RESULTS_DATA.exists():
            results_df = pd.read_parquet(RESULTS_DATA)
            if "model" in results_df.columns:
                lr_results = results_df[results_df["model"].str.contains("Linear", case=False, na=False)]
                if not lr_results.empty and "test_r2" in lr_results.columns:
                    metrics["r2"] = float(lr_results["test_r2"].mean())

        models_list.append(ModelInfo(
            name="linear_regression",
            model_type="LinearRegression",
            metrics=metrics,
            n_features=n_features,
            trained_at=modified.isoformat(),
        ))

        if active_model == "none":
            active_model = "linear_regression"

    if not models_list:
        raise HTTPException(status_code=404, detail="No trained models found")

    return ModelListResponse(models=models_list, active_model=active_model)


@router.get("/model/info")
async def get_model_info() -> dict:
    """Return detailed training metadata for the active model."""
    import json

    from src.config import DATA_FOLDER

    metadata_path = DATA_FOLDER / "model_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not available")

    with open(metadata_path) as f:
        return json.load(f)
