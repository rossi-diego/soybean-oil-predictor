"""Prediction endpoint — core business logic."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.config import DATA_FOLDER, FEATURE_COLUMNS
from src.log import get_logger
from src.serving.model_cache import get_model
from src.serving.schemas import PredictionRequest, PredictionResponse

logger = get_logger(__name__)

router = APIRouter()

_metadata_cache: dict | None = None


def _load_metadata() -> dict:
    """Load model metadata (cached after first read)."""
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache
    path = DATA_FOLDER / "model_metadata.json"
    if path.exists():
        with open(path) as f:
            _metadata_cache = json.load(f)
    else:
        _metadata_cache = {}
    return _metadata_cache


def _compute_contributions(model, input_data: pd.DataFrame) -> list[dict]:
    """Extract per-feature contributions using XGBoost's built-in SHAP.

    XGBoost's predict with pred_contribs=True returns SHAP-like values
    without requiring the shap library.
    """
    try:
        import xgboost as xgb

        if not isinstance(model, xgb.XGBRegressor):
            return []

        booster = model.get_booster()
        dmatrix = xgb.DMatrix(input_data, feature_names=list(input_data.columns))
        contribs = booster.predict(dmatrix, pred_contribs=True)

        if contribs.ndim == 2:
            values = contribs[0]
            feature_names = list(input_data.columns) + ["base_value"]
            result = []
            for name, val in zip(feature_names[:-1], values[:-1]):
                result.append({"feature": name, "contribution": round(float(val), 4)})
            result.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            return result
    except Exception:
        logger.debug("contributions_unavailable")
    return []


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate a BOC1 price prediction with confidence interval and feature contributions."""
    model, model_name = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model loaded")

    raw = request.model_dump()
    input_data = pd.DataFrame([{k: raw[k] for k in FEATURE_COLUMNS if k in raw}])

    try:
        prediction = model.predict(input_data)
        predicted_price = float(np.ravel(prediction)[0])
    except Exception as e:
        logger.exception("prediction_failed", model=model_name)
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {e}"
        ) from e

    # Confidence interval from stored residual std
    metadata = _load_metadata()
    residual_std = metadata.get("residual_std", 0)
    confidence = {}
    if residual_std > 0:
        confidence = {
            "lower": round(predicted_price - 1.96 * residual_std, 2),
            "upper": round(predicted_price + 1.96 * residual_std, 2),
        }

    # Feature contributions (XGBoost native SHAP)
    contributions = _compute_contributions(model, input_data)

    logger.info("prediction_made", model=model_name, price=predicted_price)

    return PredictionResponse(
        predicted_price=round(predicted_price, 2),
        model_name=model_name,
        confidence=confidence,
        features_used=list(input_data.columns),
        feature_contributions=contributions,
    )


@router.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]) -> list[PredictionResponse]:
    """Generate predictions for multiple input rows."""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")

    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return results
