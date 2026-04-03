"""Prediction endpoint — core business logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.log import get_logger
from src.serving.model_cache import get_model
from src.serving.schemas import PredictionRequest, PredictionResponse

logger = get_logger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate a BOC1 price prediction.

    Accepts current commodity prices and returns the predicted
    soybean oil futures price with model metadata.
    """
    model, model_name = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model loaded")

    input_data = pd.DataFrame([{
        "smc1": request.smc1,
        "sc1": request.sc1,
        "lcoc1": request.lcoc1,
        "hoc1": request.hoc1,
        "fcpoc1": request.fcpoc1,
        "rsc1": request.rsc1,
    }])

    try:
        prediction = model.predict(input_data)
        predicted_price = float(np.ravel(prediction)[0])
    except Exception as e:
        logger.exception("prediction_failed", model=model_name)
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {e}"
        ) from e

    logger.info(
        "prediction_made",
        model=model_name,
        price=predicted_price,
    )

    return PredictionResponse(
        predicted_price=round(predicted_price, 2),
        model_name=model_name,
        features_used=list(input_data.columns),
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
