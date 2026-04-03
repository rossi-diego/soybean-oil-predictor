"""Feature-related endpoints — statistics and importance."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.config import CLEAN_DATA, FEATURES_DATA, XGBOOST_MODEL
from src.log import get_logger
from src.serving.schemas import (
    FeatureImportance,
    FeatureImportanceResponse,
    FeatureStatsResponse,
)

logger = get_logger(__name__)

router = APIRouter()


@router.get("/features/stats", response_model=FeatureStatsResponse)
async def feature_stats() -> FeatureStatsResponse:
    """Return descriptive statistics of training features.

    Useful for the frontend to show input ranges and typical values.
    """
    if FEATURES_DATA.exists():
        df = pd.read_csv(FEATURES_DATA, index_col=0)
        stats = df.to_dict()
    elif CLEAN_DATA.exists():
        df = pd.read_parquet(CLEAN_DATA)
        stats = df.describe().to_dict()
    else:
        raise HTTPException(status_code=404, detail="No training data found")

    return FeatureStatsResponse(features=stats)


@router.get("/features/importance", response_model=FeatureImportanceResponse)
async def feature_importance() -> FeatureImportanceResponse:
    """Return feature importance from the active model.

    Uses XGBoost's built-in feature importance if available,
    otherwise computes coefficient magnitudes from linear models.
    """
    if XGBOOST_MODEL.exists():
        import joblib

        model = joblib.load(XGBOOST_MODEL)
        if hasattr(model, "feature_importances_"):
            names = model.get_booster().feature_names or [
                f"f{i}" for i in range(len(model.feature_importances_))
            ]
            importances = [
                FeatureImportance(feature=name, importance=float(imp))
                for name, imp in zip(names, model.feature_importances_)
            ]
            importances.sort(key=lambda x: x.importance, reverse=True)
            return FeatureImportanceResponse(
                model_name="xgboost_baseline",
                importances=importances,
            )

    if CLEAN_DATA.exists():
        from src.utils import load_model

        try:
            model = load_model()
            pipe = getattr(model, "regressor", model)
            reg = pipe.named_steps.get("reg")
            if reg and hasattr(reg, "coef_"):
                df = pd.read_parquet(CLEAN_DATA)
                feature_names = [c for c in df.columns if c != "boc1"]
                coefs = reg.coef_
                importances = [
                    FeatureImportance(feature=name, importance=float(abs(c)))
                    for name, c in zip(feature_names, coefs)
                ]
                importances.sort(key=lambda x: x.importance, reverse=True)
                return FeatureImportanceResponse(
                    model_name="linear_regression",
                    importances=importances,
                )
        except Exception:
            logger.exception("feature_importance_failed")

    raise HTTPException(status_code=404, detail="No model with feature importance found")
