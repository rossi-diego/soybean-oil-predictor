"""Pydantic v2 request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for BOC1 price prediction."""

    smc1: float = Field(..., gt=0, description="Soybean Meal futures price ($/short ton)")
    sc1: float = Field(..., gt=0, description="Soybean futures price (cents/bushel)")
    lcoc1: float = Field(..., gt=0, description="Brent Crude Oil futures price ($/barrel)")
    hoc1: float = Field(..., gt=0, description="Heating Oil futures price ($/gallon)")
    fcpoc1: float = Field(..., gt=0, description="Crude Palm Oil futures price (MYR/tonne)")
    rsc1: float = Field(..., gt=0, description="Rapeseed/Canola futures price (CAD/tonne)")
    month: int = Field(default=1, ge=1, le=12, description="Month of the year (1-12)")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "smc1": 350.0,
                "sc1": 1100.0,
                "lcoc1": 75.0,
                "hoc1": 2.20,
                "fcpoc1": 3200.0,
                "rsc1": 600.0,
                "month": 3,
            }
        ]
    }}


class PredictionResponse(BaseModel):
    """Response body with BOC1 price prediction."""

    predicted_price: float = Field(..., description="Predicted BOC1 price (cents/lb)")
    model_name: str = Field(..., description="Name of the model used")
    confidence: dict = Field(
        default_factory=dict,
        description="Confidence interval (lower, upper) if available",
    )
    features_used: list[str] = Field(
        default_factory=list,
        description="Features that contributed to the prediction",
    )
    feature_contributions: list[dict] = Field(
        default_factory=list,
        description="Per-feature contributions to this prediction (SHAP-like)",
    )


class FeatureImportance(BaseModel):
    """Single feature importance entry."""

    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    """Response with ranked feature importances."""

    model_name: str
    importances: list[FeatureImportance]


class ModelInfo(BaseModel):
    """Metadata about a trained model."""

    name: str
    model_type: str
    metrics: dict[str, float]
    n_features: int
    trained_at: str | None = None


class ModelListResponse(BaseModel):
    """Response listing all available models."""

    models: list[ModelInfo]
    active_model: str


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "healthy"
    version: str = "0.2.0"
    models_loaded: bool = False
    data_fresh: bool = False


class FeatureStatsResponse(BaseModel):
    """Response with training data feature statistics."""

    features: dict[str, dict[str, float]]
