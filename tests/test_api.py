"""Tests for the FastAPI serving layer."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app


@pytest.fixture
def client():
    """Create a test client for the API."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.2.0"

    def test_readiness_check(self, client):
        response = client.get("/ready")
        assert response.status_code == 200


class TestPredictionEndpoints:
    def test_predict_valid_input(self, client):
        payload = {
            "smc1": 350.0,
            "sc1": 1100.0,
            "lcoc1": 75.0,
            "hoc1": 2.20,
            "palm_oil": 3200.0,
            "rsc1": 600.0,
            "month": 3,
        }
        response = client.post("/api/v1/predict", json=payload)
        # May return 503 if no model is loaded, which is valid
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = response.json()
            assert "predicted_price" in data
            assert "model_name" in data
            assert isinstance(data["predicted_price"], float)

    def test_predict_invalid_input_negative_price(self, client):
        payload = {
            "smc1": -100.0,
            "sc1": 1100.0,
            "lcoc1": 75.0,
            "hoc1": 2.20,
            "palm_oil": 3200.0,
            "rsc1": 600.0,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_missing_required_field(self, client):
        payload = {"smc1": 350.0}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422


class TestFeatureEndpoints:
    def test_feature_stats(self, client):
        response = client.get("/api/v1/features/stats")
        assert response.status_code in (200, 404)

    def test_feature_importance(self, client):
        response = client.get("/api/v1/features/importance")
        assert response.status_code in (200, 404)


class TestModelEndpoints:
    def test_list_models(self, client):
        response = client.get("/api/v1/models")
        assert response.status_code in (200, 404)

        if response.status_code == 200:
            data = response.json()
            assert "models" in data
            assert "champion" in data or "active_model" in data
