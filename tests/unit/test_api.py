"""
Unit tests for the FastAPI Web API.
"""
import pytest
from ames_mlproject.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.mark.unit
def test_health_check_endpoint():
    """Test the health check endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.unit
def test_predict_endpoint_validation():
    """Test the predict endpoint returns 422 on invalid input data."""
    # Sending empty or malformed data to trigger validation error
    response = client.post("/api/predict", json={})
    assert response.status_code == 422


@pytest.mark.unit
def test_predict_endpoint_success(monkeypatch):
    """Test the predict endpoint returns 200 and a prediction (mocked pipeline)."""

    # Mock the PredictionPipeline.predict method to avoid loading binary artifacts
    def mock_predict(self, data):
        return [250000.0]

    monkeypatch.setattr("ames_mlproject.pipelines.predict.PredictionPipeline.predict", mock_predict)
    # Also mock constructor to avoid loading model file
    monkeypatch.setattr(
        "ames_mlproject.pipelines.predict.PredictionPipeline.__init__", lambda self: None
    )

    # Sample valid request data according to schemas.py
    valid_data = {
        "OverallQual": 7,
        "GrLivArea": 1500.0,
        "BsmtQual": "Gd",
        "Neighborhood": "CollgCr",
        "KitchenQual": "Gd",
        "BsmtFinSF1": 400.0,
        "TotalBsmtSF": 800.0,
        "1stFlrSF": 800.0,
        "GarageArea": 500.0,
        "FullBath": 2,
        "MasVnrArea": 100.0,
        "ExterQual": "Gd",
        "YearRemod/Add": 2005,
        "MSSubClass": 20,
        "YearBuilt": 2000,
    }

    response = client.post("/api/predict", json=valid_data)

    assert response.status_code == 200
    assert "SalePrice" in response.json()
    assert response.json()["SalePrice"] == 250000.0
