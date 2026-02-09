"""
Unit tests for the Prediction Pipeline.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from ames_mlproject.core.exceptions import CustomException
from ames_mlproject.pipelines.predict import FormData, PredictionPipeline


@pytest.fixture
def valid_input_data():
    return {
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
        "MSSubClass": 60,
        "YearBuilt": 2000,
    }


@pytest.mark.unit
def test_form_data_validation(valid_input_data):
    """Test that FormData correctly validates correct input."""
    form = FormData(valid_input_data)
    validated = form.get_validated_data()
    assert len(validated) == 15
    assert validated["OverallQual"] == 7.0
    assert validated["Neighborhood"] == "CollgCr"


@pytest.mark.unit
def test_form_data_invalid_feature(valid_input_data):
    """Test that FormData raises error on invalid feature value."""
    invalid_data = valid_input_data.copy()
    invalid_data["OverallQual"] = 11  # Invalid (1-10)

    form = FormData(invalid_data)
    with pytest.raises(CustomException):  # CustomException
        form.get_validated_data()


@pytest.mark.unit
def test_prediction_pipeline_predict(valid_input_data, monkeypatch):
    """Test the full prediction pipeline with mocked artifacts."""

    # Mock load_object to return dummy preprocessor and model
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.array([[1.0, 2.0]])

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([250000.0])

    def mock_load(path):
        if "preprocessor" in path:
            return mock_preprocessor
        return mock_model

    monkeypatch.setattr("ames_mlproject.pipelines.predict.load_object", mock_load)

    pipeline = PredictionPipeline()
    result = pipeline.predict(valid_input_data)

    assert result[0] == 250000.0
    mock_preprocessor.transform.assert_called_once()
    mock_model.predict.assert_called_once()
