"""
Unit tests for model training module.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from ames_mlproject.core.exceptions import CustomException
from ames_mlproject.models.training import ModelsConfig, ModelTraining


@pytest.fixture
def dummy_data():
    X = np.random.rand(20, 5)
    y = np.random.rand(20)
    return X, y


@pytest.mark.unit
def test_models_config_loading():
    """Test that ModelsConfig dynamically loads models from config."""
    models_config = ModelsConfig()
    assert len(models_config.model_params) > 0

    # Check if first model is loaded correctly
    estimator, params = models_config.model_params[0]
    assert hasattr(estimator, "fit")
    assert isinstance(params, dict)


@pytest.mark.unit
def test_models_config_load_failure(monkeypatch):
    """Test ModelsConfig handling of invalid model definitions."""
    from ames_mlproject.models import training

    # Mock models list to include an invalid one
    bad_models = [{"name": "Fail", "module": "non_existent", "class": "Nothing", "params": {}}]
    monkeypatch.setattr(training.config.model_training, "models", bad_models)

    models_config = ModelsConfig()
    assert len(models_config.model_params) == 0


@pytest.mark.unit
def test_select_best_model(monkeypatch, dummy_data):
    """Test model selection logic with mock scores."""
    X, y = dummy_data
    trainer = ModelTraining()

    # Mocking cross_val_score to return repeatable results
    monkeypatch.setattr(
        "ames_mlproject.models.training.cross_val_score", lambda **kwargs: np.array([0.9])
    )

    best_config = trainer.select_best_model(X, y)
    assert len(best_config) == 2
    assert hasattr(best_config[0], "fit")


@pytest.mark.unit
def test_tune_model(monkeypatch, dummy_data):
    """Test the tuning process with mocked RandomizedSearchCV."""
    X, y = dummy_data
    trainer = ModelTraining()

    mock_rs = MagicMock()
    mock_rs.best_estimator_ = MagicMock()
    mock_rs.best_params_ = {"alpha": 0.1}
    mock_rs.score.return_value = 0.95

    monkeypatch.setattr(
        "ames_mlproject.models.training.RandomizedSearchCV", lambda **kwargs: mock_rs
    )

    model_config = (MagicMock(), {"alpha": [0.1, 1.0]})
    best_model = trainer.tune_model(X, y, model_config)

    assert best_model == mock_rs.best_estimator_
    mock_rs.fit.assert_called_once()


@pytest.mark.unit
def test_evaluate_model():
    """Test model evaluation on test data."""
    trainer = ModelTraining()
    mock_model = MagicMock()
    mock_model.score.return_value = 0.88

    score = trainer.evaluate_model(np.array([[1]]), np.array([1]), mock_model)
    assert score == 0.88


@pytest.mark.unit
def test_save_model(monkeypatch, tmp_path):
    """Test model serialization."""
    trainer = ModelTraining()
    mock_save = MagicMock()
    monkeypatch.setattr("ames_mlproject.models.training.save_object", mock_save)

    mock_model = MagicMock()
    trainer.save_model(mock_model)

    mock_save.assert_called_once()


@pytest.mark.unit
def test_train_best_model(monkeypatch, dummy_data):
    """Test the full training workflow orchestration."""
    X, y = dummy_data
    trainer = ModelTraining()

    def mock_select(X, y):
        return (MagicMock(), {})

    def mock_tune(X, y, config):
        return MagicMock()

    monkeypatch.setattr(trainer, "select_best_model", mock_select)
    monkeypatch.setattr(trainer, "tune_model", mock_tune)

    model = trainer.train_best_model(X, y)
    assert model is not None


@pytest.mark.unit
def test_select_best_model_exception(monkeypatch, dummy_data):
    """Test exception handling in select_best_model."""
    X, y = dummy_data
    trainer = ModelTraining()
    monkeypatch.setattr(
        "ames_mlproject.models.training.cross_val_score",
        MagicMock(side_effect=Exception("CV Error")),
    )
    with pytest.raises(CustomException):  # CustomException
        trainer.select_best_model(X, y)


@pytest.mark.unit
def test_tune_model_exception(monkeypatch, dummy_data):
    """Test exception handling in tune_model."""
    X, y = dummy_data
    trainer = ModelTraining()
    monkeypatch.setattr(
        "ames_mlproject.models.training.RandomizedSearchCV",
        MagicMock(side_effect=Exception("Tune Error")),
    )
    with pytest.raises(CustomException):
        trainer.tune_model(X, y, (MagicMock(), {}))


@pytest.mark.unit
def test_save_model_exception(monkeypatch):
    """Test exception handling in save_model."""
    trainer = ModelTraining()
    monkeypatch.setattr(
        "ames_mlproject.models.training.save_object", MagicMock(side_effect=Exception("Save Error"))
    )
    with pytest.raises(CustomException):
        trainer.save_model(MagicMock())
