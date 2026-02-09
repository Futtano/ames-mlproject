"""
Unit tests for configuration module.
"""

import pytest
from ames_mlproject.config import get_config, reload_config


@pytest.mark.unit
def test_config_from_yaml():
    """Test loading configuration from YAML file."""
    config = get_config()

    # Check general config
    assert config.general.random_state == 42
    assert config.general.environment in ["development", "staging", "production"]

    # Check data config
    assert config.data.test_size == 0.3
    assert config.data.shuffle is True
    assert config.data.target_feature == "SalePrice"
    assert len(config.data.feature_subset) == 15

    # Check artifacts config
    assert config.artifacts.base_path == "artifacts"
    assert config.artifacts.train_data == "train.csv"

    # Check model training config
    assert config.model_training.scoring == "r2"
    assert config.model_training.cv_folds == 5
    assert config.model_training.n_jobs == -1

    # Check dynamic models config
    assert isinstance(config.model_training.models, list)
    assert len(config.model_training.models) > 0
    first_model = config.model_training.models[0]
    assert "name" in first_model
    assert "module" in first_model
    assert "class" in first_model
    assert "params" in first_model


@pytest.mark.unit
def test_artifact_paths():
    """Test artifact path properties."""
    config = get_config()

    assert config.artifacts.train_data_path == "artifacts/train.csv"
    assert config.artifacts.test_data_path == "artifacts/test.csv"
    assert config.artifacts.clean_data_path == "artifacts/data.csv"
    assert config.artifacts.preprocessor_path == "artifacts/preprocessor.pkl"
    assert config.artifacts.model_path == "artifacts/model.pkl"


@pytest.mark.unit
def test_config_singleton():
    """Test that get_config returns the same instance."""
    config1 = get_config()
    config2 = get_config()

    assert config1 is config2


@pytest.mark.unit
def test_config_reload():
    """Test configuration reload."""
    config1 = get_config()
    config2 = reload_config()

    # After reload, should be a new instance
    assert config1 is not config2

    # But with same values
    assert config1.general.random_state == config2.general.random_state
