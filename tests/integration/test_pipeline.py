"""
Integration tests for data ingestion and preprocessing pipeline.
"""

import pytest


@pytest.mark.integration
def test_data_ingestion_preprocessing_pipeline(tmp_path, sample_housing_data):
    """Test the data ingestion and preprocessing components."""
    from ames_mlproject.config import get_config
    from ames_mlproject.data.ingestion import DataIngestion
    from ames_mlproject.data.preprocessing import DataPreprocessing

    # Verify components can be instantiated
    ingestion = DataIngestion()
    assert ingestion.ingestion_config is not None

    preprocessing = DataPreprocessing()
    assert preprocessing is not None

    # Test preprocessing with sample data
    config = get_config()
    X_sample = sample_housing_data.drop(columns=[config.data.target_feature])
    y_sample = sample_housing_data[config.data.target_feature]

    preprocessor = preprocessing.get_preprocessor()
    assert preprocessor is not None

    # Verify preprocessing can transform data
    X_transformed = preprocessor.fit_transform(X_sample, y_sample)
    assert X_transformed is not None
    assert X_transformed.shape[0] == len(sample_housing_data)
    assert X_transformed.shape[1] > 0


@pytest.mark.integration
@pytest.mark.slow
def test_full_training_pipeline(tmp_path, sample_housing_data):
    """Test the complete training pipeline instantiation."""
    from ames_mlproject.pipelines.train import TrainPipeline

    # For now, just verify the pipeline can be instantiated
    pipeline = TrainPipeline()
    assert pipeline is not None


@pytest.mark.integration
def test_config_integration_with_components(tmp_path):
    """Test that all components properly use configuration."""
    from ames_mlproject.config import get_config
    from ames_mlproject.data.ingestion import DataIngestion
    from ames_mlproject.data.preprocessing import DataPreprocessing
    from ames_mlproject.models.training import ModelTraining

    config = get_config()

    # Verify each component uses config
    ingestion = DataIngestion()
    assert ingestion.ingestion_config is not None

    preprocessing = DataPreprocessing()
    assert preprocessing.data_preprocessing_config is not None

    training = ModelTraining()
    assert training.model_training_config.trained_model_file_path == config.artifacts.model_path
