"""
Pytest configuration and shared fixtures.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_housing_data():
    """Create sample housing data for testing."""
    return pd.DataFrame(
        {
            "OverallQual": [7, 8, 6, 9, 7],
            "GrLivArea": [1500, 2000, 1200, 2500, 1800],
            "BsmtQual": ["TA", "Gd", "Ex", "Gd", "TA"],
            "Neighborhood": ["SWISU", "CollgCr", "NAmes", "NoRidge", "Gilbert"],
            "KitchenQual": ["Gd", "Ex", "TA", "Ex", "Gd"],
            "BsmtFinSF1": [500, 800, 400, 1000, 600],
            "TotalBsmtSF": [1000, 1500, 800, 2000, 1200],
            "1stFlrSF": [1000, 1500, 800, 2000, 1200],
            "GarageArea": [400, 500, 300, 600, 450],
            "FullBath": [2, 3, 2, 3, 2],
            "MasVnrArea": [100, 200, 0, 300, 150],
            "ExterQual": ["TA", "Gd", "TA", "Ex", "Gd"],
            "YearRemod/Add": [2000, 2005, 1995, 2010, 2003],
            "MSSubClass": [20, 20, 30, 60, 20],
            "YearBuilt": [1995, 2000, 1990, 2005, 1998],
            "SalePrice": [200000, 300000, 180000, 450000, 250000],
        }
    )


@pytest.fixture
def sample_prediction_input():
    """Create sample input for prediction testing."""
    return {
        "OverallQual": 7,
        "GrLivArea": 1500.0,
        "BsmtQual": "TA",
        "Neighborhood": "SWISU",
        "KitchenQual": "Gd",
        "BsmtFinSF1": 500.0,
        "TotalBsmtSF": 1000.0,
        "1stFlrSF": 1000.0,
        "GarageArea": 400.0,
        "FullBath": 2,
        "MasVnrArea": 100.0,
        "ExterQual": "TA",
        "YearRemod/Add": 2000,
        "MSSubClass": 20,
        "YearBuilt": 1995,
    }


@pytest.fixture
def temp_artifact_dir(tmp_path):
    """Create temporary artifacts directory for testing."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    return artifact_dir


@pytest.fixture
def mock_config(tmp_path):
    """Create mock configuration for testing."""
    from ames_mlproject.config import (
        APIConfig,
        ArtifactsConfig,
        Config,
        DataConfig,
        GeneralConfig,
        LoggingConfig,
        ModelTrainingConfig,
        PreprocessingConfig,
        ValidationConfig,
    )

    return Config(
        general=GeneralConfig(random_state=42, environment="test"),
        data=DataConfig(
            dataset_path=str(tmp_path / "test_data.csv"),
            separator=",",
            test_size=0.3,
            shuffle=True,
            target_feature="SalePrice",
            feature_subset=[
                "OverallQual",
                "GrLivArea",
                "BsmtQual",
                "Neighborhood",
                "KitchenQual",
                "BsmtFinSF1",
                "TotalBsmtSF",
                "1stFlrSF",
                "GarageArea",
                "FullBath",
                "MasVnrArea",
                "ExterQual",
                "YearRemod/Add",
                "MSSubClass",
                "YearBuilt",
            ],
        ),
        artifacts=ArtifactsConfig(
            base_path=str(tmp_path / "artifacts"),
            train_data="train.csv",
            test_data="test.csv",
            clean_data="data.csv",
            preprocessor="preprocessor.pkl",
            model="model.pkl",
        ),
        logging=LoggingConfig(
            log_dir=str(tmp_path / "logs"), format="%(levelname)s - %(message)s", level="INFO"
        ),
        preprocessing=PreprocessingConfig(
            imputation={"numerical_strategy": "median", "categorical_strategy": "most_frequent"},
            encoding={"target_encoder_cv": 5, "target_encoder_smooth": "auto"},
        ),
        model_training=ModelTrainingConfig(
            scoring="r2",
            cv_folds=2,  # Reduced for faster tests
            random_search_iter=2,  # Reduced for faster tests
            random_search_cv=2,
            n_jobs=1,
            nested_cv_outer_folds=2,
            nested_cv_inner_folds=2,
            xgboost_params={"n_estimators": [10, 20], "max_depth": [3, 5]},
        ),
        api=APIConfig(host="127.0.0.1", port=5000, debug=False, cors_origins=["*"]),
        validation=ValidationConfig(outlier_threshold={"GrLivArea": 4000}),
    )
