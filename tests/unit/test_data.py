"""
Unit tests for the Data Ingestion and Preprocessing modules.
"""
import os

import pandas as pd
import pytest
from ames_mlproject.config import get_config
from ames_mlproject.data.ingestion import DataIngestion
from ames_mlproject.data.preprocessing import DataPreprocessing


@pytest.fixture
def sample_raw_data(tmp_path):
    """Create a sample dummy dataset for ingestion tests (10 rows for clean splits)."""
    # Use only raw names as per ingestion logic
    data = {
        "MS SubClass": [20] * 10,
        "Overall Qual": [7] * 10,
        "Gr Liv Area": [1500] * 10,
        "Year Built": [2000] * 10,
        "Year Remod/Add": [2005] * 10,
        "Yr Sold": [2008] * 10,
        "SalePrice": [200000] * 10,
        "1st Flr SF": [800] * 10,
        "2nd Flr SF": [700] * 10,
        "Total Bsmt SF": [800] * 10,
        "BsmtFin SF 1": [400] * 10,
        "BsmtFin SF 2": [0] * 10,
        "Bsmt Unf SF": [400] * 10,
        "Neighborhood": ["CollgCr"] * 10,
        "Bsmt Qual": ["Gd"] * 10,
        "Kitchen Qual": ["Gd"] * 10,
        "Exter Qual": ["Gd"] * 10,
        "MS Zoning": ["RL"] * 10,
        "Bldg Type": ["1Fam"] * 10,
        "Exterior 1st": ["VinylSd"] * 10,
        "Exterior 2nd": ["VinylSd"] * 10,
        "Sale Type": ["WD"] * 10,
        "Garage Area": [500] * 10,
        "Full Bath": [2] * 10,
        "Mas Vnr Area": [100] * 10,
    }
    df = pd.DataFrame(data)

    # Add any missing features from the subset (using raw names if possible, but the pipeline handle simple renames well)
    from ames_mlproject.utils.helpers import name_map

    reverse_map = {v: k for k, v in name_map.items()}

    config = get_config()
    for feat in config.data.feature_subset:
        raw_feat = reverse_map.get(feat, feat)
        if raw_feat not in df.columns:
            df[raw_feat] = 0

    data_path = tmp_path / "AmesHousing.txt"
    df.to_csv(data_path, sep="\t", index=False)
    return data_path


@pytest.mark.unit
def test_data_ingestion_run(sample_raw_data, monkeypatch, tmp_path):
    """Test the data ingestion run logic."""
    config = get_config()

    # Override paths to use temporary directory
    monkeypatch.setattr(config.data, "dataset_path", str(sample_raw_data))
    monkeypatch.setattr(config.artifacts, "base_path", str(tmp_path / "artifacts"))

    ingestor = DataIngestion()
    train_path, test_path = ingestor.ingest()

    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    df_train = pd.read_csv(train_path)
    assert not df_train.empty
    assert "SalePrice" in df_train.columns


@pytest.mark.unit
def test_data_preprocessing_pipeline(tmp_path):
    """Test the preprocessing pipeline building and execution."""
    df = pd.DataFrame(
        {
            "OverallQual": [7] * 10,
            "GrLivArea": [1500] * 10,
            "BsmtQual": ["Gd"] * 10,
            "Neighborhood": ["CollgCr"] * 10,
            "KitchenQual": ["Gd"] * 10,
            "BsmtFinSF1": [400] * 10,
            "TotalBsmtSF": [800] * 10,
            "1stFlrSF": [800] * 10,
            "GarageArea": [500] * 10,
            "FullBath": [2] * 10,
            "MasVnrArea": [100] * 10,
            "ExterQual": ["Gd"] * 10,
            "YearRemod/Add": [2005] * 10,
            "MSSubClass": [20] * 10,
            "YearBuilt": [2000] * 10,
            "SalePrice": [200000] * 10,
        }
    )

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    df.to_csv(train_path, index=False)
    df.to_csv(test_path, index=False)

    preprocessor = DataPreprocessing()
    X_train, X_test, y_train, y_test, obj_path = preprocessor.preprocess(
        str(train_path), str(test_path)
    )

    assert X_train.shape[0] == 10
    assert y_train.shape[0] == 10
    assert os.path.exists(obj_path)
