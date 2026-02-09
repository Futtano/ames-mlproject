"""
Unit tests for utility functions.
"""
import pytest
from ames_mlproject.utils.helpers import (
    is_valid_cat_feat,
    is_valid_num_feat,
    load_object,
    save_object,
)


@pytest.mark.unit
def test_save_and_load_object(tmp_path):
    """Test saving and loading Python objects."""
    # Create a test object
    test_dict = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": "dict"}}

    # Save object
    file_path = tmp_path / "test_object.pkl"
    save_object(str(file_path), test_dict)

    # Load object
    loaded_dict = load_object(str(file_path))

    # Verify
    assert loaded_dict == test_dict
    assert loaded_dict["key1"] == "value1"
    assert loaded_dict["key2"] == [1, 2, 3]


@pytest.mark.unit
def test_is_valid_num_feat():
    """Test numerical feature validation."""
    # Valid cases
    assert bool(is_valid_num_feat("GrLivArea", 1500.0)) is True
    assert bool(is_valid_num_feat("YearBuilt", 2000)) is True
    assert bool(is_valid_num_feat("FullBath", 2)) is True

    # Invalid cases - negative values where not allowed
    assert bool(is_valid_num_feat("GrLivArea", -100)) is False
    assert bool(is_valid_num_feat("YearBuilt", 1700)) is False  # Too old


@pytest.mark.unit
def test_is_valid_cat_feat():
    """Test categorical feature validation."""
    # Valid cases
    assert bool(is_valid_cat_feat("BsmtQual", "Ex")) is True
    assert bool(is_valid_cat_feat("BsmtQual", "Gd")) is True
    assert bool(is_valid_cat_feat("BsmtQual", "TA")) is True
    assert bool(is_valid_cat_feat("Neighborhood", "SWISU")) is True
    assert bool(is_valid_cat_feat("OverallQual", 7)) is True  # Special case - numeric category

    # Invalid cases
    assert bool(is_valid_cat_feat("BsmtQual", "InvalidValue")) is False
    assert bool(is_valid_cat_feat("Neighborhood", "NonExistent")) is False
    assert bool(is_valid_cat_feat("OverallQual", 11)) is False  # Out of range (1-10)


@pytest.mark.unit
def test_feature_subsets_consistency():
    """Test that feature subset lists are consistent."""
    from ames_mlproject.utils.helpers import cat_nom, cat_ord, feat_subset, num

    # All features should be accounted for
    all_features = num + cat_ord + cat_nom

    # Feature subset should match
    assert set(feat_subset) == set(all_features)

    # No duplicates
    assert len(feat_subset) == len(all_features)


@pytest.mark.unit
def test_parse_hyperparameters():
    """Test hyperparameter parsing from config dict."""
    from ames_mlproject.utils.helpers import parse_hyperparameters
    from scipy.stats import randint, uniform

    params = {
        "alpha": {"dist": "uniform", "args": [0.0001, 10]},
        "max_depth": {"dist": "randint", "args": [3, 20]},
        "n_estimators": [100, 200, 500],
        "simple_val": 10,
    }

    parsed = parse_hyperparameters(params)

    # Check distributions
    assert isinstance(parsed["alpha"], type(uniform(0, 1)))
    assert isinstance(parsed["max_depth"], type(randint(0, 1)))

    # Check fixed values
    assert parsed["n_estimators"] == [100, 200, 500]
    assert parsed["simple_val"] == 10


@pytest.mark.unit
def test_custom_exception():
    """Test CustomException formatting."""
    import sys

    from ames_mlproject.core.exceptions import CustomException

    try:
        raise ValueError("Original Error")
    except Exception as e:
        ce = CustomException(e, sys)
        message = str(ce)
        assert "ValueError" in message
        assert "Original Error" in message
        assert "test_utils.py" in message
