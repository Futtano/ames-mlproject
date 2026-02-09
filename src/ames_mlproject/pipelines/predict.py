"""
Prediction pipeline orchestration for the Ames ML project.
Handles input data validation and inference using trained model artifacts.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ames_mlproject.core.exceptions import CustomException
from ames_mlproject.core.logging import logger
from ames_mlproject.data.preprocessing import DataPreprocessingConfig
from ames_mlproject.models.training import ModelTrainingConfig
from ames_mlproject.utils.helpers import (
    feat_subset,
    is_valid_cat_feat,
    is_valid_num_feat,
    load_object,
)


class FormData:
    """Wrapper class to handle and validate raw input data from forms/API."""

    def __init__(self, form_data: dict[str, Any]):
        """Initialize FormData with raw input.

        Args:
            form_data (Dict[str, Any]): Key-value pairs of feature names and values.
        """
        self.form_data = form_data

    def get_validated_data(self) -> dict[str, Any]:
        """Validate input data against feature subsets and types.

        Returns:
            Dict[str, Any]: Mapping of feature names to validated/cast values.

        Raises:
            Exception: If feature counts or values are invalid.
            CustomException: Wrapped exception for internal handling.
        """
        logger.info(f"Validating input data: {self.form_data}")
        validated_data = {}
        try:
            if len(self.form_data) != len(feat_subset):
                logger.error(
                    f"Invalid input: expected {len(feat_subset)} features, got {len(self.form_data)}"
                )
                raise ValueError(f"Input data must contain exactly {len(feat_subset)} features.")

            for key, value in self.form_data.items():
                if is_valid_num_feat(key, value):
                    validated_data[key] = np.float64(value)
                elif is_valid_cat_feat(key, value):
                    # Special handling for features that are numeric codes but treated as categories
                    if key in ["MSSubClass", "OverallQual"]:
                        value = np.float64(value)
                    validated_data[key] = value
                else:
                    logger.error(f"Validation failed for feature '{key}' with value '{value}'")
                    raise ValueError(f"'{value}' is not a valid value for feature '{key}'")
        except Exception as e:
            raise CustomException(e) from e

        return validated_data

    def as_dataframe(self) -> pd.DataFrame:
        """Convert validated form data into a single-row pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing a single row of validated features.
        """
        logger.info("Converting validated data to DataFrame for inference.")
        try:
            df = pd.DataFrame([self.get_validated_data()])
            return df
        except Exception as e:
            raise CustomException(e) from e


@dataclass
class PredictionPipelineConfig:
    """Configuration for prediction artifacts.

    Attributes:
        preprocessor_file_path (str): Path to the saved preprocessor object.
        model_file_path (str): Path to the saved model object.
    """

    preprocessor_file_path: str = DataPreprocessingConfig.preprocessor_obj_file_path
    model_file_path: str = ModelTrainingConfig.trained_model_file_path


class PredictionPipeline:
    """Orchestrates inference using saved ML artifacts."""

    def __init__(self):
        """Initialize PredictionPipeline with artifact configuration."""
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def predict(self, form_data: dict[str, Any]) -> np.ndarray:
        """Process input data and generate a prediction.

        Orchestrates validation, data transformation via the preprocessor,
        and inference via the model.

        Args:
            form_data (Dict[str, Any]): Raw feature data.

        Returns:
            np.ndarray: Predicted value(s).

        Raises:
            CustomException: If prediction fails at any stage.
        """
        logger.info("Execution of prediction pipeline started.")
        try:
            # 1. Validate and convert to DataFrame
            X = FormData(form_data).as_dataframe()

            # 2. Load artifacts
            preprocessor = load_object(self.prediction_pipeline_config.preprocessor_file_path)
            model = load_object(self.prediction_pipeline_config.model_file_path)

            # 3. Transform and predict
            X_transformed = preprocessor.transform(X)
            prediction = model.predict(X_transformed)

            logger.info("Inference completed successfully.")
            return prediction

        except Exception as e:
            raise CustomException(e) from e


if __name__ == "__main__":
    sample_data = {
        "OverallQual": 7,
        "ExterQual": "TA",
        "KitchenQual": "Ex",
        "BsmtQual": "Po",
        "MSSubClass": 20,
        "Neighborhood": "SWISU",
        "YearBuilt": 1987,
        "YearRemod/Add": 1989,
        "GrLivArea": 1200,
        "1stFlrSF": 600,
        "TotalBsmtSF": 200,
        "BsmtFinSF1": 200,
        "GarageArea": 100,
        "MasVnrArea": 50,
        "FullBath": 2,
    }

    pipeline = PredictionPipeline()
    result = pipeline.predict(sample_data)
    print(f"Predicted Sale Price: {result[0]:.2f}")
