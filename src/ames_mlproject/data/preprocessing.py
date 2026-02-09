"""
Data preprocessing module for the Ames ML project.
Handles feature transformations, imputation, and encoding using scikit-learn pipelines.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from ames_mlproject.config import get_config
from ames_mlproject.core.exceptions import CustomException
from ames_mlproject.core.logging import logger
from ames_mlproject.data.ingestion import DataIngestion
from ames_mlproject.utils.helpers import cat_nom, cat_ord, enc_ord_categories, num, save_object

# Load configuration
config = get_config()


@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing artifacts.

    Attributes:
        preprocessor_obj_file_path (str): Path where the preprocessor pickle will be saved.
    """

    preprocessor_obj_file_path: str = config.artifacts.preprocessor_path


class DataPreprocessing:
    """Class to manage data transformation pipelines and execution."""

    def __init__(self):
        """Initialize DataPreprocessing with configuration."""
        self.data_preprocessing_config = DataPreprocessingConfig()

    def get_preprocessor(self) -> ColumnTransformer:
        """Create and return a configured ColumnTransformer preprocessor.

        Sets up pipelines for numerical imputation, nominal categorical encoding
        (TargetEncoder), and ordinal categorical encoding (OrdinalEncoder).

        Returns:
            ColumnTransformer: Prepared scikit-learn preprocessor.
        """
        logger.info("Building preprocessor pipelines.")

        # Pipeline for nominal categorical features (neighborhoods, etc.)
        cat_nom_pipeline = Pipeline(
            [
                (
                    "impute_most_freq",
                    SimpleImputer(strategy=config.preprocessing.imputation["categorical_strategy"]),
                ),
                (
                    "encode_nom",
                    TargetEncoder(
                        cv=config.preprocessing.encoding["target_encoder_cv"],
                        smooth=config.preprocessing.encoding["target_encoder_smooth"],
                        shuffle=True,
                        random_state=config.general.random_state,
                        target_type="continuous",
                    ),
                ),
            ]
        )

        # Pipeline for ordinal categorical features (qualities, etc.)
        cat_ord_pipeline = Pipeline(
            [
                (
                    "impute_most_freq",
                    SimpleImputer(strategy=config.preprocessing.imputation["categorical_strategy"]),
                ),
                ("encode_ord", OrdinalEncoder(categories=enc_ord_categories)),
            ]
        )

        # Combine all features into one transformer
        preprocessor = ColumnTransformer(
            [
                (
                    "numerical",
                    SimpleImputer(strategy=config.preprocessing.imputation["numerical_strategy"]),
                    num,
                ),
                ("cat_nominal", cat_nom_pipeline, cat_nom),
                ("cat_ordinal", cat_ord_pipeline, cat_ord),
            ]
        )

        return preprocessor

    def preprocess(
        self, train_path: str, test_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """Execute the preprocessing pipeline on train and test datasets.

        Reads datasets, applies transformations, saves the preprocessor object,
        and returns transformed feature arrays and target labels.

        Args:
            train_path (str): Path to the training CSV file.
            test_path (str): Path to the test CSV file.

        Returns:
            Tuple: (X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_path)

        Raises:
            CustomException: If any error occurs during preprocessing.
        """
        logger.info("Data Preprocessing execution started.")
        try:
            df_train = pd.read_csv(train_path)
            logger.info(f"Loaded train set from {train_path}")

            df_test = pd.read_csv(test_path)
            logger.info(f"Loaded test set from {test_path}")

            # Separate features and target
            X_train = df_train.drop(columns=config.data.target_feature)
            y_train = df_train[config.data.target_feature].to_numpy()

            X_test = df_test.drop(columns=config.data.target_feature)
            y_test = df_test[config.data.target_feature].to_numpy()

            logger.info("Applying transformations.")
            preprocessor = self.get_preprocessor()

            # Fit on training data and transform both sets
            X_train_transformed = preprocessor.fit_transform(X_train, y_train)
            X_test_transformed = preprocessor.transform(X_test)
            logger.info("Preprocessing pipeline transformations completed.")

            # Save the preprocessor object for future predictions
            save_object(self.data_preprocessing_config.preprocessor_obj_file_path, preprocessor)
            logger.info(
                f"Preprocessor object saved to {self.data_preprocessing_config.preprocessor_obj_file_path}"
            )

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_preprocessing_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e) from e


if __name__ == "__main__":
    ingestor = DataIngestion()
    preprocessor = DataPreprocessing()

    train_file, test_file = ingestor.ingest()
    X_tr, X_te, y_tr, y_te, _ = preprocessor.preprocess(train_file, test_file)

    print(f"X_train shape: {X_tr.shape}")
    print(f"First row of transformed features: {X_tr[0]}")
