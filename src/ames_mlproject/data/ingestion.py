"""
Data ingestion module for the Ames ML project.
Handles loading raw data, cleaning, validation, and splitting into train/test sets.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ames_mlproject.config import get_config
from ames_mlproject.core.exceptions import CustomException
from ames_mlproject.core.logging import logger
from ames_mlproject.utils.helpers import (
    bldgtype_correct_values,
    exterior1st_correct_values,
    exterior2d_correct_values,
    mszoning_correct_values,
    name_map,
    saletype_correct_values,
)

# Load configuration
config = get_config()


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths.

    Attributes:
        train_data_path (str): Path to save the training dataset.
        test_data_path (str): Path to save the testing dataset.
        clean_data_path (str): Path to save the cleaned combined dataset.
    """

    train_data_path: str = config.artifacts.train_data_path
    test_data_path: str = config.artifacts.test_data_path
    clean_data_path: str = config.artifacts.clean_data_path


class DataIngestion:
    """Class to handle the data ingestion and initial cleaning process."""

    def __init__(self):
        """Initialize DataIngestion with configuration."""
        self.ingestion_config = DataIngestionConfig()

    def ingest(self) -> tuple[str, str]:
        """Execute the data ingestion pipeline.

        Loads the raw dataset, performs basic cleaning, renames features,
        removes outliers and invalid examples, and splits into train/test sets.

        Returns:
            Tuple[str, str]: Paths to the saved train and test datasets.

        Raises:
            CustomException: If any error occurs during ingestion.
        """
        logger.info("Data ingestion started.")
        try:
            df = pd.read_csv(config.data.dataset_path, sep=config.data.separator)
            logger.info("Dataset read into a pandas Dataframe.")

            # Rename features
            df.rename(columns=name_map, inplace=True)
            logger.info("Renamed dataset features.")

            # Replace invalid values or fix inconsistent naming
            df.replace(
                {
                    "MSZoning": mszoning_correct_values,
                    "BldgType": bldgtype_correct_values,
                    "Exterior1st": exterior1st_correct_values,
                    "Exterior2nd": exterior2d_correct_values,
                    "SaleType": saletype_correct_values,
                },
                inplace=True,
            )
            logger.info("Validated dataset values via mapping.")

            # Remove examples that violate feature invariants
            # 1. Remodeling cannot come before the building was built
            df = df[df["YearBuilt"] <= df["YearRemod/Add"]]
            # 2. Year built must be before or equal to year sold
            df = df[df["YearBuilt"] <= df["YrSold"]]

            # 3. Total basement surface must be sum of basement portions
            df = df[
                np.isclose(
                    df["TotalBsmtSF"],
                    df[["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]].sum(axis=1),
                    atol=0.01,
                )
            ]

            # 4. Total living area must be sum of floor surfaces
            df = df[
                np.isclose(
                    df["GrLivArea"],
                    df[["1stFlrSF", "2ndFlrSF"]].sum(axis=1),
                    atol=0.01,
                )
            ]
            logger.info("Removed examples that violate feature invariants.")

            # Remove outliers
            df = df[df["GrLivArea"] < config.validation.outlier_threshold["GrLivArea"]]
            logger.info("Removed outliers based on GrLivArea.")

            # Keep only the subset of most relevant features plus target
            target = [config.data.target_feature]
            df = df[config.data.feature_subset + target]
            logger.info(
                f"Filtering dataset to {len(config.data.feature_subset)} relevant features."
            )

            # Save clean dataset
            os.makedirs(os.path.dirname(self.ingestion_config.clean_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.clean_data_path, index=False, header=True)
            logger.info(f"Clean dataset saved to {self.ingestion_config.clean_data_path}.")

            # Split the dataset into train and test sets
            df_train, df_test = train_test_split(
                df,
                test_size=config.data.test_size,
                shuffle=config.data.shuffle,
                random_state=config.general.random_state,
            )
            logger.info("Split dataset into train and test sets.")

            # Save train set
            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logger.info(f"Train split saved to {self.ingestion_config.train_data_path}.")

            # Save test set
            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info(f"Test split saved to {self.ingestion_config.test_data_path}.")
            logger.info("Data ingestion completed successfully.")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e) from e


if __name__ == "__main__":
    ingestor = DataIngestion()
    train_file, test_file = ingestor.ingest()
    print(f"Ingestion complete. Files saved: {train_file}, {test_file}")
