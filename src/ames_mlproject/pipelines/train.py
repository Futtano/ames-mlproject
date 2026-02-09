"""
Training pipeline orchestration for the Ames ML project.
Coordinates data ingestion, preprocessing, and model training in a single workflow.
"""


from ames_mlproject.core.exceptions import CustomException
from ames_mlproject.core.logging import logger
from ames_mlproject.data.ingestion import DataIngestion
from ames_mlproject.data.preprocessing import DataPreprocessing
from ames_mlproject.models.training import ModelTraining


class TrainPipeline:
    """Orchestrates the end-to-end model training lifecycle."""

    def __init__(self):
        """Initialize pipeline with required component instances."""
        self.ingestion = DataIngestion()
        self.preprocessing = DataPreprocessing()
        self.model_trainer = ModelTraining()

    def run(self) -> None:
        """Execute the training pipeline steps sequentially.

        Steps:
            1. Ingest raw data and split into train/test sets.
            2. Preprocess data (imputation, encoding).
            3. Select and train the best model, then save it.

        Raises:
            CustomException: If any pipeline step fails.
        """
        try:
            logger.info("Training pipeline started.")

            # 1. Data Ingestion
            train_path, test_path = self.ingestion.ingest()
            logger.info(f"Data ingestion completed. Splits at: {train_path}, {test_path}")

            # 2. Data Preprocessing
            X_train, X_test, y_train, y_test, _ = self.preprocessing.preprocess(
                train_path, test_path
            )
            logger.info("Data preprocessing completed.")

            # 3. Model Training
            trained_model = self.model_trainer.train_best_model(X_train, y_train)
            logger.info("Model selection and tuning completed.")

            # 4. Model Evaluation
            self.model_trainer.evaluate_model(X_test, y_test, trained_model)
            logger.info("Model evaluation completed.")

            # 5. Model Serialization
            self.model_trainer.save_model(trained_model)
            logger.info("Model serialization completed.")

            logger.info("Training pipeline execution finished successfully.")

        except Exception as e:
            raise CustomException(e) from e


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
