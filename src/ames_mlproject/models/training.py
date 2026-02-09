"""
Model training module for the Ames ML project.
Handles model selection, hyperparameter tuning, evaluation, and model serialization.
"""

import importlib
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from ames_mlproject.config import get_config
from ames_mlproject.core.exceptions import CustomException
from ames_mlproject.core.logging import logger
from ames_mlproject.data.ingestion import DataIngestion
from ames_mlproject.data.preprocessing import DataPreprocessing
from ames_mlproject.utils.helpers import parse_hyperparameters, save_object

# Load configuration
config = get_config()


@dataclass
class ModelsConfig:
    """Configuration for models and their hyperparameter search spaces."""

    def __post_init__(self):
        """Dynamically load models and parse parameters from configuration."""
        self.model_params: list[tuple[Any, dict[str, Any]]] = []
        for model_info in config.model_training.models:
            module_name = model_info["module"]
            class_name = model_info["class"]
            params = model_info["params"]

            try:
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                estimator = model_class()
                parsed_params = parse_hyperparameters(params)
                self.model_params.append((estimator, parsed_params))
                logger.info(f"Loaded model: {model_info['name']} ({class_name})")
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load model {class_name} from {module_name}: {e}")
                continue

    model_params: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)


@dataclass
class ModelTrainingConfig:
    """Configuration for model training artifacts.

    Attributes:
        trained_model_file_path (str): Path to save the trained model object.
        models_config (ModelsConfig): Configuration of models to evaluate.
    """

    trained_model_file_path: str = config.artifacts.model_path
    models_config: ModelsConfig = field(default_factory=ModelsConfig)


class ModelTraining:
    """Class to manage the model training lifecycle."""

    def __init__(self):
        """Initialize ModelTraining with configuration."""
        self.model_training_config = ModelTrainingConfig()

    def select_best_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[Any, dict[str, Any]]:
        """Perform model selection using nested cross-validation.

        Evaluates multiple algorithms and selects the one with the highest mean CV score.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target labels.

        Returns:
            Tuple[Any, dict]: The best estimator class and its parameter distribution.

        Raises:
            CustomException: If any error occurs during model selection.
        """
        logger.info("Model selection started.")
        try:
            nested_cv_scores = []
            for (
                estimator,
                param_distributions,
            ) in self.model_training_config.models_config.model_params:
                logger.info(f"Nested cross validation for model {estimator} started.")
                rs = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_distributions,
                    refit=True,
                    scoring=config.model_training.scoring,
                    n_iter=config.model_training.random_search_iter,
                    cv=config.model_training.nested_cv_inner_folds,
                    random_state=config.general.random_state,
                    n_jobs=config.model_training.n_jobs,
                )

                score = np.mean(
                    cross_val_score(
                        estimator=rs,
                        X=X_train,
                        y=y_train,
                        scoring=config.model_training.scoring,
                        cv=config.model_training.nested_cv_outer_folds,
                        n_jobs=config.model_training.n_jobs,
                    )
                )
                nested_cv_scores.append(score)
                logger.info(f"Nested CV score for {estimator}: {score:.3f}")

            best_model_idx = np.argmax(nested_cv_scores)
            best_model_config = self.model_training_config.models_config.model_params[
                best_model_idx
            ]
            logger.info(
                f"Selected best model: {best_model_config[0]} (score: {nested_cv_scores[best_model_idx]:.3f})"
            )

            return cast(tuple[Any, dict[str, Any]], best_model_config)

        except Exception as e:
            raise CustomException(e) from e

    def tune_model(
        self, X_train: np.ndarray, y_train: np.ndarray, model_config: tuple[Any, dict]
    ) -> Any:
        """Perform hyperparameter tuning on the selected model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target labels.
            model_config (Tuple[Any, dict]): Estimator and its parameter distribution.

        Returns:
            Any: The best estimator found after tuning.

        Raises:
            CustomException: If any error occurs during tuning.
        """
        logger.info("Hyperparameter tuning started.")
        model, param_distributions = model_config
        try:
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                refit=True,
                scoring=config.model_training.scoring,
                n_iter=config.model_training.random_search_iter,
                cv=config.model_training.random_search_cv,
                random_state=config.general.random_state,
                n_jobs=config.model_training.n_jobs,
            )
            rs.fit(X_train, y_train)
            logger.info(f"Tuning completed. Best params: {rs.best_params_}")
            logger.info(f"R2 score on training set: {rs.score(X_train, y_train):.3f}")
            return rs.best_estimator_

        except Exception as e:
            raise CustomException(e) from e

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, model: Any) -> float:
        """Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
            model (Any): The fitted model to evaluate.

        Returns:
            float: The evaluation score (R2).
        """
        logger.info(f"Evaluating model {model} on test set.")
        test_score = model.score(X_test, y_test)
        logger.info(f"Test R2 score: {test_score:.3f}")
        return float(test_score)

    def save_model(self, model: Any) -> None:
        """Serialize and save the model object to disk.

        Args:
            model (Any): The fitted model intended for saving.

        Raises:
            CustomException: If serialization fails.
        """
        logger.info("Saving model object.")
        path = self.model_training_config.trained_model_file_path
        try:
            save_object(path, model)
            logger.info(f"Model saved successfully at {path}")
        except Exception as e:
            raise CustomException(e) from e

    def train_best_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Run the full model training process: select and tune.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target labels.

        Returns:
            Any: The trained and tuned model object.
        """
        model_config = self.select_best_model(X_train, y_train)
        tuned_model = self.tune_model(X_train, y_train, model_config)
        return tuned_model


if __name__ == "__main__":
    ingestor = DataIngestion()
    preprocessor = DataPreprocessing()

    train_file, test_file = ingestor.ingest()
    X_tr, X_te, y_tr, y_te, _ = preprocessor.preprocess(train_file, test_file)

    trainer = ModelTraining()
    trainer.train_best_model(X_tr, y_tr)
