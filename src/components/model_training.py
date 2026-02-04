import sys
import os
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import elasticnet_param_distributions
from src.utils import dtree_param_distributions
from src.utils import xgb_param_distributions
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing

random_state = 42

@dataclass
class ModelsConfig:
    model_params = ((ElasticNet(), elasticnet_param_distributions),
                    (DecisionTreeRegressor(), dtree_param_distributions),
                    (XGBRegressor(), xgb_param_distributions))

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    models_config = ModelsConfig()


class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def select_best_model(self, X_train, y_train):
        logging.info(f'Model selection started for models:{self.model_training_config.models_config.model_params}.')
        try:
            nested_cv_scores = []
            for estimator, param_distributions in self.model_training_config.models_config.model_params:
                logging.info(f'Nested cross validation for model {estimator} started.')
                rs = RandomizedSearchCV(
                    estimator=estimator, 
                    param_distributions=param_distributions,
                    refit=True,
                    scoring='r2',
                    n_iter=20,
                    cv=2,
                    random_state=random_state,
                    n_jobs=-1)
                
                nested_cv_scores.append(np.mean(cross_val_score(
                        estimator=rs,
                        X=X_train,
                        y=y_train,
                        scoring='r2',
                        cv=5,
                        n_jobs=-1)))
                logging.info(f'Nested cross validation score for model {estimator} is {nested_cv_scores[-1]:.3f}.')
            best_model_idx = np.argmax(nested_cv_scores)
            best_model_config = self.model_training_config.models_config.model_params[best_model_idx]
            logging.info(f'Selected model {best_model_config[0]} as best model with score, {nested_cv_scores[best_model_idx]:.3f}.')
            
            return best_model_config
        
        except Exception as e:
            raise CustomException(e, sys)

    def tune_model(self, X_train, y_train, model_config):
        logging.info('Hyperparameter tuning started.')
        model, param_distributions = model_config
        try:
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                refit=True,
                scoring='r2',
                n_iter=20,
                cv=10,
                random_state=random_state,
                n_jobs=-1
                )
            rs.fit(X_train, y_train)
            logging.info(f'Hyperparameter tuning completed.')
            train_score = rs.score(X_train, y_train)
            logging.info(f'Best hyperparametrs found: {rs.best_params_}.')
            logging.info(f'R2 score on the training set is {train_score:.3f}.')
            return rs.best_estimator_
        
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, X_test, y_test, model):
        logging.info(f'Evaluating model {model} on the test set.')
        test_score = model.score(X_test, y_test)
        logging.info(f'Test score is {test_score:.3f}.')
        return test_score
    
    def save_model(self, model):
        logging.info(f'Saving serialized model.')
        path = self.model_training_config.trained_model_file_path
        try:
            save_object(path, model)
            logging.info(f'Model {model} saved in {path}.')
        except Exception as e:
            raise CustomException(e, sys)
        
    def save_best_model(self, X_train, y_train):
        model_config = self.select_best_model(X_train, y_train)
        tuned_model = self.tune_model(X_train, y_train, model_config)
        self.save_model(tuned_model)
        
if __name__ == '__main__':
    dingest = DataIngestion()
    dpreproc = DataPreprocessing()
    train_path, test_path = dingest.ingest()
    X_train, X_test, y_train, y_test, _ =\
        dpreproc.preprocess(train_path, test_path)
    mtrain = ModelTraining()
    mtrain.save_best_model(X_train, y_train)