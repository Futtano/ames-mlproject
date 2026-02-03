import sys
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from src.utils import num, cat_ord, cat_nom, enc_ord_categories
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion

random_state = 42
target_feature = 'SalePrice'


@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config = DataPreprocessingConfig()

    def get_preprocessor(self):

        cat_nom_pipeline = Pipeline([
            ('impute_most_freq', SimpleImputer(strategy='most_frequent')),
            ('encode_nom', TargetEncoder(cv=5, smooth='auto',shuffle=True, random_state=random_state, target_type='continuous'))
        ])
        
        cat_ord_pipeline = Pipeline([
            ('impute_most_freq', SimpleImputer(strategy='most_frequent')),
            ('encode_ord', OrdinalEncoder(categories=enc_ord_categories))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', SimpleImputer(strategy='median'), num),
            ('cat_nominal', cat_nom_pipeline, cat_nom),
            ('cat_ordinal', cat_ord_pipeline, cat_ord),
        ])

        return preprocessor
    
    def preprocess(self, train_path, test_path):
        logging.info('Data Preprocessing started.')
        try:
            df_train = pd.read_csv(train_path)
            logging.info(f'Train set read into a pandas Dataframe from {train_path}')
            
            df_test = pd.read_csv(test_path)
            logging.info(f'Test set read into a pandas Dataframe from {test_path}')

            X_train = df_train.drop(columns=target_feature)
            y_train = df_train[target_feature].to_numpy()
            X_test = df_test.drop(columns=target_feature)
            y_test = df_test[target_feature].to_numpy()

            logging.info('Starting preprocessing pipeline.')
            preprocessor = self.get_preprocessor()
            X_train = preprocessor.fit_transform(X_train, y_train)
            X_test = preprocessor.transform(X_test)
            logging.info('Completed preprocessing pipeline on train and test set.')

            save_object(self.data_preprocessing_config.preprocessor_obj_file_path, preprocessor)
            logging.info(f'Saved preprocessor object into pickle file {self.data_preprocessing_config.preprocessor_obj_file_path}.')
            
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_preprocessing_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    dingest = DataIngestion()
    dpreproc = DataPreprocessing()
    train_path, test_path = dingest.ingest()
    X_train, X_test, y_train, y_test, _ =\
        dpreproc.preprocess(train_path, test_path)
    print(f'X_train first row: {X_train[0]}')
    print(f'X_test first row: {X_test[0]}')
    print(f'y_train first value: {y_train[0]}')
    print(f'y_test first value: {y_test[0]}')