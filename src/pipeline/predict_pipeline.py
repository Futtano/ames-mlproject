import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.components.data_preprocessing import DataPreprocessingConfig
from src.components.model_training import ModelTrainingConfig
from src.utils import feat_subset
from src.utils import is_valid_num_feat, is_valid_cat_feat
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging

class FormData:
    def __init__(self, form_data):
        self.form_data = form_data
    
    def get_validated_data(self):
        logging.info(f'Validating form Data {self.form_data}.')
        validated_data = {}
        try:
            if len(self.form_data) != len(feat_subset):
                logging.error(f'Invalid data: {self.form_data} has les feature than expected.')
                raise Exception(f'Form data has less features than those expected by the model.')
            for key, value in self.form_data.items():
                if is_valid_num_feat(key, value):
                    validated_data[key] = np.float64(value)
                elif is_valid_cat_feat(key, value):
                    if key == 'MSSubClass' or key == 'OverallQual':
                        value = np.float64(value)
                    validated_data[key] = value
                else:  
                    logging.error(f'Invalid data: {key} is not a valid feature.')
                    raise KeyError(f'"{value}" is not in the allowed feature values')
        except Exception as e:
            raise CustomException(e, sys)
        
        return validated_data
    
    def as_DataFrame(self):
        logging.info(f'Converting {self.form_data} to a pandas DataFrame')
        try:
            df = pd.DataFrame([self.get_validated_data()])
        except Exception as e:
            raise CustomException(e, sys)
        
        return df


@dataclass
class PredictionPipelineConfig:
    preprocessor_file_path = DataPreprocessingConfig.preprocessor_obj_file_path
    model_file_path = ModelTrainingConfig.trained_model_file_path

class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()
        
    def predict(self, form_data):
        logging.info('Prediction pipeline started.')
        try:
            X = FormData(form_data).as_DataFrame()
            preprocessor = load_object(self.prediction_pipeline_config.preprocessor_file_path)
            model = load_object(self.prediction_pipeline_config.model_file_path)
            X = preprocessor.transform(X)
            return model.predict(X)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    form_data = {
        'OverallQual' : 7,
        'ExterQual': 'TA',
        'KitchenQual': 'Ex',
        'BsmtQual': 'Po',
        'MSSubClass': 20,
        'Neighborhood': 'SWISU',
        'YearBuilt': 1987,
        'YearRemod/Add': 1989,
        'GrLivArea': 1200,
        '1stFlrSF': 600,
        'TotalBsmtSF': 200,
        'BsmtFinSF1': 200,
        'GarageArea': 100,
        'MasVnrArea': 50,
        'FullBath': 2,
    }

    pred_pipe = PredictionPipeline()
    print(pred_pipe.predict(form_data))
        
