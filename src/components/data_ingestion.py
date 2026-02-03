import sys 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import name_map
from src.utils import mszoning_correct_values, bldgtype_correct_values
from src.utils import exterior1st_correct_values, exterior2d_correct_values
from src.utils import saletype_correct_values

dataset_path = '/mnt/c/Users/futha/source/python-projects/pyml-book/datasets/AmesHousing.txt'
random_state = 42

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    clean_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def ingest(self):
        logging.info('Data ingestion started.')
        try:
            df = pd.read_csv(dataset_path, sep='\t')
            logging.info('Dataset read into a pandas Dataframe.')

            # Rename features
            df.rename(columns=name_map, inplace=True)
            logging.info(f'Renamed dataset features.')

            # Replace invalid values
            df.replace({
                'MSZoning': mszoning_correct_values,
                'BldgType': bldgtype_correct_values,
                'Exterior1st': exterior1st_correct_values,
                'Exterior2nd': exterior2d_correct_values,
                'SaleType': saletype_correct_values
            }, inplace=True)
            logging.info(f'Validated dataset values.')

            # Remove examples that violate feature invariants
            df = df[df['YearBuilt'] <= df['YrSold']] # Remodeling cannot come before the building was built
            df = df[df['YearBuilt'] <= df['YearRemod/Add']] # The house cannot be sold before it is built            
            # The total basement surface must be equal to the sum of each portion of the basement
            df = df[np.isclose(
                    df['TotalBsmtSF'],
                    df[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']].sum(axis=1),
                    atol=0.01)]
            # The total living area surface must be equal to the sum of each floor surface
            df = df[np.isclose(
                    df['GrLivArea'],
                    df[['1stFlrSF', '2ndFlrSF']].sum(axis=1),
                    atol=0.01)]
            logging.info(f'Removed examples that violate features invariants.')

            # Remove outliers
            df = df[df['GrLivArea'] < 4000]
            logging.info(f'Removed outliers.')

            # Keep only the subset of most relevant features
            feat_subset = ['OverallQual', 'GrLivArea', 'BsmtQual', 'Neighborhood',
                        'KitchenQual', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF',
                        'GarageArea', 'FullBath', 'MasVnrArea', 'ExterQual',
                        'YearRemod/Add', 'MSSubClass', 'YearBuilt']
            target = ['SalePrice']
            df = df[feat_subset+target]
            logging.info(f'Keeping only a relevant subset of the features.')

            # Save clean dataset
            os.makedirs(os.path.dirname(self.ingestion_config.clean_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.clean_data_path, index=False, header=True)
            logging.info(f'Clean dataset saved into {self.ingestion_config.clean_data_path}.')
            
            # Split the dataset into train and test sets 
            df_train, df_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=random_state)
            logging.info('Dataset splitted into train and test sets.')

            # Save train set
            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f'Train split saved into {self.ingestion_config.train_data_path}.')

            # Save test set
            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f'Test split saved into {self.ingestion_config.test_data_path}.')
            logging.info('Data ingestion was completed.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    dingest = DataIngestion()
    train_path, test_path = dingest.ingest()