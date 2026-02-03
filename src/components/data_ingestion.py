import sys 
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

dataset_path = '/mnt/c/Users/futha/source/python-projects/pyml-book/datasets/AmesHousing.txt'
random_state = 42

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def ingest(self):
        logging.info('Data ingestion started.')
        try:
            df = pd.read_csv(dataset_path, sep='\t')
            logging.info('Dataset read into a pandas Dataframe.')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Raw dataset saved into {self.ingestion_config.raw_data_path}.')
            df_train, df_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=random_state)
            logging.info('Dataset splitted into train and test sets.')
            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f'Train split saved into {self.ingestion_config.train_data_path}.')
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
    obj = DataIngestion()
    obj.ingest()