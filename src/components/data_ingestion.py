import os
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw_data.csv")    
    """
    This class is used to store the configuration for data ingestion.
    """

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # DataIngestionConfig object to store the paths of the data files

    def initiate_data_ingestion(self):
        """
        This function is used to initiate the data ingestion process.
        """
        logging.info('Data Ingestion Process Started')

        try:
            # Read the raw data
            df = pd.read_csv("notebook/data/data.csv")
            logging.info('Raw data read successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            logging.info('Raw data saved successfully')

            logging.info('Train test split started')
            # Split the data into train and test
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info('Train test split completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Error in data ingestion process')
            raise CustomException(e,sys)
    
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    

    
