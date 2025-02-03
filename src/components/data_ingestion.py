import os,sys
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw_data.csv")    
    # Note: The @dataclass decorator simplifies the creation of this class by automatically generating __init__ and other methods.
    # Define a configuration class to store file paths for:
    # train_data_path: Path to save the training data.
    # test_data_path: Path to save the testing data.
    # raw_data_path: Path to save the raw data.

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Initialize the DataIngestion class.
    # self.ingestion_config: Creates an instance of DataIngestionConfig to store file paths.


    #  Start the data ingestion process and log the start.
    def initiate_data_ingestion(self): 
        logging.info('Data Ingestion Process Started')

        

        try:
            # Read the raw data from a CSV file (notebook/data/data.csv) and log a success message.
            df = pd.read_csv("notebook/data/data.csv")
            logging.info('Raw data read successfully')

            # Create the directory structure for saving the data files if it doesn’t already exist.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the path specified in DataIngestionConfig and log a success message.
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            logging.info('Raw data saved successfully')

          
            #  Split the raw data into training and testing sets (80% train, 20% test) and log the start of the process.
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training and testing datasets to their respective paths and log the completion of the process.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logging.info('Train test split completed')

            # : Return the paths of the saved training and testing datasets for further use.
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        # Handle any exceptions that occur during the data ingestion process, log the error, and raise a custom exception
        except Exception as e:
            logging.info('Error in data ingestion process')
            raise CustomException(e,sys)
        



   
if __name__ == '__main__':

    # Create an instance of the DataIngestion class.
    # Call the initiate_data_ingestion method to perform data ingestion and get the paths of the training and testing datasets.
    obj = DataIngestion()
    train_arr,test_arr = obj.initiate_data_ingestion()

    # Initialize the DataTransformation class and call its initiate_data_transformation method to process the training and testing datasets.
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_arr,test_arr)

    modeltrainer =  ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    

