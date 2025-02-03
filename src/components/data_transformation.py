import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import CustomException
import os
from src.utils import save_object



@dataclass
# Define a configuration class to store the file path for saving the preprocessing object.
class DataTransformationConfig:
#preprocessor_obj_file_path: Path to save the serialized preprocessing object (e.g., artifacts/preprocessor.pkl).
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


# Initialize the DataTransformation class.
class DataTransformation:
    def __init__(self):
        #Creates an instance of DataTransformationConfig to store the preprocessing object file path.
        self.data_transformation_config = DataTransformationConfig()

   
    def get_data_transformer_object(self):
        
        try:
            # Define the numerical columns to be preprocessed
            numerical_columns = ['writing_score', 'reading_score']
            # Define the categorical columns to be preprocessed
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            #create pipeline for numerical columns
            num_pipeline = Pipeline(steps=[
                # Step 1: Impute missing values with the median
                ('imputer', SimpleImputer(strategy='median')),
                # Step 2: Standardize the data having mean=0 and variance=1
                ('std_scaler', StandardScaler())
            ])
            #create pipeline for categorical columns
            cat_pipeline = Pipeline(steps=[
                # Step 1: Impute missing values with the most frequent value
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # Step 2: One hot encode the categorical columns
                ('onehot', OneHotEncoder()),
                # Step 3: Standardize the data having mean=0 and variance=1
                ('scaler',StandardScaler(with_mean=False))
            ])

            # Log the categorical and numerical columns for debugging
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #combine the numerical and categorical pipelines using ColumnTransformer
            preprocesser = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns), #apply num_pipeline on numerical columns
                    ('cat', cat_pipeline, categorical_columns) #apply cat_pipeline on categorical columns
                ])
            # Return the combined preprocessing object as array
            return preprocesser

        
        # Handle exceptions during the creation of the preprocessing object and raise a custom exception.
        except Exception as e:
            logging.info('Error in data transformation process')
            raise CustomException(e,sys)
        

    #  initiate_data_transformation Method    
    def initiate_data_transformation(self,train_path,test_path):
      
        try:
            # Read the train and test data and log the completion
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Log the start of obtaining the preprocessing object and call the get_data_transformer_object method to create it.
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # define the target column
            target_column_name = 'math_score'
            #define the numerical column for referance
            numerical_columns = ['writing_score','reading_score']

            #seperate input feature and target feature for the training dataset
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

             # Separate input features and target feature for the testing dataset
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            
            #Apply the preprocessing pipeline to the training data (fit and transform)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            #Apply the preprocessing pipeline to the testing data (transform only)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #Combine the preprocessed input features and target variable into NumPy arrays for both training and testing datasets.
            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            # Save the preprocessing object as a pickle file for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
              # Return the processed training and testing arrays, along with the file path of the saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            # Log and raise any exceptions that occur during data transformation
            raise CustomException(e, sys)
        
