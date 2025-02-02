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
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    """
    define the path to save the preprocessor object
    """

#define data transformation class
class DataTransformation:
    def __init__(self):
        #initialize the configuration object
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        this function is responsible for creating data transformation pipeline
        It defines preprocessing steps for numerical and categorical columns
        """
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
            return preprocesser
        
        except Exception as e:
            logging.info('Error in data transformation process')
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        """
        This function applies the preprocessing pipeline to the training and testing datasets.
        It also saves the preprocessing object for future use.
        """
        try:
            # Read the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")


            # Get the preprocessing object
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

            # Log that preprocessing is being applied to the datasets
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
             # Apply the preprocessing pipeline to the training data (fit and transform)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # Apply the preprocessing pipeline to the testing data (transform only)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #Combine the processed input features with the target feature for training data
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            # Combine the processed input features with the target feature for testing data
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
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Log and raise any exceptions that occur during data transformation
            raise CustomException(e, sys)