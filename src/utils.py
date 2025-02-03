import os 
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

# Create the directory structure for saving the object if it doesn’t already exist.

def save_object(file_path,obj):
    try:
        
        # os.path.dirname(file_path): Extracts the directory path from the file path.
        dir_path = os.path.dirname(file_path)
        # os.makedirs(dir_path, exist_ok=True): Creates the directory if it doesn’t exist
        os.makedirs(dir_path,exist_ok=True)

        #  Opens the file in binary write mode.
        with open(file_path,'wb') as file_obj:
            #  Serializes the object and writes it to the file.
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

#  Evaluate_models Function
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        # Initialize an empty dictionary to store the evaluation results for each model.
        report = {}

        # Iterate over the dictionary of models , Converts the dictionary values (model objects) into a list.
        for i in range(len(list(models))):
            # Retrieves the current model object.
            model = list(models.values())[i]

            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            ## Retrieves the current model object.
            # model.fit(X_train,y_train)

            #  Generate predictions for both the training and testing datasets.
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate the R² score for both the training and testing predictions.
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Purpose: Store the testing R² score in the report dictionary.
            # list(models.keys())[i]: Retrieves the name of the current model.
            # test_model_score: The R² score for the testing data.
            report[list(models.keys())[i]] = test_model_score

        # Purpose: Return the dictionary containing the evaluation results for all models.
        return report

    except Exception as e:
        raise CustomException(e, sys)
