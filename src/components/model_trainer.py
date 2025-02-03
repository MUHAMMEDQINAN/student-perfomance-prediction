import os, sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_models


# Define a configuration class to store the file path for saving the trained model.train_model_file_path: Path to save the trained model (e.g., artifacts/model.pkl).

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')

# Initialize the ModelTrainer class.
# self.model_trainer_config: Creates an instance of ModelTrainerConfig to store the model file path.

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    # Split the training and testing arrays into input features (X_train, X_test) and target variables (y_train, y_test).
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            # Define a dictionary of regression models to evaluate.
            models= {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
              
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            # Evaluate the performance of all models using the evaluate_models utility function.
            # Returns a dictionary (model_report) where keys are model names and values are their performance metrics (e.g., R² score). 
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param = params)
            
            #  highest performance score (e.g., R² score).
            best_model_score = max(sorted(model_report.values()))

            # The name of the best-performing model.
            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            #Logging  the best  model
            logging.info(f"Best model is :{best_model_name}")

            # The actual model object with the best performance.
            best_model = models[best_model_name]

            #  Check if the best model's performance is acceptable (R² score ≥ 0.6).
            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info("Best model found on both training and testing dataset")

            # Save the best-performing model to a file using the save_object utility function.The file path is specified in ModelTrainerConfig
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            # Make predictions on the test data using the best model and calculate the R² score.Return the R² score as the final result.
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return  r2_square

        except Exception as e:
            raise CustomException(e,sys)
            
