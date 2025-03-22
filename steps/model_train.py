import logging
import pandas as pd
from zenml import step
from src.model_dev import RfrModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

# init the experiment tracker object
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series,
                config : ModelNameConfig = ModelNameConfig()) -> RegressorMixin:
    model = None
    try:
        if config.train_model_name == "RandomForestRegressor":
            mlflow.sklearn.autolog()
            model = RfrModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise ValueError(f"Model not available in the list {config.train_model_name}")
    except Exception as err:
        logging.error(msg=f'Encountered error {err} while training')
        raise err
        
    
