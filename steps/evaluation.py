import logging
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
from src.evaluation import RmseLoss, MaeLoss
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
                       Annotated[float, "rmse score"],
                       Annotated[float, "mae score"]]:
    try:
        prediction = model.predict(X_test)
        rmse_class = RmseLoss()
        rmse_score = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse_score)
        
        mae_class = MaeLoss()
        mae_score = mae_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mae", mae_score)
        return rmse_score, mae_score
    except Exception as err:
        logging.error(f"Encountered error {err} while evaluating")
        
        