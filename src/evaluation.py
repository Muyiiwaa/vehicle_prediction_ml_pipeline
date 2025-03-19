from abc import ABC, abstractmethod
import numpy as np
import logging
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


class Evaluation(ABC):
    """_Abstract class for defining evaluation._

    """
    
    
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray,
                         y_pred: np.ndarray):
        pass
    
class RmseLoss(Evaluation):
    
    def calculate_scores(self, y_true:np.ndarray,
                         y_pred:np.ndarray) -> float:
        """_Method for calculating RMSE loss._

        Args:
            y_true (np.ndarray): _Real target value._
            y_pred (np.ndarray): _Predicted target value._

        Returns:
            _float_: _root mean squared error._
        """
        try:
            logging.info(msg='Calculating RMSE')
            score = root_mean_squared_error(y_true, y_pred)
            return score
        except Exception as err:
            logging.error(msg=f'Encountered error {err} while evaluating RMSE')

class MaeLoss(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, 
                         y_pred: np.ndarray) -> float:
        """_method calculates mean absolute error._

        Args:
            y_true (np.ndarray): _real target column._
            y_pred (np.ndarray): _predicted target column._

        Returns:
            float: _The mean absolute error value._
        """
        try:
            logging.info(msg=f"Evaluating the mse loss")
            score = mean_absolute_error(y_true,y_pred)
            return score
        except Exception as err:
            logging.error(msg=f'Encountered error {err} while evaluating MAE')