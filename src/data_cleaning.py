import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Union


class DataHandlingStrategy(ABC):
    """_Abstract class for defining the data cleaning strategy. This class serves
    as the blueprint for defining handling data ingestion._"""
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """_summary_

        Args:
            data (pd.DataFrame): _description_

        Returns:
            Union[pd.DataFrame, pd.Series]: _description_
        """
        pass
    
class DataPreprocessStrategy(DataHandlingStrategy):
    """_Abstract class for preprocessing the data._

    """
    
    @abstractmethod
    def handle_data(self, data) -> pd.DataFrame:
        """_method for preprocessing the data_

        Args:
            data (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """