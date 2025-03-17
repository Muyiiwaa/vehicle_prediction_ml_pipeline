import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Union
import time


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
    def handle_data(self, data:pd.DataFrame) -> pd.DataFrame:
        """_method for preprocessing the data_

        Args:
            data (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        try:
            # removed commas and changed the type of distance from str to float
            logging.info(msg=f'Removing commas and converting distance to float')
            data['Distance'] = data['Distance'].str.replace(pat=',',
                                                            repl='').astype(float)
            # handle missing values in distance by filling with median
            logging.info(msg=f'Filling the missing values in distance column with the median')
            data['Distance'] = data['Distance'].fillna(value=data['Distance'].median())
            
            # drop the missing values in the remaining dataset
            data.dropna(inplace= True)
            data.reset_index(drop=True, inplace= True)
            logging.info(msg="Handled missing values completely")
            
            # create age column from  year column
            data['Year'] = data['Year'].str.replace(pat=',',repl='').astype(int)
            data['age'] = time.localtime() - data['Year']
            
            # drop columns that are not needed
            data.drop(columns=['Year', 'VehicleID'],inplace=True)
            
            # implement label encoding.
            
            return data
        except Exception as err:
            logging.error(msg=f'Encountered error while preprocessing: {err}')
            raise err
            
            
            
            