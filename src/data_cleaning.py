import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Union,Dict,Tuple
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
    

    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
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
            data['age'] = time.localtime().tm_year - data['Year']
            
            # drop columns that are not needed
            data.drop(columns=['Year', 'VehicleID'],inplace=True)
            
            # rename the amount column
            data.rename(columns= {'Amount (Million Naira)':'amount'},
                        inplace= True)
            
            # implement label encoding.
            categorical_columns = list(data.select_dtypes(include='object'))
            label_encoders: Dict = {}
            logging.info(msg=f'Label encoding {len(categorical_columns)} columns')
            for column in categorical_columns:
                encoder = LabelEncoder() # init the encoder for that column instance
                data[column] = encoder.fit_transform(data[column])
                label_encoders.update({column:encoder}) # update the encoder dictionary
                
            return data, label_encoders
        except Exception as err:
            logging.error(msg=f'Encountered error while preprocessing: {err}')
            raise err
        
class DataDivideStrategy(DataHandlingStrategy):
    """_Strategy for spiliting the dataset into train and test sets
    for the features and targets._

    Args:
        DataHandlingStrategy (_type_): _description_
    """
    def handle_data(self, data:pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """_Method for splitting the data into train and test set._

        Args:
            data (pd.DataFrame): _description_

        Returns:
            Union[pd.Series, pd.DataFrame]: _description_
        """
        try:
            # rename the amount column
            data.rename(columns= {'Amount (Million Naira)':'amount'},
                        inplace= True)
            X: pd.DataFrame = data.drop(columns=['amount'])
            y: pd.Series = data['amount']
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                random_state=23,
                                                                test_size=0.25)
            return X_train, X_test, y_train, y_test
        except Exception as err:
            logging.error(msg=f'Encountered Error {err} while spliting')
            raise err
        
class DataCleaning:
    
    
    def __init__(self, data:pd.DataFrame, 
                 strategy:DataHandlingStrategy) -> None:
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """_summary_

        Returns:
            Union[pd.DataFrame, pd.Series, Dict]: _description_
        """
        try:
            return self.strategy.handle_data(self.data)
        
        except Exception as err:
            logging.error(msg=f'Encountered error {err} while cleaning')
            raise err
        
if __name__ == '__main__':
    data = pd.read_csv("data\\train.csv")
    data_cleaning = DataCleaning(data=data, strategy=DataDivideStrategy())
    data_cleaning.handle_data()
        
    
            
            
            
            