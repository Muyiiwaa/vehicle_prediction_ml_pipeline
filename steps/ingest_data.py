import logging
import pandas as pd
from zenml import step
from typing import Tuple



# create IngestData class

class IngestData:
    """_This class handles the loading of dataset into the pipeline. It takes 
    both the training and test data which is currently a csv file and returns a
    pandas DataFrame to be used in the pipeline._
    
    Attributes:
        train_data_path (str): The path to the train data
        test_data_path (str): The path to the test data
    
    Methods:
        get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
            Takes both training and test set and returns a tuple of pandas 
            dataframe.
    """
    def __init__(self, train_path) -> None:
        """Initializes the IngestData arguments

        Args:
            train_path (_str_): _Path to the train data._
            test_path (_str_): _Path to the test data._
        """
        self.train = train_path
        
    def get_data(self) -> pd.DataFrame:
        """_The method for returning the pandas dataframe needed
        for this pipeline from the provided csv files._

        Returns:
            Tuple[pd.DataFrame]: _A tuple of train and test pandas dataframe._
        """
        logging.info(msg=f"Now ingesting the train_data from {self.train}")
        train_data = pd.read_csv(self.train)
        
        return train_data
    
# create the ingest_data step in the pipeline

@step
def ingest_df(train_data_path: str) -> pd.DataFrame:
    """_Function for ingesting the train and test data within the pipeline._

    Args:
        train_data_path (str): _path to the training data._
        test_data_path (str): _path to the test data._

    Raises:
        err: _Error from loading the dataset or reading the file path._

    Returns:
        Tuple[pd.DataFrame]: _train and test data dataframe object._
    """
    try:
        data_object = IngestData(train_path=train_data_path)
        final_data = data_object.get_data()
        
        return final_data
    except Exception as err:
        logging.error(msg=f'Encounterd error {err} while loading the dataset')
        raise err
        