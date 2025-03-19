import logging
from zenml import step
from src.data_cleaning import DataDivideStrategy,DataCleaning,DataPreprocessStrategy
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated


@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:
    """_summary_

    Raises:
        err: _description_

    Returns:
        _type_: _description_
    """
    try:
        # preprocess the dataset
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data=data, strategy=preprocess_strategy)
        data, _ = data_cleaning.handle_data()
        
        # divide the dataset
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(data = data, strategy=divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(msg=f"Data Cleaning completed.")
        return X_train, X_test, y_train, y_test
    except Exception as err:
        logging.error(msg=f'Encounterd error {err} while cleaning the data')
        raise err
        
        