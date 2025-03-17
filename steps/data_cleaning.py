import logging
from zenml import step
import pandas as pd


@step
def clean_data(data: pd.DataFrame) -> None:
    pass