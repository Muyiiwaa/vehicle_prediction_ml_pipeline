from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.data_cleaning import clean_data
from steps.evaluation import evaluate_model
from steps.model_train import train_model


@pipeline
def training_pipelines(train_data_path: str) -> None:
    train_df = ingest_df(train_data_path)
    clean_data(train_df)
    train_model(train_df)
    evaluate_model(train_df)
    