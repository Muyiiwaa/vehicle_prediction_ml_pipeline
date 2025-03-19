from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.data_cleaning import clean_data
from steps.evaluation import evaluate_model
from steps.model_train import train_model


@pipeline(enable_cache=False)
def training_pipelines(train_data_path: str) -> None:
    train_df = ingest_df(train_data_path)
    X_train, X_test, y_train, y_test = clean_data(train_df)
    model = train_model(X_train, X_test, y_train, y_test)
    rmse, mae = evaluate_model(model, X_test, y_test)
    