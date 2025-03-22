from pipelines.training_pipeline import training_pipelines
from steps.config import ModelNameConfig
from steps.model_train import train_model
from zenml.client import Client



if __name__ == "__main__":
    model_config = ModelNameConfig(train_model_name="RandomForestRegressor")
    # run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipelines(train_data_path="data\\train.csv")
    
