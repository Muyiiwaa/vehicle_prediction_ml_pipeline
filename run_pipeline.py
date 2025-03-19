from pipelines.training_pipeline import training_pipelines
from steps.config import ModelNameConfig
from steps.model_train import train_model



if __name__ == "__main__":
    model_config = ModelNameConfig(train_model_name="RandomForestRegressor")
    # run the pipeline
    training_pipelines(train_data_path="data\\train.csv")