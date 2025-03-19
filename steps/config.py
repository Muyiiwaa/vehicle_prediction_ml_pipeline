from pydantic import BaseModel

# Define step parameters by subclassing BaseModel
class ModelNameConfig(BaseModel):
    train_model_name: str = "RandomForestRegressor"