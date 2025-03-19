import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor


class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    
class RfrModel(Model):
    
    def train(self, X_train, y_train, **params):
        try:
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            logging.info(msg=f"Model Training complete")
            
            return model
        except Exception as err:
            logging.info(msg=f"Encountered error {err} while fitting the model")
            raise err