import pandas as pd
from pydantic import BaseModel
from common.constants import ModelType

class ModelConfig(BaseModel):
    name: str
    version: float = 0.0
    type: ModelType
    
    
    def transformation_step(self, data: pd.DataFrame):
        return data
    
    def filter_step(self, data: pd.DataFrame):
        return data
    
    def preprocessing_fn(self, data: pd.DataFrame):
        data = self.transformation_step(data)
        data = self.filter_step(data)
        return data