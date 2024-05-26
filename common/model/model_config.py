import pandas as pd
from pydantic import BaseModel
from common.constants import ModelType
from common.data.feature_config import FeatureConfig

class ModelConfig(BaseModel):
    name: str
    version: float = 0.0
    type: ModelType
    features: FeatureConfig
    optimizer_clz: str = 'adam'
    lr: float = 1e-3
    beta: float = 0.99
    
    def transformation_step(self, data: pd.DataFrame):
        return data
    
    def filter_step(self, data: pd.DataFrame):
        return data
    
    def preprocessing_fn(self, data: pd.DataFrame):
        data = self.transformation_step(data)
        data = self.filter_step(data)
        return data