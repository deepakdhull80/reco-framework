import pandas as pd
from common.model import ModelConfig

class TwoTowerConfig(ModelConfig):
    
        
    def transformation_step(self, data: pd.DataFrame):
        return data
    
    def filter_step(self, data: pd.DataFrame):
        return data