import pandas as pd
from common.model import ModelConfig

class TwoTowerConfig(ModelConfig):
    optimizer_clz: str = 'adam'
    lr: float = 1e-3
    beta: float = 0.99
    
    def transformation_step(self, data: pd.DataFrame):
        return data
    
    def filter_step(self, data: pd.DataFrame):
        return data