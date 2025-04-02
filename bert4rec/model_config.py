import pandas as pd
from common.model import ModelConfig

class Bert4RecConfig(ModelConfig):
    optimizer_clz: str = 'Adam'
    sparse_optimizer_clz: str = 'Adam'
    sparse_lr: float = 1e-3
    lr: float = 1e-3
    beta: float = 0.99
    
    def transformation_step(self, data: pd.DataFrame):
        return data
    
    def filter_step(self, data: pd.DataFrame):
        return data