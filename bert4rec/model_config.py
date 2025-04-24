import pandas as pd
from common.model import ModelConfig

class Bert4RecConfig(ModelConfig):
    history_feature_name: str = 'history_feature'
    latent_dim: int = 32
    padding_key: int = 0
    mask_key: int = 1
    cloze_masking_factor: float = 0.2
    
    def transformation_step(self, data: pd.DataFrame):
        return data
    
    def filter_step(self, data: pd.DataFrame):
        return data