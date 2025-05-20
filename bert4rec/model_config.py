import pandas as pd
from common.model import ModelConfig

class Bert4RecConfig(ModelConfig):
    history_feature_name: str = 'history_feature'
    latent_dim: int = 32
    padding_key: int = 0
    mask_key: int = 1
    cloze_masking_factor: float = 0.2
    beta: list = [0.9, 0.999]
    tl_heads: int = 4
    tl_layers: int = 2
    tl_dropout: float = 0.1
    bias_enable: bool = True
    model_dir: str = "artifacts"
    
    def transformation_step(self, data: dict):
        cardinality = self.features.get_feature_by_name(self.history_feature_name).cardinality
        data[self.history_feature_name] = data[self.history_feature_name]%cardinality
        return data
    
    def filter_step(self, data: dict):
        return data