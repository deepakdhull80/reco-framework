from enum import Enum
from typing import List, Optional

class FeatureKind(Enum):
    NUMERICAL: str = "numerical"
    CATEGORICAL: str = "categorical"
    DATETIME: str = "datetime"
    CATEGORICAL_LIST: str = "categorical_list"
    NUMERICAL_LIST: str = "numerical_list"
    
    def get_feature_kind(feature_type: str):
        """ Get the FeatureKind based on the string type """
        for kind in FeatureKind:
            if kind.value == feature_type:
                return kind
        return None

class FeatureConfig:
    # kind: FeatureKind = FeatureKind.CATEGORICAL
    # name: str = None
    def __init__(self, name: str, kind: FeatureKind = FeatureKind.CATEGORICAL) -> None:
        self.name = name
        self.kind = kind

class DataConfig:
    src_data_path: str = None
    rating_threshold: float = 0
    features_cfg: List[FeatureConfig] = []

class ModelConfig:
    model_name: str = None

class Config:
    exp_name: str = ""
    data_cfg: DataConfig = None
    model_cfg: ModelConfig = None
    
    def set_src_data_path(self, src_path: str):
        self.data_cfg.src_data_path = src_path
    
    def set_model_name(self, model_name: str):
        self.model_cfg.model_name = model_name