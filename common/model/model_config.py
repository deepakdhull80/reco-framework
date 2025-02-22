import pandas as pd
from pydantic import BaseModel
from typing import List
from common.constants import ModelType
from common.data.feature_config import FeatureConfig, Feature, TowerType

class ModelConfig(BaseModel):
    name: str
    version: float = 0.0
    type: ModelType
    query_id_name: str
    item_id_name: str
    target_id_name: str
    features: FeatureConfig
    optimizer_clz: str = 'Adam'
    sparse_optimizer_clz: str = 'Adam'
    sparse_lr: float = 1e-3
    lr: float = 1e-3
    beta: float = 0.99
    
    def transformation_step(self, data: pd.DataFrame):
        return data
    
    def filter_step(self, data: pd.DataFrame):
        return data
    
    def get_item_features(self) -> List[Feature]:
        item_features_li = []
        for feature in self.features.all_features():
            if feature.tower == TowerType.ITEM:
                item_features_li.append(feature)
        return item_features_li
    
    def get_query_features(self) -> List[Feature]:
        query_features_li = []
        for feature in self.features.all_features():
            if feature.tower == TowerType.QUERY:
                query_features_li.append(feature)
        return query_features_li
    
    def preprocessing_fn(self, data: pd.DataFrame):
        data = self.transformation_step(data)
        data = self.filter_step(data)
        return data