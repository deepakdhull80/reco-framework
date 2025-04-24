import torch
import pandas as pd
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from enum import Enum

class ImputeNa(Enum):
    Numerical = -1
    Categorical = -1
    String = ''

class FeatureType(Enum):
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    TIMESTAMP = 'timestamp'
    CATEGORICAL_LIST = 'categorical_list'
    NUMERICAL_LIST = 'numerical_list'
    TIMESTAMP_LIST = 'timestamp_list'
    COMMON = 'common'
    VECTOR = 'vector'

class TowerType(Enum):
    QUERY = 'query'
    ITEM = 'item'
    NEITHER = 'neither'

class DataType(str, Enum):
    STRING = 'str'
    INT = 'int32'
    FLOAT = 'float32'
    LONG = 'long'
    DATETIME = 'datetime'
    BOOLEAN = 'bool'

class Feature(BaseModel):
    name: str
    f_type: FeatureType
    tower: TowerType
    f_dtype: DataType
    dim: int = 32
    is_datetime: bool = False

class NumericalFeature(Feature):
    f_type: FeatureType = FeatureType.NUMERICAL

class CategoricalFeature(Feature):
    f_type: FeatureType = FeatureType.CATEGORICAL
    cardinality: int = 10
    sparse: bool = False

class VectorFeature(Feature):
    f_type: FeatureType = FeatureType.VECTOR
    dim: int = 32

class CommonFeature(Feature):
    f_type: FeatureType = FeatureType.COMMON
    tower: TowerType = TowerType.NEITHER

class HistoryFeature(Feature):
    f_type: FeatureType = FeatureType.CATEGORICAL_LIST
    f_dtype: DataType = DataType.LONG
    cardinality: int = 10
    padding_key: int = 0
    pad_at_end: bool = True
    max_length: int = 100

class FeatureConfig(BaseModel):
    numerical_features: Optional[List[NumericalFeature]] = []
    categorical_features: Optional[List[CategoricalFeature]] = []
    common_features: Optional[List[CommonFeature]] = []
    vector_features: Optional[List[VectorFeature]] = []
    history_features: Optional[List[HistoryFeature]] = []
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def all_features(self) -> List[Feature]:
        features_li = []
        for features in [self.numerical_features, self.categorical_features, self.common_features, self.vector_features, self.history_features]:
            if features is None:
                continue
            features_li.extend(features)
        return features_li
    
    def get_feature_by_name(self, name: str) -> Feature:
        for feature in self.all_features():
            if feature.name == name:
                return feature
        raise ValueError(f'feature: {name} not found in features')
    
    def boolean_features(self) -> List[Feature]:
        return list(filter(lambda x: x.f_dtype == DataType.BOOLEAN , self.all_features()))
    
    def convert_to_platform_type(self, batch: pd.DataFrame):
        _batch = {}
        for feature in self.all_features():
            if feature.name not in batch:
                raise ValueError(f'feature: {feature.name} is not present in batch')
            
            value: np.array = batch[feature.name].values
            
            if feature.f_type == FeatureType.CATEGORICAL:
                value = value.astype(feature.f_dtype)
                # handling value exceeding the cardinality
                value = value % feature.cardinality
            if feature.f_type == FeatureType.NUMERICAL:
                value = value.astype(feature.f_dtype)
            if feature.f_type == FeatureType.VECTOR:
                # TODO: f_dtype need to validate once.
                value = np.stack([v.astype(feature.f_dtype) for v in value], axis=0)
            if feature.f_type == FeatureType.COMMON:
                value = value.astype(feature.f_dtype)
            
            if feature.f_type == FeatureType.CATEGORICAL_LIST:
                # Handle categorical list, where length should be max_length
                # and padding_key should be used for padding
                # and pad_at_end should used to determine padding position
                dtype = feature.f_dtype
                max_length = feature.max_length
                pad_at_end = feature.pad_at_end
                padding_key = feature.padding_key
                arr = []
                for row in value:
                    if isinstance(row, np.ndarray):
                        row = row.tolist()
                    delta = max(0, max_length - len(row))
                    if pad_at_end:
                        row.extend([padding_key] * delta)
                    else:
                        row = [padding_key] * delta + row
                    row = np.array(row)
                    row = row[:max_length]
                    arr.append(row)
                value = np.stack(arr, axis=0)
                value = value.astype(dtype)
                
            _batch[feature.name] = torch.from_numpy(value)
        
        
        # convert boolean to integer
        for feature in self.boolean_features():
            if feature.name not in _batch:
                continue
            _batch[feature.name] = _batch[feature.name].int()
        
        return _batch