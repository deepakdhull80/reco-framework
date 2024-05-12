from typing import List
from pydantic import BaseModel
from enum import Enum

class FeatureType(Enum):
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    TIMESTAMP = 'timestamp'
    CATEGORICAL_LIST = 'categorical_list'
    NUMERICAL_LIST = 'numerical_list'
    TIMESTAMP_LIST = 'timestamp_list'

class TowerType(Enum):
    QUERY = 'query'
    ITEM = 'item'

class DataType(Enum):
    STRING: 'str'
    INT: 'int'
    FLOAT: 'float'
    LONG: 'long'
    DATETIME: 'datetime'

class Feature(BaseModel):
    name: str
    type: FeatureType
    tower: TowerType
    dtype: DataType

class NumericalFeature(Feature):
    type = FeatureType.NUMERICAL

class FeatureConfig(BaseModel):
    numerical_features: List[NumericalFeature]