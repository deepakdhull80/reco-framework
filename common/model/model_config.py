from pydantic import BaseModel
from common.constants import ModelType


class ModelConfig(BaseModel):
    name: str
    version: float = 0.0
    type: ModelType
    