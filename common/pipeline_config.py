from pydantic import BaseModel

from common.constants import Environment
from common.model import ModelConfig
from common.data import DataLoaderStrategy, DataConfig


class PipelineConfig(BaseModel):
    
    env: Environment
    model: ModelConfig
    data: DataConfig
    dataloader: DataLoaderStrategy