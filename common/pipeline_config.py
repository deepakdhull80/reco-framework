from pydantic import BaseModel

from common.constants import Environment
from common.model import ModelConfig
from common.data import DataLoaderStrategy, DataConfig
from common.trainer import TrainingStrategy

class PipelineConfig(BaseModel):
    
    env: Environment
    model: ModelConfig
    data: DataConfig
    dataloader: DataLoaderStrategy
    trainer: TrainingStrategy