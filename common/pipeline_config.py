from pydantic import BaseModel

from common.constants import Environment
from common.model import ModelConfig
from common.data import (
    DataLoaderStrategy, DataConfig, DataLoaderType, SimpleDataLoaderStrategy
    )
from common.trainer import TrainingStrategy

class PipelineConfig(BaseModel):
    pipeline_name: str
    env: Environment
    model: ModelConfig
    data: DataConfig
    dataloader: DataLoaderStrategy
    trainer: TrainingStrategy
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        if kwargs['dataloader']['name'] == DataLoaderType.SIMPLE:
            self.dataloader = SimpleDataLoaderStrategy.parse_obj(kwargs['dataloader'])
        else:
            raise ModuleNotFoundError(f'Dataloader strategy not found: %s' % kwargs['dataloader'])