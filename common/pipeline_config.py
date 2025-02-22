from pydantic import BaseModel

from common.constants import Environment
from common.model import ModelConfig
from common.data import (
    DataLoaderConfig, DataConfig, DataLoaderType, SimpleDataLoaderConfig
    )
from common.trainer import TrainingStrategyConfig

class PipelineConfig(BaseModel):
    pipeline_name: str
    env: Environment
    model: ModelConfig
    data: DataConfig
    dataloader: DataLoaderConfig
    trainer: TrainingStrategyConfig
    device: str
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        if kwargs['dataloader']['name'] == DataLoaderType.SIMPLE:
            self.dataloader = SimpleDataLoaderConfig.model_validate(kwargs['dataloader'])
        else:
            raise ModuleNotFoundError(f'Dataloader strategy not found: %s' % kwargs['dataloader'])