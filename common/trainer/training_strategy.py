from pydantic import BaseModel

from common.constants import TrainingStrategyNames
from common.model.model_builder import ModelBuilder
from common.data.dataloader import DataLoaderStrategy
from common.model.model_config import ModelConfig
class TrainingStrategyConfig(BaseModel):
    strategy: TrainingStrategyNames
    epochs: int


class TrainingStrategy:
    def __init__(self, 
                 model_builder: ModelBuilder, 
                 dataloader_strategy: DataLoaderStrategy, 
                 trainer_config: TrainingStrategyConfig,
                 model_config: ModelConfig
                 ) -> None:
        self.model_builder = model_builder
        self.trainer_config = trainer_config
        self.dataloader_strategy = dataloader_strategy
        self.model_config = model_config
    
    def fit(self):
        raise NotImplementedError()
    
    def train(self):
        raise NotImplementedError()
    
    def val(self):
        raise NotImplementedError()
    