from pydantic import BaseModel

from common.constants import TrainingStrategyNames

class TrainingStrategyConfig(BaseModel):
    strategy: TrainingStrategyNames


class TrainingStrategy:
    def __init__(self, model_builder, pipeline_config) -> None:
        self.model_builder = model_builder
        self.pipeline_config = pipeline_config
    
    def train(self):
        raise NotImplementedError()
    
    def val(self):
        raise NotImplementedError()
    