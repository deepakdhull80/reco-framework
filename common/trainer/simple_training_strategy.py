from common.trainer.training_strategy import TrainingStrategy


class SimpleTrainingStrategy(TrainingStrategy):
    def __init__(self, model_builder, pipeline_config) -> None:
        self.model_builder = model_builder
        self.pipeline_config = pipeline_config
    
    def train(self):
        pass
    
    def val(self):
        pass