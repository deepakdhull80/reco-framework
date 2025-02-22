
class ModelBuilder(object):
    def __init__(self, model_config) -> None:
        self.model_config = model_config
    
    def build(self, device: str):
        raise NotImplementedError()