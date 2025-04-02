
from common.model import ModelBuilder
from bert4rec.model_config import Bert4RecConfig
from bert4rec.models.model import Bert4RecModel

class Bert4RecBuilder(ModelBuilder):
    def __init__(self, model_config: Bert4RecConfig):
        self.model_config = model_config
    
    def build(self, device: str):
        return Bert4RecModel(self.model_config, device)
