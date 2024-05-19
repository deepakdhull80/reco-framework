
from common.model import ModelBuilder
from two_tower.model_config import TwoTowerConfig
from two_tower.models.model import TwoTowerModel

class TwoTowerBuilder(ModelBuilder):
    def __init__(self, model_config: TwoTowerConfig):
        self.model_config = model_config
    
    def build(self):
        return TwoTowerModel(self.model_config)
