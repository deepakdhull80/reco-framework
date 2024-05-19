import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model import ModelWrapper
from two_tower.model_config import TwoTowerConfig

class TwoTowerModel(ModelWrapper):
    def __init__(self, model_config: TwoTowerConfig) -> None:
        super().__init__()
        self.model_config = model_config
    
    def forward(self, batch: nn.ModuleDict):
        pass
    
    def train_step(self, batch: nn.ModuleDict):
        pass
    
    def val_step(self, batch: nn.ModuleDict):
        pass