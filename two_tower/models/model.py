from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model import ModelWrapper
from two_tower.model_config import TwoTowerConfig

class ItemTower(nn.Module):
    def __init__(self, model_config: TwoTowerConfig) -> None:
        super().__init__()
        
    
    def forward(self, batch):
        pass


class QueryTower(nn.Module):
    def __init__(self, model_config: TwoTowerConfig) -> None:
        super().__init__()
    
    
    def forward(self, batch):
        pass

class TwoTowerModel(ModelWrapper):
    def __init__(self, model_config: TwoTowerConfig) -> None:
        super().__init__(model_config)
        self.model_config = model_config
        
        self.item_tower = ItemTower(model_config)
        self.query_tower = QueryTower(model_config)
        
    
    def forward(self, batch: nn.ModuleDict):
        query_output = self.query_tower(batch)
        item_output = self.item_tower(batch)
        return query_output, item_output
    
    def train_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        query, item = self.forward(batch)
        
    
    def val_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        pass
    
    def get_optimizer_clz(self):
        if self.model_config.optimizer_clz == 'adam':
            return torch.optim.Adam
        else:
            raise ValueError(f"{self.model_config.optimizer_clz} is not available.")
        