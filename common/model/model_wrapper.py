from typing import Tuple, Dict
import torch
import torch.nn as nn

from common.model.model_config import ModelConfig

class ModelWrapper(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config = model_config
    
    def forward(self, batch: nn.ModuleDict) -> torch.Tensor:
        raise NotImplementedError()
    
    def train_step(self, batch: nn.ModuleDict) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError()
    
    def val_step(self, batch: nn.ModuleDict) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError()
    
    def inference_models(self):
        raise NotImplementedError()
    
    def get_optimizer_clz(self, clz: str):
        return torch.optim.Adam