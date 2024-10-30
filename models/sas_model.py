import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MultiHeadAttentionLayers
from config.sas_config import SAS4RecModelConfig

class SAS4Rec(nn.Module):
    def __init__(self, cfg: SAS4RecModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding_layer = nn.Embedding(cfg.items_cardinality, cfg.dim)
        self.position_layer = nn.Embedding(cfg.max_length, cfg.dim)
        
        self.attn_layers = MultiHeadAttentionLayers(
            d=cfg.dim,
            n_blocks=cfg.n_layers,
            head=cfg.head,
            ff_proj=cfg.ff_proj,
            dropout=cfg.dropout
        )
    
    def forward(self, x: torch.LongTensor, position: torch.LongTensor = None):
        x = self.embedding_layer(x)
        if position is not None:
            x = x + self.position_layer(position)
        x = self.attn_layers(x)
        x = x.matmul(self.embedding_layer.weight.T)
        return x