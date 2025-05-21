from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, in_features, intermediate_feature:list, bias=True):
        super().__init__()
        self.in_features = in_features
        _in = in_features
        layers = []
        for out in intermediate_feature:
            layers.append(nn.Linear(_in, out, bias=bias))
        layers.append(nn.Linear(out, in_features, bias=bias))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)
        self.ffn = LinearLayer(embed_dim, [embed_dim*2], bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.layernorm1(x + out[0])
        x = self.layernorm2(x + self.dropout(self.ffn(x)))
        return x