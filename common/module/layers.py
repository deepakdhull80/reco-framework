from typing import Tuple
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
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)
        self.ffn = LinearLayer(embed_dim, [embed_dim*2], bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, attn_mask=None):
        x = self.layernorm1(x + self.mha(x, x, x, attn_mask=attn_mask))
        x = self.layernorm2(x + self.dropout(self.ffn(x)))
        return x