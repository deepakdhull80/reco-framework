from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
    
    def forward(self, q, k, v, causal=False, mask: torch.Tensor = None) -> torch.Tensor:
        sim = torch.matmul(q, k.transpose(1, 2))
        if causal:
            mask = torch.ones(sim.shape[-1], sim.shape[-1]).triu(1).bool().to(sim.device)
            print(mask.dtype)
            sim[:, mask] = - torch.inf
        
        if mask is not None:
            assert mask.dtype in (torch.bool, torch.long), "mask should be one of the type bool, long"
            assert mask.shape == sim.shape[1:], f"mask shape {mask.shape} didn't match with input sim shape {sim.shape[1:]}"
            sim[:, mask] = -1 * torch.inf
        
        return F.softmax((sim) / math.sqrt(self.d), dim=2) @ v

class MHA(nn.Module):
    def __init__(self, d: int, head: int = 1) -> None:
        super().__init__()
        self.head = head
        self.q_layer = nn.Linear(d, d, bias=False)
        self.k_layer = nn.Linear(d, d, bias=False)
        self.v_layer = nn.Linear(d, d, bias=False)
        self.attn = Attention(d)
        
    def forward(self, q, k, v, causal=False, mask: torch.Tensor = None) -> torch.Tensor:
        b, n, d = q.shape
        q = self.q_layer(q).reshape(-1, n, self.head, d//self.head).permute(0, 2, 1, 3).reshape(-1, n, d//self.head).contiguous()
        k = self.k_layer(k).reshape(-1, n, self.head, d//self.head).permute(0, 2, 1, 3).reshape(-1, n, d//self.head).contiguous()
        v = self.v_layer(v).reshape(-1, n, self.head, d//self.head).permute(0, 2, 1, 3).reshape(-1, n, d//self.head).contiguous()
        
        x = self.attn(q, k, v, causal=causal, mask = mask)
        
        x = x.reshape(b, self.head, n, d//self.head).permute(0, 2, 1, 3).reshape(b, n, d).contiguous()
        return x
        


class LayerNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, ))
        self.beta = nn.Parameter(torch.randn(1, ))
        self.epsilon = 1e-6
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        return self.alpha * (x - mu) / torch.sqrt(var + self.epsilon) + self.beta


class FeedForwardLayer(nn.Module):
    def __init__(self, d: int, projection: List[int] = []) -> None:
        super().__init__()
        assert len(projection) > 0, "Feed Forward layer need projection a list\neg: [32, 32]"
        self.d = d
        layers = []
        prev = d
        for p in projection:
            layers.append(nn.Linear(prev, p))
            layers.append(nn.ReLU())
            prev = p
        
        layers.append(
            nn.Linear(prev, d)
        )
        layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d: int, head: int = 1, ff_proj: List[int] = [], dropout: float = 0.2) -> None:
        super().__init__()
        self.layer_norm = LayerNorm()
        self.mha = MHA(d=d, head=head)
        self.ffn = FeedForwardLayer(d=d, projection=ff_proj)
        self.activation = nn.Dropout(p=dropout)
        
    def forward(self, x, kv: torch.Tensor = None, causal: bool = False, mask: torch.Tensor = None) -> torch.Tensor:
        _x = self.layer_norm(x)
        _kv = self.layer_norm(kv) if kv is not None else _x
        x = x + self.activation(self.mha(_x, _kv, _kv, causal=causal, mask=mask))
        x = x + self.activation(self.ffn(self.layer_norm(x)))
        
        return x


class MultiHeadAttentionLayers(nn.Module):
    def __init__(self, d: int, n_blocks: int = 1, head: int = 1, ff_proj: List[int] = [], dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
                MultiHeadAttentionBlock(
                d=d, head=head, ff_proj=ff_proj, dropout=dropout
            )   for _ in range(n_blocks)
            ])
        
    def forward(self, x, kv: torch.Tensor = None, causal: bool = False, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            if kv is None:
                kv = x.clone()
            x = layer(x, kv=kv, causal=causal, mask=mask)
        return x