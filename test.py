import torch

from layers import MultiHeadAttentionLayers
from config import (
    ALSMovielensConfig, SAS4RecConfig
)
from models import (
    ALSTrainer, SAS4Rec
    )

if __name__ == '__main__':
    """
    # ALS integration testing
    cfg = ALSMovielensConfig()
    trainer = ALSTrainer(cfg)
    trainer.train()
    """
    
    
    # Layers unit testing
    """    
    fn = MultiHeadAttentionLayers(128, n_blocks=2, head=1, ff_proj=[256], dropout=0.2)
    x = torch.randn(2, 100, 128)
    kv = torch.randn(2, 100, 128)
    
    print(fn(x, casual=True).shape)
    print(fn(x, kv=kv, casual=True).shape)
    """
    cfg = SAS4RecConfig()
    x = torch.randint(0, 1_000_000, size=(3, cfg.model_cfg.max_length))
    position = torch.arange(0, cfg.model_cfg.max_length).repeat(3, 1)
    print(x.shape)
    model = SAS4Rec(cfg = cfg.model_cfg)
    out = model(x, position=position)
    print(out.shape)