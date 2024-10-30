from config import ALSMovielensConfig
from models import ALSTrainer


if __name__ == '__main__':
    cfg = ALSMovielensConfig()
    trainer = ALSTrainer(cfg)
    trainer.train()