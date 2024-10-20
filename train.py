from config.als_movielens import ALSMovielensConfig
from models.als_model import ALSTrainer


if __name__ == '__main__':
    cfg = ALSMovielensConfig()
    trainer = ALSTrainer(cfg)
    trainer.train()