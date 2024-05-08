import hydra
from omegaconf import DictConfig, OmegaConf

from common.hydra.util import init_hydra



@hydra.main(version_base=None, config_path="hydra-config")
def main_fn(cfg: DictConfig) -> None:
    obj = OmegaConf.to_object(cfg)
    print(obj)

if __name__ == '__main__':
    init_hydra()
    main_fn()