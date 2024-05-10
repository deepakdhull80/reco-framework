import hydra
from omegaconf import DictConfig, OmegaConf

from common.hydra.util import init_hydra
from common.pipeline_config import PipelineConfig
from common.constants import ModelType


@hydra.main(version_base=None, config_path="hydra-config")
def main_fn(cfg: DictConfig) -> None:
    obj = OmegaConf.to_object(cfg)
    print(obj)
    pipeline_cfg = PipelineConfig.parse_obj(obj)
    print(pipeline_cfg)

if __name__ == '__main__':
    init_hydra()
    main_fn()