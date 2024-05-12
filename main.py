import warnings
import hydra
from omegaconf import DictConfig, OmegaConf

from common.hydra.util import init_hydra
from common.pipeline_config import PipelineConfig
from common.pipeline_builder import TrainerPipeline, PipelineOptions

################################################################
warnings.filterwarnings("ignore", category=UserWarning)
################################################################


def execute(pipeline_config: PipelineConfig):
    
    # Model builder
    model_builder = None
    
    # Pipeline Builder
    if pipeline_config.pipeline_name == PipelineOptions.SIMPLE:
        from common.pipeline.simple_pipeline_builder import SimpleTrainerPipeline
        pipeline_cls = SimpleTrainerPipeline
    else:
        raise ModuleNotFoundError(f'Trainer pipeline not found: %s' % pipeline_config.pipeline_name)
    
    pipeline: TrainerPipeline = pipeline_cls(
        model_builder,
        pipeline_config.trainer,
        pipeline_config.dataloader
    )
    
    # start pipeline
    pipeline.execute()

@hydra.main(version_base=None, config_path="hydra-config")
def main_fn(cfg: DictConfig) -> None:
    obj = OmegaConf.to_object(cfg)
    pipeline_cfg = PipelineConfig.parse_obj(obj)
    print(pipeline_cfg)    
    execute(pipeline_cfg)

if __name__ == '__main__':
    init_hydra()
    main_fn()