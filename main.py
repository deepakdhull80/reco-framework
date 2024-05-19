import warnings
import hydra
from omegaconf import DictConfig, OmegaConf

from common.hydra.util import init_hydra
from common.pipeline_config import PipelineConfig
from common.pipeline_builder import TrainerPipeline, PipelineOptions
from common.trainer.simple_training_strategy import SimpleTrainingStrategy
from two_tower.model_builder import TwoTowerBuilder
from common.data.dataloader import SimpleDataLoaderStrategy

################################################################
warnings.filterwarnings("ignore", category=UserWarning)
################################################################


def execute(pipeline_config: PipelineConfig):
    
    # Model builder
    model_config = pipeline_config.model
    model_builder = TwoTowerBuilder(model_config)
    
    # Pipeline Builder
    if pipeline_config.pipeline_name == PipelineOptions.SIMPLE:
        from common.pipeline.simple_pipeline_builder import SimpleTrainerPipeline
        pipeline_cls = SimpleTrainerPipeline
    else:
        raise ModuleNotFoundError(f'Trainer pipeline not found: %s' % pipeline_config.pipeline_name)
    
    # Training Strategy
    training_strategy = SimpleTrainingStrategy(
        model_builder, pipeline_config
    )
    
    # Dataloader
    dataloader_strategy = SimpleDataLoaderStrategy(pipeline_cfg=pipeline_config)
    
    pipeline: TrainerPipeline = pipeline_cls(
        model_builder,
        training_strategy,
        dataloader_strategy
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