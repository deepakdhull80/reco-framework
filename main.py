import os
import warnings
import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from common.hydra.util import init_hydra
from common.pipeline_config import PipelineConfig
from common.pipeline_builder import TrainerPipeline, PipelineOptions
from common.trainer.simple_training_strategy import SimpleTrainingStrategy
from two_tower.model_builder import TwoTowerBuilder
from bert4rec.model_builder import Bert4RecBuilder
from common.data.dataloader import SimpleDataLoaderStrategy

################################################################
warnings.filterwarnings("ignore", category=UserWarning)

# todo: instead of static file fix with dynamic based upon timestamp.
os.makedirs("logs", exist_ok=True)
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs/train.log')
logging.basicConfig(filename=log_file_path,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
################################################################


def execute(pipeline_config: PipelineConfig):
    # Model Builder
    model_config = pipeline_config.model
    print(f"Model Selection: {model_config.name}")
    if 'two_tower' in model_config.name:
        model_builder = TwoTowerBuilder(model_config)
    elif "bert4rec" in model_config.name:
        model_builder = Bert4RecBuilder(model_config)
    else:
        raise NotImplementedError(f"{model_config.name} implemented not found.")
    
    # Pipeline Builder
    if pipeline_config.pipeline_name == PipelineOptions.SIMPLE:
        from common.pipeline.simple_pipeline_builder import SimpleTrainerPipeline
        pipeline_cls = SimpleTrainerPipeline
    else:
        raise ModuleNotFoundError(f'Trainer pipeline not found: %s' % pipeline_config.pipeline_name)
    
    # Dataloader
    dataloader_strategy = SimpleDataLoaderStrategy(pipeline_cfg=pipeline_config)
    
    # Training Strategy
    training_strategy = SimpleTrainingStrategy(
        model_builder = model_builder, 
        dataloader_strategy = dataloader_strategy, 
        trainer_config = pipeline_config.trainer,
        model_config = model_config
    )
    
    pipeline: TrainerPipeline = pipeline_cls(
        model_builder,
        training_strategy,
        dataloader_strategy,
        device=pipeline_config.device
    )
    
    # start pipeline
    pipeline.execute()

@hydra.main(version_base=None, config_path="hydra-config")
def main_fn(cfg: DictConfig) -> None:
    obj = OmegaConf.to_object(cfg)
    pipeline_cfg = PipelineConfig.model_validate(obj)
    print(pipeline_cfg)    
    execute(pipeline_cfg)

if __name__ == '__main__':
    init_hydra()
    main_fn()