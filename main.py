import os
import warnings
import hydra
import logging
from accelerate import Accelerator

from omegaconf import DictConfig, OmegaConf

from common.hydra.util import init_hydra
from common.pipeline_config import PipelineConfig
from common.pipeline_builder import TrainerPipeline, PipelineOptions
from common.trainer.simple_training_strategy import SimpleTrainingStrategy
from common.trainer.accelerate_training_strategy import AccelerateTrainingStrategy
from two_tower.model_builder import TwoTowerBuilder
from bert4rec.model_builder import Bert4RecBuilder
from common.data.dataloader import SimpleDataLoaderStrategy



################################################################
warnings.filterwarnings("ignore", category=UserWarning)

################################################################


def execute(pipeline_config: PipelineConfig, accelerator: Accelerator):
    # Model Builder
    model_config = pipeline_config.model
    os.makedirs(model_config.model_dir, exist_ok=True)
    
    if accelerator.is_main_process:
        print(f"Model Selection: {model_config.name}")
        print(f"Distributed Training: {pipeline_config.distributed}")
        print(f"Device: {accelerator.device}")
    
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
    
    # Dataloader Strategy - ALWAYS use SimpleDataLoaderStrategy
    # Accelerate handles wrapping. SimpleDataGenerator handles sharding via dist.get_rank().
    dataloader_strategy = SimpleDataLoaderStrategy(pipeline_cfg=pipeline_config)
    
    # Training Strategy
    if pipeline_config.distributed:
        training_strategy = AccelerateTrainingStrategy(
            accelerator=accelerator,
            model_builder=model_builder,
            dataloader_strategy=dataloader_strategy,
            trainer_config=pipeline_config.trainer,
            model_config=model_config
        )
    else:
        training_strategy = SimpleTrainingStrategy(
            model_builder=model_builder,
            dataloader_strategy=dataloader_strategy,
            trainer_config=pipeline_config.trainer,
            model_config=model_config
        )
    
    # Use accelerator's device
    device = str(accelerator.device)
    
    pipeline: TrainerPipeline = pipeline_cls(
        model_builder,
        training_strategy,
        dataloader_strategy,
        device=device,
        artifact_dir=model_config.model_dir,
        accelerator=accelerator
    )
    
    # start pipeline
    pipeline.execute()


@hydra.main(version_base=None, config_path="hydra-config")
def main_fn(cfg: DictConfig) -> None:
    obj = OmegaConf.to_object(cfg)
    pipeline_cfg = PipelineConfig.model_validate(obj)
    
    # Initialize Accelerator
    # Default to no mixed precision unless specified
    # We could add a 'mixed_precision' field to PipelineConfig if needed, but for now we trust env or default.
    # Note: If mixed_precision is not passed, it can still be enabled via `accelerate config` or CLI args.
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(pipeline_cfg)
    
    execute(pipeline_cfg, accelerator)

if __name__ == '__main__':
    init_hydra()
    main_fn()