import os
import warnings
import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from common.hydra.util import init_hydra
from common.pipeline_config import PipelineConfig
from common.pipeline_builder import TrainerPipeline, PipelineOptions
from common.trainer.simple_training_strategy import SimpleTrainingStrategy
from common.trainer.distributed_training_strategy import DistributedTrainingStrategy
from two_tower.model_builder import TwoTowerBuilder
from bert4rec.model_builder import Bert4RecBuilder
from common.data.dataloader import SimpleDataLoaderStrategy
from common.data.distributed_dataloader_strategy import DistributedDataLoaderStrategy
from common.utils.distributed_utils import (
    setup_distributed, cleanup_distributed, is_distributed, 
    get_local_rank, is_main_process
)


################################################################
warnings.filterwarnings("ignore", category=UserWarning)

################################################################


def execute(pipeline_config: PipelineConfig):
    # Model Builder
    model_config = pipeline_config.model
    os.makedirs(model_config.model_dir, exist_ok=True)
    
    if is_main_process():
        print(f"Model Selection: {model_config.name}")
        print(f"Distributed Training: {pipeline_config.distributed}")
        print(f"Device: {pipeline_config.device}")
    
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
    
    # Dataloader Strategy - select based on distributed mode
    if pipeline_config.distributed and is_distributed():
        dataloader_strategy = DistributedDataLoaderStrategy(pipeline_cfg=pipeline_config)
    else:
        dataloader_strategy = SimpleDataLoaderStrategy(pipeline_cfg=pipeline_config)
    
    # Training Strategy - select based on distributed mode
    if pipeline_config.distributed and is_distributed():
        training_strategy = DistributedTrainingStrategy(
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
    
    # Determine device
    device = pipeline_config.device
    if is_distributed():
        # In distributed mode, each process uses its own GPU
        import torch
        if torch.cuda.is_available():
            device = f"cuda:{get_local_rank()}"
        else:
            device = "cpu"
    
    pipeline: TrainerPipeline = pipeline_cls(
        model_builder,
        training_strategy,
        dataloader_strategy,
        device=device,
        artifact_dir=model_config.model_dir
    )
    
    # start pipeline
    pipeline.execute()


@hydra.main(version_base=None, config_path="hydra-config")
def main_fn(cfg: DictConfig) -> None:
    obj = OmegaConf.to_object(cfg)
    pipeline_cfg = PipelineConfig.model_validate(obj)
    
    # Initialize distributed training if enabled
    if pipeline_cfg.distributed:
        os.environ['DIST_BACKEND'] = pipeline_cfg.backend
        setup_distributed()
    
    if is_main_process():
        print(pipeline_cfg)
    
    try:
        execute(pipeline_cfg)
    finally:
        # Cleanup distributed training
        if pipeline_cfg.distributed:
            cleanup_distributed()

if __name__ == '__main__':
    init_hydra()
    main_fn()