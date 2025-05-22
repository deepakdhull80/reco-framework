import os
from typing import *
import numpy as np
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader


class DataLoaderType:
    SIMPLE: str = 'simple'

class DataLoaderConfig(BaseModel):
    name: str
    mini_batch_size: int
    batch_size: int
    total_train_steps: int
    total_val_steps: int
    num_workers: int = 0
    
    def get_file_paths(self, path: str) -> List:
        raise NotImplementedError()


class SimpleDataLoaderConfig(DataLoaderConfig):
    name: str = 'simple'
    
    def get_file_paths(self, path: str, file_format: str) -> List:
        if not os.path.exists(path):
            print("Replace <BASE_PATH> with actual path in hydra-config/data/movielens.yaml")
            raise FileNotFoundError(f"Path not found: {path}")
        files = [f"{path}/{p}" for p in os.listdir(path) if p.endswith(file_format)]
        assert len(files) != 0, "File not found: {}".format(path)
        
        return files


class DataLoaderStrategy:
    def __init__(self, pipeline_cfg) -> None:
        self.pipeline_cfg = pipeline_cfg
    
    def get_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError()


def _collate_fn(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = batches[0]
    result = {}
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            result[key] = torch.cat([batch_[key] for batch_ in batches], dim=0)
        else:
            result[key] = np.concatenate([batch_[key] for batch_ in batches], axis=0)
    return result

class SimpleDataLoaderStrategy(DataLoaderStrategy):
    def __init__(self, pipeline_cfg) -> None:
        self.pipeline_cfg = pipeline_cfg
        self.dataloader_config = pipeline_cfg.dataloader
    
    def get_generator(self):
        from common.data.data_generator import SimpleDataGenerator
        
        return SimpleDataGenerator('train', self.pipeline_cfg), SimpleDataGenerator('val', self.pipeline_cfg)
    
    def get_dataloader(self)-> Tuple[DataLoader, DataLoader]:
        train_gen, val_gen = self.get_generator()
        mini_batch_size = self.pipeline_cfg.dataloader.mini_batch_size
        batch_size = self.pipeline_cfg.dataloader.batch_size
        
        no_mini_batches = int(batch_size / mini_batch_size)
        
        train_dl = DataLoader(
            dataset=train_gen, 
            batch_size=no_mini_batches, 
            collate_fn=_collate_fn,
            num_workers=self.dataloader_config.num_workers
        )
        
        val_dl = DataLoader(
            dataset=val_gen, 
            batch_size=no_mini_batches, 
            collate_fn=_collate_fn,
            num_workers=self.dataloader_config.num_workers
        )
        
        return train_dl, val_dl