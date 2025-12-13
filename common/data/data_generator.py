import pandas as pd
import torch
from torch.utils.data import IterableDataset

from common.pipeline_config import PipelineConfig
import torch.distributed as dist

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0

def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1

class SimpleDataGenerator(IterableDataset):
    def __init__(self, kind: str, pipeline_cfg: PipelineConfig) -> None:
        super().__init__()
        paths = pipeline_cfg.dataloader.get_file_paths(path=pipeline_cfg.data.base_path, file_format=pipeline_cfg.data.file_format)
        self._model_config = pipeline_cfg.model
        
        # Distributed training info
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_distributed = is_distributed()
        
        paths = list(filter(lambda x: kind in x, paths))
        assert len(paths) != 0, "File not found for kind: {}".format(kind)
        
        self.df: pd.DataFrame = None
        if pipeline_cfg.data.file_format == 'pq':
            ## TODO: add functionality to read chunks instead of all data in memory: for large scale solutions.
            self.df = pd.read_parquet(paths)
        else:
            raise NotImplementedError()
        
        self.mini_batch_size = min(pipeline_cfg.dataloader.mini_batch_size, self.df.shape[0])
        self.total_samples = self.df.shape[0]
        # self.total_steps = min(self.df.shape[0]//self.mini_batch_size, getattr(pipeline_cfg.dataloader, f'total_{kind}_steps'))
        base_total_steps = getattr(pipeline_cfg.dataloader, f'total_{kind}_steps')
        
        # Partition data for distributed training
        if self.is_distributed:
            # Each rank processes a subset of the data
            self.total_steps = base_total_steps // self.world_size
            # Offset start position based on rank
            self.start = self.rank * self.total_steps
            print(f'[DEBUG] Rank {self.rank}/{self.world_size}: total_samples: {self.total_samples}, '
                  f'total_steps: {self.total_steps}, start: {self.start}')
        else:
            self.total_steps = base_total_steps
            self.start = 0
            print(f'[DEBUG] total_samples: {self.total_samples}, total_steps: {self.total_steps}')
        
        self.df_idx = 0
        self.preprocess_fn = pipeline_cfg.model.preprocessing_fn
    
    def get_batch(self, idx):
        batch = self.df.iloc[idx * self.mini_batch_size: (idx + 1) * self.mini_batch_size]
        if batch.shape[0] == 0:
            return None
        batch = self._model_config.features.convert_to_platform_type(batch)
        return batch
    
    def __iter__(self):
        # Calculate the end position for this rank
        if self.is_distributed:
            end_step = self.rank * self.total_steps + self.total_steps
        else:
            end_step = self.total_steps
        
        current_step = self.start if self.is_distributed else 0
        
        while current_step < end_step:
            # try:
            batch = self.get_batch(current_step)
            if batch is None:
                break
            self.df_idx = (self.df_idx + 1) % self.total_samples
            current_step += 1
            yield self.preprocess_fn(batch)
            # except Exception as e:
            #     print(e)
            #     print(current_step, end_step, self.df.shape)
            #     # continue
            #     break
    
    def __len__(self):
        """Return the total number of steps/batches in the dataset."""
        return self.total_steps