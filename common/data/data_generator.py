import pandas as pd
import torch
from torch.utils.data import IterableDataset

from common.pipeline_config import PipelineConfig

class SimpleDataGenerator(IterableDataset):
    def __init__(self, kind: str, pipeline_cfg: PipelineConfig) -> None:
        super().__init__()
        paths = pipeline_cfg.dataloader.get_file_paths(path=pipeline_cfg.data.base_path, file_format=pipeline_cfg.data.file_format)
        self._model_config = pipeline_cfg.model
        
        
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
        self.total_steps = getattr(pipeline_cfg.dataloader, f'total_{kind}_steps')
        self.start = 0
        self.df_idx = 0
        self.preprocess_fn = pipeline_cfg.model.preprocessing_fn
    
    def get_batch(self, idx):
        batch = self.df.iloc[idx * self.mini_batch_size: (idx + 1) * self.mini_batch_size]
        batch = self._model_config.features.convert_to_platform_type(batch)
        return batch
    
    def __iter__(self):
        while self.start < self.total_steps:
            try:
                batch = self.get_batch(self.start)
                self.df_idx = (self.df_idx + 1) % self.total_samples
                self.start += 1
                yield self.preprocess_fn(batch)
            except Exception as e:
                print(e)
                print(self.start, self.total_steps, self.df.shape)
                # continue
                break
        
        self.start = 0