"""
Distributed dataloader strategy for multi-GPU training.
"""
import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple

from common.data.dataloader import SimpleDataLoaderStrategy, _collate_fn
from common.utils.distributed_utils import is_distributed, get_rank, get_world_size
import logging

logger = logging.getLogger(__name__)


class DistributedDataLoaderStrategy(SimpleDataLoaderStrategy):
    """
    DataLoader strategy for distributed training.
    
    This strategy extends SimpleDataLoaderStrategy and adds:
    - DistributedSampler to partition data across GPUs
    - Proper handling of num_workers per GPU
    - Automatic epoch setting for proper shuffling
    """
    
    def __init__(self, pipeline_cfg) -> None:
        super().__init__(pipeline_cfg)
        self.is_distributed = is_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        if self.is_distributed:
            logger.info(
                f"Initialized DistributedDataLoaderStrategy: "
                f"rank={self.rank}, world_size={self.world_size}"
            )
    
    def get_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create distributed dataloaders for IterableDataset.
        
        Note: SimpleDataGenerator handles distributed partitioning internally,
        so we don't need DistributedSampler or worker_init_fn.
        
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        from torch.utils.data import IterableDataset
        
        train_gen, val_gen = self.get_generator()
        mini_batch_size = self.pipeline_cfg.dataloader.mini_batch_size
        batch_size = self.pipeline_cfg.dataloader.batch_size
        
        no_mini_batches = int(batch_size / mini_batch_size)
        
        # Check if dataset is IterableDataset
        is_iterable = isinstance(train_gen, IterableDataset)
        
        if is_iterable:
            # IterableDataset - no sampler needed
            # SimpleDataGenerator handles distributed partitioning internally
            logger.info(
                f"Using IterableDataset on rank {self.rank}/{self.world_size}"
            )
            
            train_dl = DataLoader(
                dataset=train_gen,
                batch_size=no_mini_batches,
                collate_fn=_collate_fn,
                num_workers=self.dataloader_config.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            
            val_dl = DataLoader(
                dataset=val_gen,
                batch_size=no_mini_batches,
                collate_fn=_collate_fn,
                num_workers=self.dataloader_config.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            
        else:
            # Map-style dataset - use DistributedSampler if in distributed mode
            if self.is_distributed:
                train_sampler = DistributedSampler(
                    train_gen,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    drop_last=False
                )
                val_sampler = DistributedSampler(
                    val_gen,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    drop_last=False
                )
                
                logger.info(
                    f"Created DistributedSampler: "
                    f"train_samples={len(train_gen)}, val_samples={len(val_gen)}, "
                    f"samples_per_gpu={len(train_gen) // self.world_size}"
                )
            else:
                train_sampler = None
                val_sampler = None
            
            # Create DataLoaders
            train_dl = DataLoader(
                dataset=train_gen,
                batch_size=no_mini_batches,
                sampler=train_sampler,
                shuffle=True if train_sampler is None else False,
                collate_fn=_collate_fn,
                num_workers=self.dataloader_config.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False
            )
            
            val_dl = DataLoader(
                dataset=val_gen,
                batch_size=no_mini_batches,
                sampler=val_sampler,
                shuffle=False,
                collate_fn=_collate_fn,
                num_workers=self.dataloader_config.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False
            )
        
        logger.info(
            f"Created DataLoaders on rank {self.rank}"
        )
        
        return train_dl, val_dl
