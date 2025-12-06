"""
Distributed training strategy for multi-GPU training using PyTorch DDP.
"""
import logging
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from common.trainer.simple_training_strategy import SimpleTrainingStrategy
from common.utils.distributed_utils import (
    is_distributed, get_rank, is_main_process, reduce_dict, barrier
)

logger = logging.getLogger(__name__)


class DistributedTrainingStrategy(SimpleTrainingStrategy):
    """
    Training strategy for distributed multi-GPU training using DDP.
    
    This strategy extends SimpleTrainingStrategy and adds:
    - Model wrapping with DistributedDataParallel
    - Metric synchronization across processes
    - Checkpoint saving only on rank 0
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_distributed = is_distributed()
        self.rank = get_rank()
        self.is_main = is_main_process()
        
        if self.is_distributed:
            logger.info(f"Initialized DistributedTrainingStrategy on rank {self.rank}")
    
    def fit(self, train_dl, val_dl, model: nn.Module):
        """
        Fit the model using distributed training.
        
        Args:
            train_dl: Training dataloader with DistributedSampler
            val_dl: Validation dataloader with DistributedSampler
            model: Model to train (will be wrapped with DDP)
        """
        # Wrap model with DDP if in distributed mode
        if self.is_distributed and not isinstance(model, DDP):
            model = self._wrap_model_ddp(model)
        
        # Initialize optimizer after wrapping model
        if not self._optimizer_initialized:
            self._init_optimizer(model)
        
        g_ndcg = 0
        for epoch in range(self.trainer_config.epochs):
            # Set epoch for DistributedSampler to ensure proper shuffling
            if hasattr(train_dl.sampler, 'set_epoch'):
                train_dl.sampler.set_epoch(epoch)
            if hasattr(val_dl.sampler, 'set_epoch'):
                val_dl.sampler.set_epoch(epoch)
            
            if self.is_main:
                logger.info(f"Training Epoch {epoch}")
            
            # Training and validation
            self.train(epoch, train_dl, model)
            self.val(epoch, val_dl, model)
            
            # Evaluation (only on main process to avoid redundant computation)
            if self.is_main:
                from common.module.evaluate import evaluate
                # Unwrap DDP model for evaluation
                eval_model = model.module if isinstance(model, DDP) else model
                hr, ndcg = evaluate(eval_model, val_dl, eval_model.device, model_config=self.model_config)
                
                if g_ndcg < ndcg:
                    g_ndcg = ndcg
                    logger.info(f"Best NDCG: {g_ndcg}")
                    # Save the model state (only on main process)
                    from common.pipeline.simple_pipeline_builder import SimpleTrainerPipeline
                    SimpleTrainerPipeline.export_model(
                        self.artifact_dir, eval_model, None, None, training_done=False
                    )
                logger.info(f"\nEval HR: {hr}, NDCG: {ndcg}")
            
            # Synchronize before continuing to next epoch
            barrier()
            
            # Update learning rates
            self.scheduler.step()
            if self.sparse_scheduler is not None:
                self.sparse_scheduler.step()
            
            if self.is_main:
                current_lr = self.scheduler.get_last_lr()[0]
                current_sparse_lr = -1
                if self.sparse_scheduler is not None:
                    current_sparse_lr = self.sparse_scheduler.get_last_lr()[0]
                print(f"  Current Learning Rate: {current_lr:.6f}, Sparse Learning Rate: {current_sparse_lr:.6f}")
                print("*" * 20)
    
    def _wrap_model_ddp(self, model: nn.Module) -> DDP:
        """
        Wrap the model with DistributedDataParallel.
        
        Args:
            model: Model to wrap
        
        Returns:
            DDP-wrapped model
        """
        # Find parameters that don't require gradients (e.g., frozen embeddings)
        # DDP requires all parameters to have gradients or be explicitly marked
        find_unused_parameters = False
        
        ddp_model = DDP(
            model,
            device_ids=[get_rank()] if torch.cuda.is_available() else None,
            output_device=get_rank() if torch.cuda.is_available() else None,
            find_unused_parameters=find_unused_parameters,
        )
        
        logger.info(f"Wrapped model with DDP on rank {self.rank}")
        return ddp_model
    
    def train(self, epoch, train_dl, model: nn.Module):
        """
        Training loop with metric synchronization across processes.
        """
        loss, metrics = super().train(epoch, train_dl, model)
        
        # Synchronize metrics across all processes
        if self.is_distributed:
            metrics_to_reduce = {'loss': loss, **metrics}
            reduced_metrics = reduce_dict(metrics_to_reduce, average=True)
            loss = reduced_metrics.pop('loss')
            metrics = reduced_metrics
        
        return loss, metrics
    
    def val(self, epoch, val_dl, model: nn.Module):
        """
        Validation loop with metric synchronization across processes.
        """
        loss, metrics = super().val(epoch, val_dl, model)
        
        # Synchronize metrics across all processes
        if self.is_distributed:
            metrics_to_reduce = {'loss': loss, **metrics}
            reduced_metrics = reduce_dict(metrics_to_reduce, average=True)
            loss = reduced_metrics.pop('loss')
            metrics = reduced_metrics
        
        return loss, metrics
    
    def print_log(self, epoch, idx, loss, metrics, train: bool = True, _c_loss=None):
        """
        Log training/validation progress (only on main process).
        """
        if self.is_main:
            super().print_log(epoch, idx, loss, metrics, train, _c_loss)
