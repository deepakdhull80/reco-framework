import logging
from collections import defaultdict, deque
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from common.trainer.training_strategy import TrainingStrategy
from common.module.evaluate import evaluate
from common.pipeline.simple_pipeline_builder import SimpleTrainerPipeline

logger = logging.getLogger(__name__)

class AccelerateTrainingStrategy(TrainingStrategy):
    def __init__(self, accelerator: Accelerator, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.accelerator = accelerator
        self.log_kth_train_step = 100
        self.aggregate_k_steps = 100
        self._optimizer_initialized = False
        self.artifact_dir = self.model_config.model_dir
    
    def fit(self, train_dl, val_dl, model: nn.Module):
        if not self._optimizer_initialized:
            self._init_optimizer(model)
        
        # Prepare everything with accelerator
        # Note: If optimizers are None (e.g. sparse optimizer not used), we shouldn't pass None to prepare
        prepare_args = [model, self.optimizer, train_dl, val_dl]

        train_step_fn = model.train_step
        val_step_fn = model.val_step

        if self.sparse_optimizer is not None:
            prepare_args.append(self.sparse_optimizer)
            if self.sparse_scheduler is not None:
                prepare_args.append(self.sparse_scheduler)
        
        prepare_args.append(self.scheduler)
        
        prepared_objects = self.accelerator.prepare(*prepare_args)
        
        # Unpack prepared objects
        # The order matches the input list
        idx = 0
        model = prepared_objects[idx]; idx += 1
        self.optimizer = prepared_objects[idx]; idx += 1
        train_dl = prepared_objects[idx]; idx += 1
        val_dl = prepared_objects[idx]; idx += 1
        
        if self.sparse_optimizer is not None:
            self.sparse_optimizer = prepared_objects[idx]; idx += 1
            if self.sparse_scheduler is not None:
                self.sparse_scheduler = prepared_objects[idx]; idx += 1
        
        self.scheduler = prepared_objects[idx]; idx += 1
        
        # Store for retrieval
        self.model = model

        g_ndcg = 0
        for epoch in range(self.trainer_config.epochs):
            logger.info("Training Epoch %d" % epoch)
            self.train(epoch, train_dl, model, train_step_fn)
            self.val(epoch, val_dl, model, val_step_fn)
            
            # For evaluation, we ideally want to run on all processes and average, 
            # or just run on main process if data fits. 
            # Current evaluate() runs on passed dataloader. 
            # Since val_dl is prepared, it is sharded. 
            # We need to aggregate metrics.
            # For now, let's run evaluate which returns local metrics, and we should average them?
            # Actually evaluate() calculates HR/NDCG on the local slice.
            # We can use a simplified approach: just run evaluate as is, 
            # but we need to handle the metric aggregation if we want the "global" best model.
            
            hr, ndcg = self._distributed_evaluate(model, val_dl)
            
            if self.accelerator.is_main_process:
                if g_ndcg < ndcg:
                    g_ndcg = ndcg
                    logger.info(f"Best NDCG: {g_ndcg}")
                    # Save the model state
                    # Unwrap model before saving
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    SimpleTrainerPipeline.export_model(self.artifact_dir, unwrapped_model, None, None, training_done=False)
                logger.info(f"\nEval HR: {hr}, NDCG: {ndcg}")
            
            self.scheduler.step()
            if self.sparse_scheduler is not None:
                self.sparse_scheduler.step()
            
            # Logging LR
            current_lr = self.scheduler.get_last_lr()[0]
            current_sparse_lr = -1
            if self.sparse_scheduler is not None:
                current_sparse_lr = self.sparse_scheduler.get_last_lr()[0]
                
            if self.accelerator.is_main_process:
                print(f"  Current Learning Rate: {current_lr:.6f}, Sparse Learning Rate: {current_sparse_lr:.6f}")
                print("*" * 20)
    
    def get_trained_model(self):
        """Returns the prepared model."""
        return self.model

    def _distributed_evaluate(self, model, dataloader):
        """
        Runs evaluation and aggregates metrics across processes.
        """
        # Run local evaluation
        # evaluate() expects a model and dataloader. 
        # The dataloader is already prepared (sharded).
        # evaluate returns (avg_hr, avg_ndcg) for the local shard.
        # We need to know the number of samples to properly weight the average?
        # evaluate() implementation in common/module/evaluate.py computes simple averages.
        # Ideally we refactor evaluate to return totals, but let's assume balanced shards for now 
        # or just simple averaging of averages (approximate).
        
        local_hr, local_ndcg = evaluate(model, dataloader, self.accelerator.device, model_config=self.model_config)
        
        # Convert to tensor for reduction
        metrics = torch.tensor([local_hr, local_ndcg], device=self.accelerator.device)
        
        # Reduce (sum) across processes
        reduced_metrics = self.accelerator.reduce(metrics, reduction="mean")
        
        return reduced_metrics[0].item(), reduced_metrics[1].item()

    def _init_optimizer(self, model: nn.Module):
        # We can use the unwrap_model to access underlying attributes if needed,
        # but here model should have the methods as it's a pytorch module (or wrapped)
        # But if wrapped by DDP/Accumulate, it might hide methods if not handled.
        # Accelerate wrapped model still behaves like the model usually.
        # However, accessing `get_optimizer_clz` might require unwrapping if it's a custom method on the model class.
        
        unwrapped = self.accelerator.unwrap_model(model)
        
        print("Initializing optimizer")
        optimizer_clz = unwrapped.get_optimizer_clz(self.model_config.optimizer_clz)
        sparse_optimizer_clz = unwrapped.get_optimizer_clz(self.model_config.sparse_optimizer_clz)
        
        sparse_params = []
        non_sparse_params = []
        for n, p in model.named_parameters():
             # We should check if model is wrapped, param names might change (e.g. module.layer...)
             # but named_parameters() usually handles it.
            if "embedding_table" in n:
                sparse_params.append(p)
            else:
                non_sparse_params.append(p)
        
        if not self._optimizer_initialized or not hasattr(self, 'optimizer'):
            if len(sparse_params) != 0:
                self.sparse_optimizer = sparse_optimizer_clz(sparse_params, self.model_config.sparse_lr)
                self.sparse_scheduler = CosineAnnealingLR(self.sparse_optimizer, self.trainer_config.epochs//2)
                self.sparse_optimizer.zero_grad()
            else:
                self.sparse_optimizer = None
                self.sparse_scheduler = None
                
            self.optimizer = optimizer_clz(non_sparse_params, self.model_config.lr)
            self.scheduler = CosineAnnealingLR(self.optimizer, self.trainer_config.epochs//2)
            self.optimizer.zero_grad()
            self._optimizer_initialized = True
        
    def train(self, epoch, train_dl, model: nn.Module, train_step_fn):
        loss = 0
        metrics = defaultdict(int)
        metric_history = defaultdict(lambda: deque(maxlen=self.aggregate_k_steps))
        model.train()
        
        idx = 0
        train_iter = iter(train_dl)
        _loss = 0
        num_batches = 0 

        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error during training: {e}")
                raise
            
            self.optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()
            
            # Forward
            _loss, _metrics = train_step_fn(batch)
            
            # Backward - use accelerator
            self.accelerator.backward(_loss)
            
            self.optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()
            
            _loss = _loss.item() # Accelerate handles device sync if needed? .item() triggers sync usually
            metrics, loss = self.update_metrics(idx, metrics, _metrics, _loss, metric_history)
            num_batches += 1

            if (idx + 1) % self.log_kth_train_step == 0:
                self.print_log(
                    epoch=epoch,
                    idx=idx,
                    loss=loss,  # Average loss
                    metrics=metrics,
                    train=True,
                    _c_loss=_loss
                )
            idx += 1

        # Final logging
        self.print_log(
            epoch=epoch,
            idx=-1,  # Indicate final step
            loss=loss,  # Average loss
            metrics=metrics,
            train=True,
            _c_loss=None  # No current loss for final step
        )
        return loss, metrics  # Return average loss


    @torch.no_grad()
    def val(self, epoch, val_dl, model: nn.Module, val_step_fn):
        loss = 0
        metrics = defaultdict(int)
        metric_history = defaultdict(lambda: deque(maxlen=self.aggregate_k_steps))
        model.eval()
        _loss = 0
        num_batches = 0

        for idx, batch in enumerate(val_dl):
            _loss, _metrics = val_step_fn(batch)
            _loss = _loss.item()
            metrics, loss = self.update_metrics(idx, metrics, _metrics, _loss, metric_history)
            num_batches += 1

            if (idx + 1) % self.log_kth_train_step == 0:
                self.print_log(
                    epoch=epoch,
                    idx=idx,
                    loss=loss,  # Average loss
                    metrics=metrics,
                    train=False,
                    _c_loss=_loss
                )

        # Final logging
        self.print_log(
            epoch=epoch,
            idx=-1,  # Indicate final step
            loss=loss,  # Average loss
            metrics=metrics,
            train=False,
            _c_loss=None  # No current loss for final step
        )
        return loss, metrics  # Return average loss
    
    
    def update_metrics(self, batch_idx, metric, _b, loss, metric_history):
        for k, v in _b.items():
            metric_history[k].append(v)
            metric[k] = sum(metric_history[k]) / len(metric_history[k])
        
        # Use metric_history for loss as well
        metric_history['loss'].append(loss)
        avg_loss = sum(metric_history['loss']) / len(metric_history['loss'])
        
        return metric, avg_loss
    
    
    def print_log(self, epoch, idx, loss, metrics, train: bool = True, _c_loss=None):
        if not self.accelerator.is_main_process:
            return

        step_type = 'TRAIN' if train else 'EVAL'
        
        if idx == -1:
            step_info = "FINAL"
        else:
            step_info = f"Step: {idx + 1}"
        
        loss = float(loss) if loss is not None else 0.0
        _c_loss = float(_c_loss) if _c_loss is not None else 0.0

        m = {k: round(v, 4) for k, v in metrics.items()}
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in m.items()])

        logger.info(
            "[{step_type}], Epoch: {epoch}, {step_info}, "
            "AVGLoss: {loss:.4f}, CLoss: {c_loss}, "
            "Metrics (avg last {k_steps} steps): {metrics}".format(
                step_type=step_type,
                epoch=epoch,
                step_info=step_info,
                loss=loss,
                c_loss=f"{_c_loss:.4f}" if _c_loss is not None else "N/A",
                k_steps=self.aggregate_k_steps,
                metrics=metrics_str
            )
        )
