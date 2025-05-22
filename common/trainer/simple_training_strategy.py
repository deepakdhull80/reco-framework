import logging
from collections import defaultdict, deque
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from common.trainer.training_strategy import TrainingStrategy
from common.module.evaluate import evaluate
from common.pipeline.simple_pipeline_builder import SimpleTrainerPipeline

logger = logging.getLogger(__name__)

class SimpleTrainingStrategy(TrainingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_kth_train_step = 100
        self.aggregate_k_steps = 100
        self._optimizer_initialized = False
        self.artifact_dir = self.model_config.model_dir
    
    def fit(self, train_dl, val_dl, model: nn.Module):
        g_ndcg = 0
        for epoch in range(self.trainer_config.epochs):
            logger.info("Training Epoch %d" % epoch)
            # self.train_val(epoch, train_dl, val_dl, model)
            self.train(epoch, train_dl, model)
            self.val(epoch, val_dl, model)
            hr, ndcg = evaluate(model, val_dl, eval_k=10)
            if g_ndcg < ndcg:
                g_ndcg = ndcg
                logger.info(f"Best NDCG: {g_ndcg}")
                # Save the model state
                SimpleTrainerPipeline.export_model(self.artifact_dir, model, None, None, training_done=False)
            logger.info(f"\nEval HR: {hr}, NDCG: {ndcg}")
            self.scheduler.step()
            self.sparse_scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            current_sparse_lr = self.sparse_scheduler.get_last_lr()[0]
            print(f"  Current Learning Rate: {current_lr:.6f}, Sparse Learning Rate: {current_sparse_lr:.6f}")
            print("*" * 20)
    
    def train_val(self, epoch, train_dl, val_dl, model: nn.Module):
        loss = 0
        metrics = defaultdict(int)
        metric_history = defaultdict(lambda: deque(maxlen=self.aggregate_k_steps))
        model.train()
        if self._optimizer_initialized is False:
            self._init_optimizer(model)
        train_dl = iter(train_dl)
        idx = 0
        while True:
            try:
                batch = next(train_dl)
            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error during training: {e}")
                raise
            
            self.optimizer.zero_grad()
            self.sparse_optimizer.zero_grad()
            
            _loss, _metrics = model.train_step(batch)
            _loss.backward()
            
            self.optimizer.step()
            self.sparse_optimizer.step()
            
            loss += _loss.cpu().item()
            metrics, loss = self.update_metrics(idx, metrics, _metrics, loss, metric_history)
            if (idx+1) % self.log_kth_train_step == 0:
                self.print_log(
                    epoch=epoch,
                    idx=idx,
                    loss=loss,
                    metrics=metrics,
                    train=True
                )
                self.val(epoch, val_dl, model)
            idx += 1
        self.print_log(
                epoch=epoch,
                idx=-1,
                loss=loss,
                metrics=metrics,
                train=True
            )
        return loss, metrics
    
    def _init_optimizer(self, model: nn.Module):
        print("Initializing optimizer")
        optimizer_clz = model.get_optimizer_clz(self.model_config.optimizer_clz)
        sparse_optimizer_clz = model.get_optimizer_clz(self.model_config.sparse_optimizer_clz)
        
        sparse_params = []
        non_sparse_params = []
        for n, p in model.named_parameters():
            if "embedding_table" in n:
                sparse_params.append(p)
            else:
                non_sparse_params.append(p)
        
        # Reinitialize optimizers if they are not already initialized or if parameters change
        if not self._optimizer_initialized or not hasattr(self, 'optimizer'):
            self.sparse_optimizer: torch.optim.Optimizer = sparse_optimizer_clz(sparse_params, self.model_config.sparse_lr)
            self.optimizer: torch.optim.Optimizer = optimizer_clz(non_sparse_params, self.model_config.lr)
            self.scheduler = CosineAnnealingLR(self.optimizer, self.trainer_config.epochs//2)
            self.sparse_scheduler = CosineAnnealingLR(self.sparse_optimizer, self.trainer_config.epochs//2)
            self.sparse_optimizer.zero_grad()
            self.optimizer.zero_grad()
            self._optimizer_initialized = True
        
    def train(self, epoch, train_dl, model: nn.Module):
        loss = 0
        metrics = defaultdict(int)
        metric_history = defaultdict(lambda: deque(maxlen=self.aggregate_k_steps))
        model.train()
        if self._optimizer_initialized is False:
            self._init_optimizer(model)
        idx = 0
        train_dl = iter(train_dl)
        _loss = 0
        num_batches = 0

        while True:
            try:
                batch = next(train_dl)
            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error during training: {e}")
                raise
            
            self.optimizer.zero_grad()
            self.sparse_optimizer.zero_grad()
            
            _loss, _metrics = model.train_step(batch)
            _loss.backward()
            
            self.optimizer.step()
            self.sparse_optimizer.step()
            _loss = _loss.cpu().item()
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
    def val(self, epoch, val_dl, model: nn.Module):
        loss = 0
        metrics = defaultdict(int)
        metric_history = defaultdict(lambda: deque(maxlen=self.aggregate_k_steps))
        model.eval()
        _loss = 0
        num_batches = 0

        for idx, batch in enumerate(val_dl):
            _loss, _metrics = model.val_step(batch)
            _loss = _loss.cpu().item()
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
        """
        Logs the training or evaluation progress.

        Args:
            epoch (int): Current epoch number.
            idx (int): Current step index. Use -1 for final logging.
            loss (float): Average loss.
            metrics (dict): Dictionary of metrics.
            train (bool): Whether this is a training log (True) or evaluation log (False).
            _c_loss (float): Current loss for the step (optional).
        """
        step_type = 'TRAIN' if train else 'EVAL'
        
        # Handle final logging when idx = -1
        if idx == -1:
            step_info = "FINAL"
        else:
            step_info = f"Step: {idx + 1}"  # Use 1-based indexing for readability
        
        # Ensure loss and _c_loss are floats
        loss = float(loss) if loss is not None else 0.0
        _c_loss = float(_c_loss) if _c_loss is not None else 0.0

        # Format metrics to show averages over the last k steps
        m = {k: round(v, 4) for k, v in metrics.items()}  # Round metrics to 4 decimal places
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in m.items()])  # Convert metrics to a string

        # Log the information
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