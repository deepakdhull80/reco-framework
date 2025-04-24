import logging
from collections import defaultdict, deque
import torch
import torch.nn as nn

from common.trainer.training_strategy import TrainingStrategy


logger = logging.getLogger(__name__)

class SimpleTrainingStrategy(TrainingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_kth_train_step = 100
        self.aggregate_k_steps = 100
    
    def fit(self, train_dl, val_dl, model: nn.Module):
        for epoch in range(self.trainer_config.epochs):
            logger.info("Training Epoch %d" % epoch)
            self.train_val(epoch, train_dl, val_dl, model)
            # self.train(epoch, train_dl, model)
            # self.val(epoch, val_dl, model)
    
    def train_val(self, epoch, train_dl, val_dl, model: nn.Module):
        loss = 0
        metrics = defaultdict(int)
        metric_history = defaultdict(lambda: deque(maxlen=self.aggregate_k_steps))
        model.train()
        
        optimizer_clz = model.get_optimizer_clz(self.model_config.optimizer_clz)
        sparse_optimizer_clz = model.get_optimizer_clz(self.model_config.sparse_optimizer_clz)
        
        sparse_params = []
        non_sparse_params = []
        for n, p in model.named_parameters():
            if "embedding_table" in n:
                sparse_params.append(p)
            else:
                non_sparse_params.append(p)
        
        sparse_optimizer: torch.optim.Optimizer = sparse_optimizer_clz(sparse_params, self.model_config.sparse_lr)
        optimizer: torch.optim.Optimizer = optimizer_clz(non_sparse_params, self.model_config.lr)
        sparse_optimizer.zero_grad()
        optimizer.zero_grad()
        train_dl = iter(train_dl)
        idx = 0
        while True:
            try:
                batch = next(train_dl)
            except Exception as e:
                print(e)
                break
            
            optimizer.zero_grad()
            sparse_optimizer.zero_grad()
            
            _loss, _metrics = model.train_step(batch)
            _loss.backward()
            
            optimizer.step()
            sparse_optimizer.step()
            
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
    
    def train(self, epoch, train_dl, model: nn.Module):
        loss = 0
        metrics = defaultdict(int)
        metric_history = defaultdict(lambda: deque(maxlen=self.aggregate_k_steps))
        model.train()
        
        optimizer_clz = model.get_optimizer_clz(self.model_config.optimizer_clz)
        sparse_optimizer_clz = model.get_optimizer_clz(self.model_config.sparse_optimizer_clz)
        
        sparse_params = []
        non_sparse_params = []
        for n, p in model.named_parameters():
            if "embedding_table" in n:
                sparse_params.append(p)
            else:
                non_sparse_params.append(p)
        
        sparse_optimizer: torch.optim.Optimizer = sparse_optimizer_clz(sparse_params, self.model_config.sparse_lr)
        optimizer: torch.optim.Optimizer = optimizer_clz(non_sparse_params, self.model_config.lr)
        sparse_optimizer.zero_grad()
        optimizer.zero_grad()
        
        idx = 0
        # for idx, batch in enumerate(train_dl):
        while True:
            try:
                batch = next(train_dl)
            except:
                break
            
            optimizer.zero_grad()
            sparse_optimizer.zero_grad()
            
            _loss, _metrics = model.train_step(batch)
            _loss.backward()
            
            optimizer.step()
            sparse_optimizer.step()
            
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
            idx += 1
        self.print_log(
                epoch=epoch,
                idx=-1,
                loss=loss,
                metrics=metrics,
                train=True
            )
        return loss, metrics


    @torch.no_grad()
    def val(self, epoch, val_dl, model: nn.Module):
        loss = 0
        metrics = defaultdict(int)
        metric_history = defaultdict(lambda: deque(maxlen=self.aggregate_k_steps))
        model.eval()
        
        for idx, batch in enumerate(val_dl):
            _loss, _metrics = model.val_step(batch)
        
            loss += _loss.cpu().item()
            metrics, loss = self.update_metrics(idx, metrics, _metrics, loss, metric_history)
            if (idx+1) % self.log_kth_train_step == 0:
                self.print_log(
                    epoch=epoch,
                    idx=idx,
                    loss=loss,
                    metrics=metrics,
                    train=False
                )
        self.print_log(
                epoch=epoch,
                idx=-1,
                loss=loss,
                metrics=metrics,
                train=False
            )
        return loss, metrics
    
    
    def update_metrics(self, batch_idx, metric, _b, loss_sum, metric_history):
        for k, v in _b.items():
            metric_history[k].append(v)
            metric[k] = sum(metric_history[k]) / len(metric_history[k])
        
        return metric, loss_sum / (batch_idx + 1)
    
    
    def print_log(self, epoch, idx, loss, metrics, train: bool = True):
        step_type = 'TRAIN' if train else 'EVAL'
        # Format metrics to show averages over the last k steps
        m = {k: f"{v:.4f}" for k, v in metrics.items()}
        logger.info(
            f"[{step_type}], Epoch: {epoch}, Step: {idx}, "
            f"Loss: {loss:.4f}, Metrics (avg last {self.aggregate_k_steps} steps): {m}"
        )