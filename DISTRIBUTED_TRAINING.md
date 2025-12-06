# Distributed Multi-GPU Training Guide

This guide explains how to use distributed multi-GPU training in the recommendation framework.

## Overview

The framework now supports distributed training using PyTorch's DistributedDataParallel (DDP). This allows you to:
- Train models across multiple GPUs on a single machine or multiple machines
- Automatically partition data across GPUs using `DistributedSampler`
- Synchronize gradients and metrics across all processes
- Scale training to larger batch sizes and faster training times

## Quick Start

### Single-GPU Training (Existing)

```bash
python main.py +runner=train-bert4rec
```

### Multi-GPU Distributed Training (New)

```bash
# Using torchrun (recommended for PyTorch 1.9+)
torchrun --nproc_per_node=2 main.py +runner=train-bert4rec-distributed

# Or using torch.distributed.launch (older PyTorch versions)
python -m torch.distributed.launch --nproc_per_node=2 main.py +runner=train-bert4rec-distributed
```

Replace `2` with the number of GPUs you want to use.

## Configuration

### Distributed Training Config

Create or modify a runner config file (e.g., `hydra-config/runner/train-bert4rec-distributed.yaml`):

```yaml
defaults:
  - dataloader: distributed
  - model: bert4rec
  - data: movielens
  - trainer: default

dataloader:
  mini_batch_size: 256
  batch_size: 256
  num_workers: 4  # Workers per GPU

trainer:
  epochs: 100

pipeline_name: simple
env: pytorch
device: cuda  # Will be automatically set to cuda:0, cuda:1, etc.
distributed: true
backend: nccl  # Use 'nccl' for NVIDIA GPUs, 'gloo' for CPU
```

### Key Configuration Parameters

- **`distributed: true`**: Enables distributed training mode
- **`backend: nccl`**: Communication backend
  - `nccl`: Best for NVIDIA GPUs (recommended)
  - `gloo`: For CPU or mixed GPU/CPU training
- **`dataloader.name: distributed`**: Uses `DistributedDataLoaderStrategy`
- **`num_workers`**: Number of data loading workers **per GPU**

## How It Works

### Architecture

1. **Process Initialization**: Each GPU runs a separate process with a unique rank
2. **Data Partitioning**: `DistributedSampler` ensures each GPU sees different data
3. **Model Replication**: Each process has its own copy of the model
4. **Gradient Synchronization**: DDP automatically averages gradients across GPUs
5. **Metric Aggregation**: Custom reduction logic averages metrics across processes
6. **Checkpoint Saving**: Only rank 0 saves checkpoints to avoid conflicts

### Components

#### 1. Distributed Utilities (`common/utils/distributed_utils.py`)
- `setup_distributed()`: Initialize process group
- `cleanup_distributed()`: Clean up resources
- `reduce_dict()`: Average metrics across GPUs
- `is_main_process()`: Check if current process is rank 0

#### 2. Distributed Training Strategy (`common/trainer/distributed_training_strategy.py`)
- Wraps model with `DistributedDataParallel`
- Synchronizes metrics after each epoch
- Only saves checkpoints on rank 0

#### 3. Distributed DataLoader Strategy (`common/data/distributed_dataloader_strategy.py`)
- Uses `DistributedSampler` to partition data
- Automatically handles epoch shuffling
- Configures workers per GPU

## Best Practices

### 1. Batch Size Scaling

When using N GPUs, the effective batch size is `batch_size * N`. You may want to:
- Keep the same per-GPU batch size and increase total throughput
- Or reduce per-GPU batch size if memory is limited

### 2. Learning Rate Scaling

Consider scaling the learning rate when using multiple GPUs:
```yaml
model:
  lr: 0.001  # For 1 GPU
  # lr: 0.002  # For 2 GPUs (linear scaling)
```

### 3. Number of Workers

Set `num_workers` based on your CPU cores:
```yaml
dataloader:
  num_workers: 4  # 4 workers per GPU
```

### 4. Backend Selection

- **NVIDIA GPUs**: Use `backend: nccl` (fastest)
- **CPU only**: Use `backend: gloo`
- **Mixed**: Use `backend: gloo`

## Troubleshooting

### Issue: "Address already in use"

**Solution**: Kill existing processes or change the master port:
```bash
torchrun --nproc_per_node=2 --master_port=29501 main.py +runner=train-bert4rec-distributed
```

### Issue: "NCCL error"

**Solution**: 
1. Check CUDA and NCCL versions are compatible
2. Try using `backend: gloo` instead
3. Set environment variable: `export NCCL_DEBUG=INFO` for detailed logs

### Issue: Different loss values across GPUs

**Solution**: This is normal during training. Metrics are synchronized at the end of each epoch.

### Issue: Out of memory

**Solution**: Reduce per-GPU batch size:
```yaml
dataloader:
  mini_batch_size: 128  # Reduced from 256
  batch_size: 128
```

## Multi-Node Training

To train across multiple machines:

```bash
# On node 0 (master)
torchrun \
  --nproc_per_node=2 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="192.168.1.1" \
  --master_port=29500 \
  main.py +runner=train-bert4rec-distributed

# On node 1
torchrun \
  --nproc_per_node=2 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr="192.168.1.1" \
  --master_port=29500 \
  main.py +runner=train-bert4rec-distributed
```

## Performance Tips

1. **Use NCCL backend** for NVIDIA GPUs
2. **Pin memory** (automatically enabled in distributed mode)
3. **Increase num_workers** to saturate GPU utilization
4. **Use mixed precision training** (future enhancement)
5. **Profile your code** to identify bottlenecks

## Monitoring

During training, only rank 0 will print logs to avoid duplicate output. You can monitor:
- GPU utilization: `nvidia-smi -l 1`
- Training logs: Check the output from rank 0
- Checkpoints: Saved only by rank 0 in the model directory

## Example Output

```
Initialized distributed training: rank=0, world_size=2, local_rank=0
Initialized distributed training: rank=1, world_size=2, local_rank=1
Model Selection: bert4rec
Distributed Training: True
Device: cuda:0
Created DistributedSampler: train_samples=100000, val_samples=10000, samples_per_gpu=50000
Wrapped model with DDP on rank 0
Wrapped model with DDP on rank 1
[TRAIN], Epoch: 0, Step: 100, AVGLoss: 0.5234, ...
```

## Backward Compatibility

All existing single-GPU configurations continue to work without modification:
```bash
python main.py +runner=train-bert4rec  # Still works!
```

The framework automatically detects whether to use distributed or single-GPU mode based on:
1. The `distributed: true` flag in config
2. Whether `torchrun` or `torch.distributed.launch` was used
