# Distributed Multi-GPU Training Guide (Accelerate)

This guide explains how to use distributed multi-GPU training in the recommendation framework using **Hugging Face Accelerate**.

## Overview

The framework uses [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index) to handle distributed training, mixed precision, and device placement. This simplifies the workflow and supports:
- Multi-GPU training (DDP)
- Mixed precision (fp16, bf16)
- Automatic device handling
- Easy scaling from single GPU to multi-node

## Quick Start

### 1. Configuration

To run in distributed mode, set `distributed: true` in your runner config (though Accelerate handles single process too, this flag ensures the correct strategy is picked).

```yaml
# hydra-config/runner/train-bert4rec-distributed.yaml
...
distributed: true
...
```

### 2. Launching Training

Use `accelerate launch` to run the training script. This replaces `torchrun`.

**Single Machine, Multiple GPUs:**

```bash
accelerate launch --num_processes=2 main.py +runner=train-bert4rec-distributed
```
(Replace `2` with the number of GPUs).

**Mixed Precision:**

```bash
accelerate launch --num_processes=2 --mixed_precision=fp16 main.py +runner=train-bert4rec-distributed
```

**Single GPU (Backward Compatible):**

You can still run with python directly, but `accelerate launch` is recommended even for single GPU to benefit from mixed precision.

```bash
# Standard python run
python main.py +runner=train-bert4rec

# Or with accelerate (recommended)
accelerate launch --num_processes=1 main.py +runner=train-bert4rec
```

## Configuration Details

The framework automatically detects the distributed environment via `Accelerator`.

### Key Configs
- **`distributed: true`**: Activates `AccelerateTrainingStrategy`.
- **`dataloader.num_workers`**: Number of workers per GPU. Accelerate/PyTorch handles dividing the global batch size if configured, but currently the framework uses per-worker batch size concepts.
    - **Note**: The `batch_size` in config is currently treated as the batch size **per process** (or handled by the DataGenerator partitioning).

## Distributed Data Loading

Data loading relies on `SimpleDataGenerator` which partitions the dataset based on the process rank.
- **Rank 0** gets the first chunk.
- **Rank 1** gets the second chunk, etc.

Since `IterableDataset` is used, shuffling is limited to the order in the parquet files unless pre-shuffled.

## Troubleshooting

### "Address already in use"
If you restart training quickly, the port might be occupied.
```bash
accelerate launch --main_process_port 29501 ...
```

### Monitoring
Logs are printed only on the main process (Rank 0).
Watch GPU usage with `nvidia-smi`.
