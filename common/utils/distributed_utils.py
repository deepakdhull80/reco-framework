"""
Distributed training utilities for multi-GPU support using PyTorch DDP.
"""
import os
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def setup_distributed() -> None:
    """
    Initialize the distributed process group.
    
    This should be called at the beginning of the training script when running
    in distributed mode. It reads environment variables set by torchrun/torch.distributed.launch.
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training requires torch.distributed to be available")
    
    # torchrun sets these environment variables
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        logger.warning("Distributed environment variables not found. Running in single-GPU mode.")
        return
    
    # Initialize the process group
    dist.init_process_group(backend=get_backend())
    
    # Set the device for this process
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    
    logger.info(
        f"Initialized distributed training: rank={get_rank()}, "
        f"world_size={get_world_size()}, local_rank={local_rank}"
    )


def cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    if is_distributed():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed process group")


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get the local rank of the current process (rank within the node)."""
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    return get_rank() == 0


def get_backend() -> str:
    """Get the distributed backend to use."""
    backend = os.environ.get('DIST_BACKEND', 'nccl')
    if backend == 'nccl' and not torch.cuda.is_available():
        logger.warning("NCCL backend requires CUDA. Falling back to gloo.")
        backend = 'gloo'
    return backend


def barrier() -> None:
    """Synchronization barrier across all processes."""
    if is_distributed():
        dist.barrier()


def reduce_dict(input_dict: Dict[str, Any], average: bool = True) -> Dict[str, Any]:
    """
    Reduce a dictionary of tensors across all processes.
    
    Args:
        input_dict: Dictionary with string keys and tensor values
        average: If True, average the values. If False, sum them.
    
    Returns:
        Dictionary with reduced values (only meaningful on rank 0)
    """
    if not is_distributed():
        return input_dict
    
    world_size = get_world_size()
    
    # Convert all values to tensors if they aren't already
    tensor_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            tensor_dict[k] = v.detach().clone()
        else:
            tensor_dict[k] = torch.tensor(v, dtype=torch.float32)
    
    # Move tensors to the current device
    device = torch.device(f'cuda:{get_local_rank()}' if torch.cuda.is_available() else 'cpu')
    for k in tensor_dict:
        tensor_dict[k] = tensor_dict[k].to(device)
    
    # Reduce all tensors
    for k in tensor_dict:
        dist.all_reduce(tensor_dict[k], op=dist.ReduceOp.SUM)
        if average:
            tensor_dict[k] /= world_size
    
    # Convert back to Python scalars
    reduced_dict = {}
    for k, v in tensor_dict.items():
        reduced_dict[k] = v.item()
    
    return reduced_dict


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all processes.
    
    Args:
        tensor: Input tensor
        average: If True, average the tensor. If False, sum it.
    
    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if average:
        rt /= get_world_size()
    
    return rt


def gather_object(obj: Any) -> list:
    """
    Gather objects from all processes to rank 0.
    
    Args:
        obj: Object to gather (must be picklable)
    
    Returns:
        List of objects from all processes (only on rank 0, None on other ranks)
    """
    if not is_distributed():
        return [obj]
    
    world_size = get_world_size()
    
    if is_main_process():
        gather_list = [None] * world_size
        dist.gather_object(obj, gather_list, dst=0)
        return gather_list
    else:
        dist.gather_object(obj, dst=0)
        return None


def print_once(*args, **kwargs):
    """Print only on the main process."""
    if is_main_process():
        print(*args, **kwargs)


def log_once(logger_obj: logging.Logger, level: int, msg: str, *args, **kwargs):
    """Log only on the main process."""
    if is_main_process():
        logger_obj.log(level, msg, *args, **kwargs)
