import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Iterable, List

from common.pipeline_config import PipelineConfig

from common.data import ray_dataloader


def is_distributed():
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()


def get_rank():
    import torch.distributed as dist
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    import torch.distributed as dist
    if is_distributed():
        return dist.get_world_size()
    return 1


class RayDataGenerator(IterableDataset):
    """IterableDataset that reads parquet files lazily using Ray.

    It partitions the provided files across all global workers (distributed ranks * dataloader workers)
    so the same file is not processed by multiple workers.
    """
    def __init__(self, kind: str, pipeline_cfg: PipelineConfig, batch_size: int = None) -> None:
        super().__init__()
        paths = pipeline_cfg.dataloader.get_file_paths(path=pipeline_cfg.data.base_path, file_format=pipeline_cfg.data.file_format)
        self._model_config = pipeline_cfg.model

        # Distributed training info
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_distributed = is_distributed()

        paths = list(filter(lambda x: kind in x, paths))
        assert len(paths) != 0, "File not found for kind: {}".format(kind)

        self.paths: List[str] = paths
        print(self.paths)
        self.mini_batch_size = batch_size if batch_size is not None else min(pipeline_cfg.dataloader.mini_batch_size, 1024)

        base_total_steps = getattr(pipeline_cfg.dataloader, f'total_{kind}_steps')

        # Number of dataloader worker processes per training process
        # Note: get_worker_info will be available only inside worker process; default to 1 here
        self.worker_count_per_process = max(1, getattr(pipeline_cfg.dataloader, 'num_workers', 0))

        # total steps per global worker will be computed in __iter__ when worker info known
        self.base_total_steps = base_total_steps

        self.preprocess_fn = pipeline_cfg.model.preprocessing_fn

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # total global workers across all distributed ranks
        total_workers = self.world_size * num_workers
        global_worker_id = self.rank * num_workers + worker_id

        # Partition files by global_worker_id so each global worker gets a disjoint subset
        assigned_files = [p for i, p in enumerate(self.paths) if (i % total_workers) == global_worker_id]
        print("DEBUG", assigned_files)
        # Compute total steps for this global worker
        if total_workers > 0:
            if self.is_distributed:
                total_steps = max(0, self.base_total_steps // total_workers)
            else:
                total_steps = max(0, self.base_total_steps // num_workers)
        else:
            total_steps = self.base_total_steps

        current_step = 0

        # If no files assigned, return an empty iterator
        if len(assigned_files) == 0:
            return iter(())

        # Prefer reading assigned files as a single Ray Dataset for efficiency.
        # If Ray is unavailable or fails, fall back to pandas-based per-file batching.
        try:
            # Try using Ray to read all assigned files at once
            ds_iter = ray_dataloader.iter_parquet_batches(assigned_files, batch_size=self.mini_batch_size, batch_format='pandas')
            for batch in ds_iter:
                if current_step >= total_steps:
                    break
                if isinstance(batch, pd.DataFrame):
                    df_batch = batch
                else:
                    df_batch = pd.DataFrame(batch)

                # Convert to platform types if configured
                try:
                    df_batch = self._model_config.features.convert_to_platform_type(df_batch)
                except Exception:
                    pass

                current_step += 1
                yield self.preprocess_fn(df_batch)
        except ImportError as ie:
            # Ray not installed — fallback to pandas
            print(f"[WARN] Ray not available, falling back to pandas for assigned files: {ie}")
            for file_path in assigned_files:
                if current_step >= total_steps:
                    break
                try:
                    df = pd.read_parquet(file_path)
                except Exception as e:
                    print(f"[WARN] pandas failed to read {file_path}: {e}")
                    continue

                # Yield batches from this file
                for i in range(0, df.shape[0], self.mini_batch_size):
                    if current_step >= total_steps:
                        break
                    df_batch = df.iloc[i:i + self.mini_batch_size]
                    try:
                        df_batch = self._model_config.features.convert_to_platform_type(df_batch)
                    except Exception:
                        pass

                    current_step += 1
                    yield self.preprocess_fn(df_batch)
        except Exception as e:
            # Ray failed at runtime for other reasons; surface warning and try pandas fallback
            print(f"[WARN] RayDataGenerator encountered an error reading with Ray, falling back to pandas: {e}")
            for file_path in assigned_files:
                if current_step >= total_steps:
                    break
                try:
                    df = pd.read_parquet(file_path)
                except Exception as e:
                    print(f"[WARN] pandas failed to read {file_path}: {e}")
                    continue

                for i in range(0, df.shape[0], self.mini_batch_size):
                    if current_step >= total_steps:
                        break
                    df_batch = df.iloc[i:i + self.mini_batch_size]
                    try:
                        df_batch = self._model_config.features.convert_to_platform_type(df_batch)
                    except Exception:
                        pass

                    current_step += 1
                    yield self.preprocess_fn(df_batch)

    def __len__(self):
        # Best-effort: return base_total_steps divided among global workers assuming single-worker per process
        return max(0, self.base_total_steps)
