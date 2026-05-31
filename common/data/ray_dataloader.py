"""Helpers to read parquet files using Ray Dataset with a pandas-friendly interface.

Ray is optional. If Ray is unavailable, functions raise ImportError so callers can
fallback to pandas.
"""
from typing import Iterable


def ensure_ray_initialized() -> None:
    try:
        import ray
    except Exception as e:
        raise ImportError("ray is required for ray_dataloader but is not installed") from e

    if not ray.is_initialized():
        # Lightweight init
        ray.init(ignore_reinit_error=True, include_dashboard=False)


def read_parquet_to_pandas(paths: Iterable[str]):
    """Read one or more parquet files (local or cloud URIs) using Ray and return a pandas.DataFrame.

    This uses ray.data.read_parquet which can efficiently read many files in parallel and support s3/gs schemes.
    """
    try:
        import ray
    except Exception as e:
        raise ImportError("ray is required for read_parquet_to_pandas but is not installed") from e

    ensure_ray_initialized()
    ds = ray.data.read_parquet(paths)
    return ds.to_pandas()


def iter_parquet_batches(paths: Iterable[str], batch_size: int = 1024, batch_format: str = 'pandas'):
    """Yield batches from parquet files using Ray Dataset.iter_batches.

    Yields pandas.DataFrame batches when batch_format=='pandas'.
    """
    try:
        import ray
    except Exception as e:
        raise ImportError("ray is required for iter_parquet_batches but is not installed") from e

    ensure_ray_initialized()
    ds = ray.data.read_parquet(paths)
    for batch in ds.iter_batches(batch_size=batch_size, batch_format=batch_format):
        yield batch
