from functools import lru_cache

from app.core.config import get_settings
from app.storage.base import StorageBackend
from app.storage.local import LocalStorage


@lru_cache
def get_storage() -> StorageBackend:
    # P1: return an S3-compatible backend when WT_STORAGE=s3.
    return LocalStorage(get_settings().media_root)


__all__ = ["LocalStorage", "StorageBackend", "get_storage"]
