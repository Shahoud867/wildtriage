from pathlib import Path
from typing import BinaryIO, Protocol


class StorageBackend(Protocol):
    """Where media, masks and exports live. Metadata stays in Postgres; only URIs are stored there."""

    def put_file(self, key: str, src: Path) -> str:
        """Move a finished temp file into storage under `key`; returns the storage URI."""
        ...

    def open(self, uri: str) -> BinaryIO: ...

    def local_path(self, uri: str) -> Path:
        """A filesystem path for tools that need one (ffmpeg, model loaders)."""
        ...

    def delete(self, uri: str) -> None: ...

    def temp_dir(self) -> Path: ...
