import shutil
from pathlib import Path
from typing import BinaryIO

SCHEME = "local://"


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError(f"storage key escapes the media root: {key!r}")
        return path

    def put_file(self, key: str, src: Path) -> str:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst)
        return SCHEME + key

    def local_path(self, uri: str) -> Path:
        if not uri.startswith(SCHEME):
            raise ValueError(f"not a local storage URI: {uri!r}")
        return self._path(uri.removeprefix(SCHEME))

    def open(self, uri: str) -> BinaryIO:
        return self.local_path(uri).open("rb")

    def delete(self, uri: str) -> None:
        self.local_path(uri).unlink(missing_ok=True)

    def temp_dir(self) -> Path:
        tmp = self.root / "_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp
