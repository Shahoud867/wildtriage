"""Upload validation: content sniffing (never trust extensions), size limits and SHA-256 hashing."""

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.models.enums import MediaKind, Modality

CHUNK = 1024 * 1024


class UploadError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class SniffResult:
    container: str  # jpeg | png | tiff | mp4 | avi
    extension: str
    is_video: bool


def sniff(head: bytes) -> SniffResult | None:
    """Identify supported formats from their magic bytes."""
    if head.startswith(b"\xff\xd8\xff"):
        return SniffResult("jpeg", ".jpg", False)
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return SniffResult("png", ".png", False)
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return SniffResult("tiff", ".tif", False)
    if len(head) >= 12 and head[4:8] == b"ftyp":
        # ISO base media: MP4 and QuickTime MOV share this box layout.
        return SniffResult("mp4", ".mov" if head[8:10] == b"qt" else ".mp4", True)
    if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        return SniffResult("avi", ".avi", True)
    return None


def media_kind(result: SniffResult, modality: Modality) -> MediaKind:
    if result.is_video:
        if modality == Modality.aerial:
            # Aerial video tracking is out of scope (P2); reject explicitly rather than mis-process it.
            raise UploadError("UNSUPPORTED_MODALITY", "aerial video is not supported yet")
        return MediaKind.video
    return MediaKind.aerial_image if modality == Modality.aerial else MediaKind.image


@dataclass
class SpooledUpload:
    path: Path
    sha256: str
    size_bytes: int
    sniffed: SniffResult


def spool_and_validate(stream: BinaryIO, tmp_dir: Path, max_bytes: int) -> SpooledUpload:
    """Copy an upload to a temp file while hashing it; reject unknown formats and oversize files."""
    digest = hashlib.sha256()
    size = 0
    sniffed: SniffResult | None = None
    fd, name = tempfile.mkstemp(dir=tmp_dir)
    path = Path(name)
    try:
        with open(fd, "wb") as out:
            while chunk := stream.read(CHUNK):
                if sniffed is None:
                    sniffed = sniff(chunk[:16])
                    if sniffed is None:
                        raise UploadError("UNSUPPORTED_TYPE", "file content is not a supported image or video")
                size += len(chunk)
                if size > max_bytes:
                    raise UploadError("TOO_LARGE", f"file exceeds {max_bytes // (1024 * 1024)} MB")
                digest.update(chunk)
                out.write(chunk)
        if sniffed is None:
            raise UploadError("EMPTY_FILE", "file is empty")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return SpooledUpload(path=path, sha256=digest.hexdigest(), size_bytes=size, sniffed=sniffed)
