"""Datasets, frozen dataset versions, media items and sampled frames."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, JSONType, TimestampMixin, enum_col, fk, pk
from app.models.enums import MediaKind, MediaStatus, Modality


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"

    id: Mapped[int] = pk()
    project_id: Mapped[int] = fk("projects.id")
    name: Mapped[str] = mapped_column(String(255))
    modality: Mapped[Modality] = mapped_column(enum_col(Modality), default=Modality.camera_trap)
    source: Mapped[str] = mapped_column(String(255), default="upload")
    license: Mapped[str] = mapped_column(String(64), default="unknown")
    created_by: Mapped[int | None] = fk("users.id", nullable=True, ondelete="SET NULL")


class DatasetVersion(Base):
    """An immutable snapshot: a manifest of files (with hashes) plus the split specification."""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)

    id: Mapped[int] = pk()
    dataset_id: Mapped[int] = fk("datasets.id")
    version: Mapped[int]
    manifest_uri: Mapped[str] = mapped_column(String(1024))
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    split_spec: Mapped[dict] = mapped_column(JSONType, default=dict)
    frozen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MediaItem(TimestampMixin, Base):
    __tablename__ = "media_items"
    __table_args__ = (UniqueConstraint("dataset_id", "sha256"),)

    id: Mapped[int] = pk()
    dataset_id: Mapped[int] = fk("datasets.id")
    deployment_id: Mapped[int | None] = fk("deployments.id", nullable=True, ondelete="SET NULL")
    kind: Mapped[MediaKind] = mapped_column(enum_col(MediaKind))
    original_filename: Mapped[str] = mapped_column(String(1024))
    storage_uri: Mapped[str] = mapped_column(String(1024))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    width: Mapped[int | None] = mapped_column(nullable=True)
    height: Mapped[int | None] = mapped_column(nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    is_night: Mapped[bool | None] = mapped_column(nullable=True)
    # brightness, blur (Laplacian variance), chroma, per-clip MaxN, ...
    quality: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[MediaStatus] = mapped_column(enum_col(MediaStatus), default=MediaStatus.uploaded, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Frame(Base):
    """Only frames we actually sampled from a video (e.g. 1 fps screening, 6 fps tracking)."""

    __tablename__ = "frames"
    __table_args__ = (UniqueConstraint("media_id", "frame_index"),)

    id: Mapped[int] = pk()
    media_id: Mapped[int] = fk("media_items.id")
    frame_index: Mapped[int]
    timestamp_ms: Mapped[int]
    storage_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
