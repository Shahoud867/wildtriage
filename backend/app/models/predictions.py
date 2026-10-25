"""Model outputs. Masks live in object storage (RLE files); boxes and scores live here."""

from sqlalchemy import CheckConstraint, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, JSONType, TimestampMixin, enum_col, fk, pk
from app.models.enums import DetectionCategory


class Track(TimestampMixin, Base):
    __tablename__ = "tracks"

    id: Mapped[int] = pk()
    media_id: Mapped[int] = fk("media_items.id")
    model_version_id: Mapped[int] = fk("model_versions.id", ondelete="RESTRICT")
    job_id: Mapped[int | None] = fk("jobs.id", nullable=True, ondelete="SET NULL")
    track_key: Mapped[int]  # identity within the clip, as produced by the tracker
    start_frame: Mapped[int]
    end_frame: Mapped[int]
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    taxon_id: Mapped[int | None] = fk("taxa.id", nullable=True, ondelete="SET NULL")
    score: Mapped[float] = mapped_column(Float, default=0.0)


class Detection(TimestampMixin, Base):
    __tablename__ = "detections"

    id: Mapped[int] = pk()
    media_id: Mapped[int] = fk("media_items.id")
    frame_id: Mapped[int | None] = fk("frames.id", nullable=True)
    track_id: Mapped[int | None] = fk("tracks.id", nullable=True)
    model_version_id: Mapped[int] = fk("model_versions.id", ondelete="RESTRICT")
    job_id: Mapped[int | None] = fk("jobs.id", nullable=True, ondelete="SET NULL")
    category: Mapped[DetectionCategory] = mapped_column(enum_col(DetectionCategory))
    # Normalised [0, 1] box: top-left x, y, width, height (MegaDetector convention).
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    mask_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)


class Classification(TimestampMixin, Base):
    __tablename__ = "classifications"
    __table_args__ = (
        CheckConstraint("detection_id IS NOT NULL OR track_id IS NOT NULL", name="ck_classification_target"),
    )

    id: Mapped[int] = pk()
    detection_id: Mapped[int | None] = fk("detections.id", nullable=True)
    track_id: Mapped[int | None] = fk("tracks.id", nullable=True)
    model_version_id: Mapped[int] = fk("model_versions.id", ondelete="RESTRICT")
    # [{"label": "Panthera onca", "p": 0.81}, ...] with calibrated probabilities.
    topk: Mapped[list] = mapped_column(JSONType, default=list)
    label: Mapped[str] = mapped_column(String(255), index=True)
    taxon_id: Mapped[int | None] = fk("taxa.id", nullable=True, ondelete="SET NULL")
    raw_score: Mapped[float] = mapped_column(Float)
    calibrated_score: Mapped[float] = mapped_column(Float)
