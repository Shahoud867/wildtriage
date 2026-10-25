"""Ground-truth annotations, evaluation runs, exports and the audit log."""

from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, JSONType, TimestampMixin, fk, pk


class Annotation(TimestampMixin, Base):
    """Imported ground truth (COCO, SA-FARI, LILA), always tied to a frozen dataset version."""

    __tablename__ = "annotations"

    id: Mapped[int] = pk()
    dataset_version_id: Mapped[int] = fk("dataset_versions.id")
    media_id: Mapped[int] = fk("media_items.id")
    frame_index: Mapped[int | None] = mapped_column(nullable=True)
    source: Mapped[str] = mapped_column(String(64))
    x: Mapped[float | None] = mapped_column(Float, nullable=True)
    y: Mapped[float | None] = mapped_column(Float, nullable=True)
    w: Mapped[float | None] = mapped_column(Float, nullable=True)
    h: Mapped[float | None] = mapped_column(Float, nullable=True)
    mask_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    taxon_id: Mapped[int | None] = fk("taxa.id", nullable=True, ondelete="SET NULL")
    track_identity: Mapped[int | None] = mapped_column(nullable=True)


class EvaluationRun(TimestampMixin, Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = pk()
    name: Mapped[str] = mapped_column(String(255))
    task: Mapped[str] = mapped_column(String(64))
    model_version_id: Mapped[int | None] = fk("model_versions.id", nullable=True, ondelete="SET NULL")
    dataset_version_id: Mapped[int | None] = fk("dataset_versions.id", nullable=True, ondelete="SET NULL")
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    artifacts_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Export(TimestampMixin, Base):
    __tablename__ = "exports"

    id: Mapped[int] = pk()
    project_id: Mapped[int] = fk("projects.id")
    format: Mapped[str] = mapped_column(String(32))  # csv | coco | camtrap_dp
    filters: Mapped[dict] = mapped_column(JSONType, default=dict)
    job_id: Mapped[int | None] = fk("jobs.id", nullable=True, ondelete="SET NULL")
    storage_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_by: Mapped[int | None] = fk("users.id", nullable=True, ondelete="SET NULL")


class AuditLog(TimestampMixin, Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = pk()
    actor_id: Mapped[int | None] = fk("users.id", nullable=True, ondelete="SET NULL")
    action: Mapped[str] = mapped_column(String(64))
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[int] = mapped_column(index=True)
    before: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    after: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
