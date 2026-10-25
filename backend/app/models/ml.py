"""Model versions and the jobs table (which is also the work queue)."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, JSONType, TimestampMixin, enum_col, fk, pk, utcnow
from app.models.enums import JobStatus, JobType, ModelTask


class ModelVersion(TimestampMixin, Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = pk()
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(64))
    task: Mapped[ModelTask] = mapped_column(enum_col(ModelTask))
    framework: Mapped[str] = mapped_column(String(64), default="pytorch")
    weights_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    weights_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    license: Mapped[str] = mapped_column(String(64), default="unknown")
    commercial_ok: Mapped[bool] = mapped_column(default=False)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    # e.g. a site-adapted head points at the zero-shot model it was derived from.
    parent_id: Mapped[int | None] = fk("model_versions.id", nullable=True, ondelete="SET NULL")
    is_active: Mapped[bool] = mapped_column(default=False)


class Job(TimestampMixin, Base):
    """A unit of background work. Workers claim rows with SELECT ... FOR UPDATE SKIP LOCKED."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_claim", "queue", "status", "priority", "run_after"),)

    id: Mapped[int] = pk()
    project_id: Mapped[int | None] = fk("projects.id", nullable=True)
    type: Mapped[JobType] = mapped_column(enum_col(JobType))
    queue: Mapped[str] = mapped_column(String(32), default="cpu")
    priority: Mapped[int] = mapped_column(default=0)
    status: Mapped[JobStatus] = mapped_column(enum_col(JobStatus), default=JobStatus.queued)
    attempts: Mapped[int] = mapped_column(default=0)
    max_attempts: Mapped[int] = mapped_column(default=3)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    total_items: Mapped[int] = mapped_column(default=0)
    done_items: Mapped[int] = mapped_column(default=0)
    params: Mapped[dict] = mapped_column(JSONType, default=dict)
    result: Mapped[dict] = mapped_column(JSONType, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[int | None] = fk("users.id", nullable=True, ondelete="SET NULL")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
