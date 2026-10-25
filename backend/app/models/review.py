"""Triage decisions, the review queue and append-only verification decisions."""

from sqlalchemy import Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, JSONType, TimestampMixin, enum_col, fk, pk
from app.models.enums import ReviewAction, ReviewStatus, TriageOutcome


class TriageDecision(TimestampMixin, Base):
    __tablename__ = "triage_decisions"

    id: Mapped[int] = pk()
    media_id: Mapped[int] = fk("media_items.id")
    detection_id: Mapped[int | None] = fk("detections.id", nullable=True)
    policy_version_id: Mapped[int] = fk("model_versions.id", ondelete="RESTRICT")
    decision: Mapped[TriageOutcome] = mapped_column(enum_col(TriageOutcome))
    risk: Mapped[float] = mapped_column(Float)
    # Reason codes shown to reviewers: low_confidence, night, tiny_object, disagreement, camouflage, audit_sample
    reasons: Mapped[list] = mapped_column(JSONType, default=list)
    features: Mapped[dict] = mapped_column(JSONType, default=dict)


class ReviewTask(TimestampMixin, Base):
    __tablename__ = "review_tasks"

    id: Mapped[int] = pk()
    project_id: Mapped[int] = fk("projects.id")
    triage_decision_id: Mapped[int] = fk("triage_decisions.id")
    media_id: Mapped[int] = fk("media_items.id")
    detection_id: Mapped[int | None] = fk("detections.id", nullable=True)
    priority: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    status: Mapped[ReviewStatus] = mapped_column(enum_col(ReviewStatus), default=ReviewStatus.open, index=True)
    assigned_to: Mapped[int | None] = fk("users.id", nullable=True, ondelete="SET NULL")


class VerificationDecision(TimestampMixin, Base):
    """Never updated or deleted; corrections are new rows. Mirrored into audit_log."""

    __tablename__ = "verification_decisions"

    id: Mapped[int] = pk()
    review_task_id: Mapped[int] = fk("review_tasks.id")
    reviewer_id: Mapped[int] = fk("users.id", ondelete="RESTRICT")
    action: Mapped[ReviewAction] = mapped_column(enum_col(ReviewAction))
    new_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    new_taxon_id: Mapped[int | None] = fk("taxa.id", nullable=True, ondelete="SET NULL")
    new_bbox: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    comment: Mapped[str] = mapped_column(Text, default="")
    time_spent_ms: Mapped[int] = mapped_column(default=0)
