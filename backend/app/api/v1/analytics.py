"""Project summaries. W4 adds diel activity; P1 adds independent events and RAI (wildtriage_ml.pipeline.counting)."""

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.deps import DB, CurrentUser
from app.models import Classification, Dataset, Detection, MediaItem, ReviewTask, TriageDecision, VerificationDecision
from app.models.enums import ReviewAction, ReviewStatus, TriageOutcome
from app.schemas import SummaryOut

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/{project_id}/summary", response_model=SummaryOut)
def summary(project_id: int, db: DB, user: CurrentUser):
    project_media = (
        select(MediaItem.id).join(Dataset, Dataset.id == MediaItem.dataset_id).where(Dataset.project_id == project_id)
    )

    by_status = dict(db.execute(
        select(MediaItem.status, func.count()).where(MediaItem.id.in_(project_media)).group_by(MediaItem.status)
    ).all())
    triage = dict(db.execute(
        select(TriageDecision.decision, func.count())
        .where(TriageDecision.media_id.in_(project_media)).group_by(TriageDecision.decision)
    ).all())
    open_reviews = db.execute(
        select(func.count()).select_from(ReviewTask)
        .where(ReviewTask.project_id == project_id, ReviewTask.status == ReviewStatus.open)
    ).scalar_one()
    # Predicted counts include auto-accepted and not-yet-reviewed detections; the UI labels them as such.
    species = dict(db.execute(
        select(Classification.label, func.count())
        .join(Detection, Detection.id == Classification.detection_id)
        .where(Detection.media_id.in_(project_media)).group_by(Classification.label)
    ).all())
    # Accepted = auto-accepted detections + the latest human decision per review task
    # (confirm keeps the predicted label, relabel uses the new one; empty/reject/unsure count nothing).
    accepted: dict[str, int] = {}
    for label in db.execute(
        select(Classification.label)
        .join(TriageDecision, TriageDecision.detection_id == Classification.detection_id)
        .where(TriageDecision.decision == TriageOutcome.auto_accept, TriageDecision.media_id.in_(project_media))
    ).scalars():
        accepted[label] = accepted.get(label, 0) + 1
    latest: dict[int, str | None] = {}
    for task_id, action, new_label, label in db.execute(
        select(ReviewTask.id, VerificationDecision.action, VerificationDecision.new_label, Classification.label)
        .join(ReviewTask, ReviewTask.id == VerificationDecision.review_task_id)
        .outerjoin(Classification, Classification.detection_id == ReviewTask.detection_id)
        .where(ReviewTask.project_id == project_id)
        .order_by(VerificationDecision.id)
    ):
        latest[task_id] = {ReviewAction.relabel: new_label, ReviewAction.confirm: label}.get(action)
    for final in latest.values():
        if final:
            accepted[final] = accepted.get(final, 0) + 1
    return SummaryOut(
        media_by_status={str(k): v for k, v in by_status.items()},
        triage={str(k): v for k, v in triage.items()},
        open_reviews=open_reviews,
        species_counts=species,
        accepted_species_counts=accepted,
    )
