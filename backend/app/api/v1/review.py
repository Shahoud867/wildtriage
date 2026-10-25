from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import DB, Reviewer
from app.models import Classification, ReviewTask, TriageDecision, VerificationDecision
from app.models.enums import ReviewAction, ReviewStatus
from app.schemas import ReviewDecisionIn, ReviewDecisionOut, ReviewItemOut
from app.services import audit

router = APIRouter(prefix="/review", tags=["review"])


@router.get("/queue", response_model=list[ReviewItemOut])
def review_queue(project_id: int, db: DB, user: Reviewer, limit: int = 50):
    """Open review tasks, riskiest first."""
    rows = db.execute(
        select(ReviewTask, TriageDecision, Classification)
        .join(TriageDecision, TriageDecision.id == ReviewTask.triage_decision_id)
        .outerjoin(Classification, Classification.detection_id == ReviewTask.detection_id)
        .where(ReviewTask.project_id == project_id, ReviewTask.status == ReviewStatus.open)
        .order_by(ReviewTask.priority.desc(), ReviewTask.id)
        .limit(min(limit, 200))
    ).all()
    return [
        ReviewItemOut(task_id=t.id, media_id=t.media_id, detection_id=t.detection_id, priority=t.priority,
                      decision=td.decision, reasons=td.reasons or [], label=c.label if c else None,
                      calibrated_score=c.calibrated_score if c else None, topk=c.topk if c else [])
        for t, td, c in rows
    ]


@router.post("/tasks/{task_id}/decision", response_model=ReviewDecisionOut, status_code=201)
def decide(task_id: int, body: ReviewDecisionIn, db: DB, user: Reviewer):
    task = db.get(ReviewTask, task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "review task not found")
    if body.action == ReviewAction.relabel and not body.new_label:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "relabel needs new_label")
    # Append-only: a second decision on the same task is a correction, recorded as a new row.
    decision = VerificationDecision(review_task_id=task.id, reviewer_id=user.id, **body.model_dump())
    db.add(decision)
    before = {"status": task.status.value}
    task.status = ReviewStatus.done if body.action != ReviewAction.unsure else ReviewStatus.skipped
    db.flush()
    audit.record(db, actor_id=user.id, action=f"review.{body.action.value}", entity_type="review_task",
                 entity_id=task.id, before=before, after={"status": task.status.value, **body.model_dump()})
    db.commit()
    return decision
