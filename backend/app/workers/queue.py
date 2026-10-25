"""A small Postgres-backed job queue (see docs/adr/0001-postgres-job-queue.md).

Jobs are rows in `jobs`. Workers claim the next runnable row with SELECT ... FOR UPDATE SKIP LOCKED,
so several workers never take the same job. SQLite (tests) ignores the locking clause, which is
fine for a single worker.
"""

from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.models import Job
from app.models.enums import JobStatus, JobType


def enqueue(
    db: Session,
    *,
    type: JobType,
    queue: str = "cpu",
    params: dict | None = None,
    project_id: int | None = None,
    priority: int = 0,
    max_attempts: int = 3,
    total_items: int = 0,
    created_by: int | None = None,
    commit: bool = True,
) -> Job:
    job = Job(
        type=type,
        queue=queue,
        params=params or {},
        project_id=project_id,
        priority=priority,
        max_attempts=max_attempts,
        total_items=total_items,
        created_by=created_by,
        run_after=utcnow(),
    )
    db.add(job)
    if commit:
        db.commit()
    else:
        db.flush()
    return job


def claim_next(db: Session, *, queues: list[str], worker_id: str) -> Job | None:
    stmt = (
        select(Job)
        .where(Job.status == JobStatus.queued, Job.queue.in_(queues), Job.run_after <= utcnow())
        .order_by(Job.priority.desc(), Job.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = db.execute(stmt).scalar_one_or_none()
    if job is None:
        db.rollback()
        return None
    now = utcnow()
    job.status = JobStatus.running
    job.locked_by = worker_id
    job.attempts += 1
    job.heartbeat_at = now
    job.started_at = job.started_at or now
    job.error = None
    db.commit()
    return job


def heartbeat(db: Session, job: Job, *, done_items: int | None = None) -> None:
    job.heartbeat_at = utcnow()
    if done_items is not None:
        job.done_items = done_items
        job.progress = done_items / job.total_items if job.total_items else 0.0
    db.commit()


def complete(db: Session, job: Job, result: dict | None = None) -> None:
    job.status = JobStatus.succeeded
    job.result = result or {}
    job.progress = 1.0
    job.finished_at = utcnow()
    job.locked_by = None
    db.commit()


def fail(db: Session, job: Job, error: str, *, backoff_s: float = 30.0) -> None:
    """Retry with exponential backoff until max_attempts, then mark failed."""
    job.error = error[-4000:]
    job.locked_by = None
    if job.attempts < job.max_attempts:
        job.status = JobStatus.queued
        job.run_after = utcnow() + timedelta(seconds=backoff_s * 2 ** (job.attempts - 1))
    else:
        job.status = JobStatus.failed
        job.finished_at = utcnow()
    db.commit()


def cancel(db: Session, job: Job) -> bool:
    """Queued jobs are cancelled immediately; running jobs check `is_cancelled` between items."""
    if job.status not in (JobStatus.queued, JobStatus.running):
        return False
    job.status = JobStatus.cancelled
    job.finished_at = utcnow()
    db.commit()
    return True


def is_cancelled(db: Session, job: Job) -> bool:
    db.refresh(job, attribute_names=["status"])
    return job.status == JobStatus.cancelled


def requeue_stale(db: Session, *, timeout_s: int) -> int:
    """Return jobs whose worker stopped sending heartbeats (crash, reboot) to the queue."""
    cutoff = utcnow() - timedelta(seconds=timeout_s)
    res = db.execute(
        update(Job)
        .where(Job.status == JobStatus.running, Job.heartbeat_at < cutoff)
        .values(status=JobStatus.queued, locked_by=None, error="requeued: worker heartbeat timed out")
    )
    db.commit()
    return res.rowcount or 0
