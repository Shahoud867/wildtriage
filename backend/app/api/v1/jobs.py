import asyncio
import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import DB, CurrentUser, Reviewer
from app.db.session import session_factory
from app.models import Job
from app.models.enums import JobStatus
from app.schemas import JobIn, JobOut
from app.workers import queue

router = APIRouter(prefix="/jobs", tags=["jobs"])

QUEUES = {"cpu", "gpu", "gpu-heavy"}
TERMINAL = {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled}


def _job(db, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job


@router.get("", response_model=list[JobOut])
def list_jobs(db: DB, user: CurrentUser, project_id: int | None = None, limit: int = 50):
    q = select(Job).order_by(Job.id.desc()).limit(min(limit, 200))
    if project_id is not None:
        q = q.where(Job.project_id == project_id)
    return db.execute(q).scalars().all()


@router.post("", response_model=JobOut, status_code=201)
def create_job(body: JobIn, db: DB, user: Reviewer):
    if body.queue not in QUEUES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"queue must be one of {sorted(QUEUES)}")
    return queue.enqueue(db, type=body.type, queue=body.queue, params=body.params, project_id=body.project_id,
                         priority=body.priority, total_items=len(body.params.get("media_ids", [])),
                         created_by=user.id)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: DB, user: CurrentUser):
    return _job(db, job_id)


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: int, db: DB, user: Reviewer):
    job = _job(db, job_id)
    if not queue.cancel(db, job):
        raise HTTPException(status.HTTP_409_CONFLICT, f"job is already {job.status}")
    return job


@router.get("/{job_id}/events")
async def job_events(job_id: int, user: CurrentUser):
    """Server-sent events with job progress until the job finishes. The UI falls back to polling GET /jobs/{id}."""

    async def stream():
        while True:
            with session_factory()() as db:
                job = db.get(Job, job_id)
                if job is None:
                    yield "event: error\ndata: {\"detail\": \"job not found\"}\n\n"
                    return
                payload = JobOut.model_validate(job).model_dump(mode="json")
            yield f"data: {json.dumps(payload)}\n\n"
            if payload["status"] in TERMINAL:
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream")
