from datetime import timedelta

from sqlalchemy import func, select

from app.db.base import utcnow
from app.models import Job, MediaItem, TriageDecision
from app.models.enums import JobStatus, JobType
from app.workers import queue
from app.workers.runner import run_one
from app.workers.tasks import load_all
from conftest import create_project_and_dataset, make_png

load_all()


def test_claim_respects_queues_and_priority(db):
    queue.enqueue(db, type=JobType.ingest, queue="gpu", params={})
    low = queue.enqueue(db, type=JobType.ingest, queue="cpu", params={}, priority=0)
    high = queue.enqueue(db, type=JobType.ingest, queue="cpu", params={}, priority=5)
    assert queue.claim_next(db, queues=["cpu"], worker_id="w").id == high.id
    assert queue.claim_next(db, queues=["cpu"], worker_id="w").id == low.id
    assert queue.claim_next(db, queues=["cpu"], worker_id="w") is None


def test_failed_job_is_retried_then_marked_failed(db):
    job = queue.enqueue(db, type=JobType.export, queue="cpu", params={}, max_attempts=2)  # no export task yet
    assert run_one(["cpu"], "w")
    db.refresh(job)
    assert job.status == JobStatus.queued and job.attempts == 1 and "no task registered" in job.error
    job.run_after = utcnow() - timedelta(seconds=1)  # skip the backoff
    db.commit()
    assert run_one(["cpu"], "w")
    db.refresh(job)
    assert job.status == JobStatus.failed and job.attempts == 2


def test_stale_running_jobs_are_requeued(db):
    job = queue.enqueue(db, type=JobType.ingest, queue="cpu", params={})
    queue.claim_next(db, queues=["cpu"], worker_id="crashed")
    job.heartbeat_at = utcnow() - timedelta(hours=1)
    db.commit()
    assert queue.requeue_stale(db, timeout_s=300) == 1
    db.refresh(job)
    assert job.status == JobStatus.queued and job.locked_by is None


def test_upload_to_triage_end_to_end_with_stub_models(admin_client, db):
    project_id, dataset_id = create_project_and_dataset(admin_client)
    files = [("files", (f"img{i}.png", make_png(rgb=(i * 40, 90, 200 - i * 30)), "image/png")) for i in range(5)]
    upload = admin_client.post(f"/api/v1/datasets/{dataset_id}/uploads", files=files).json()
    assert all(r["ok"] for r in upload["results"])

    assert run_one(["cpu"], "w-cpu")  # ingest -> enqueues inference on the "gpu" queue
    inference = db.execute(select(Job).where(Job.type == JobType.inference)).scalar_one()
    assert inference.queue == "gpu"
    assert run_one(["gpu"], "w-gpu")
    db.refresh(inference)
    assert inference.status == JobStatus.succeeded and inference.done_items == 5

    statuses = db.execute(select(MediaItem.status)).scalars().all()
    assert set(statuses) == {"processed"}
    # Every image gets at least one triage decision (per detection, or an empty/missed-animal decision).
    n_decisions = db.execute(select(func.count()).select_from(TriageDecision)).scalar_one()
    assert n_decisions >= 5

    summary = admin_client.get(f"/api/v1/analytics/{project_id}/summary").json()
    assert summary["media_by_status"] == {"processed": 5}
    assert sum(summary["triage"].values()) == n_decisions

    items = admin_client.get("/api/v1/review/queue", params={"project_id": project_id}).json()
    assert items == sorted(items, key=lambda i: -i["priority"])
    if items:
        first = items[0]
        r = admin_client.post(f"/api/v1/review/tasks/{first['task_id']}/decision",
                              json={"action": "relabel", "new_label": "Puma concolor", "time_spent_ms": 2300})
        assert r.status_code == 201
        after = admin_client.get("/api/v1/review/queue", params={"project_id": project_id}).json()
        assert first["task_id"] not in [i["task_id"] for i in after]
        if first["detection_id"] is not None:
            accepted = admin_client.get(f"/api/v1/analytics/{project_id}/summary").json()["accepted_species_counts"]
            assert accepted.get("Puma concolor", 0) >= 1
