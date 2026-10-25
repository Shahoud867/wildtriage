"""Worker process: claims jobs from the given queues and runs them.

    python -m app.workers.runner --queues cpu                  # any machine
    python -m app.workers.runner --queues gpu                  # T1000 (WT_GPU_VRAM_BUDGET_GB=3.5)
    python -m app.workers.runner --queues gpu,gpu-heavy        # lab PC (WT_GPU_VRAM_BUDGET_GB=18)

One long-lived process per GPU: models load once and CUDA is never forked.
"""

import argparse
import os
import socket
import time
import traceback

import structlog

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import session_factory
from app.models import Job
from app.models.enums import JobStatus
from app.workers import queue
from app.workers.tasks import TASKS, load_all

log = structlog.get_logger()


def run_one(queues: list[str], worker_id: str) -> bool:
    """Claim and run a single job. Returns False when there was nothing to do."""
    with session_factory()() as db:
        job = queue.claim_next(db, queues=queues, worker_id=worker_id)
        if job is None:
            return False
        fn = TASKS.get(job.type)
        log.info("job.start", job_id=job.id, type=job.type, attempt=job.attempts)
        if fn is None:
            queue.fail(db, job, f"no task registered for job type {job.type!r}")
            return True
        try:
            result = fn(db, job) or {}
        except Exception:
            db.rollback()
            job = db.get(Job, job.id)
            queue.fail(db, job, traceback.format_exc())
            log.exception("job.failed", job_id=job.id)
            return True
        db.refresh(job)
        if job.status != JobStatus.cancelled:
            queue.complete(db, job, result)
        log.info("job.done", job_id=job.id, status=job.status, result=result)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queues", default="cpu", help="comma-separated: cpu, gpu, gpu-heavy")
    parser.add_argument("--once", action="store_true", help="run at most one job, then exit")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.env)
    load_all()
    queues = [q.strip() for q in args.queues.split(",") if q.strip()]
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    log.info("worker.start", worker_id=worker_id, queues=queues, model_backend=settings.model_backend,
             vram_budget_gb=settings.gpu_vram_budget_gb)

    last_sweep = 0.0
    while True:
        if time.monotonic() - last_sweep > 60:
            with session_factory()() as db:
                if n := queue.requeue_stale(db, timeout_s=settings.worker_heartbeat_timeout_s):
                    log.warning("worker.requeued_stale_jobs", count=n)
            last_sweep = time.monotonic()
        did_work = run_one(queues, worker_id)
        if args.once:
            break
        if not did_work:
            time.sleep(settings.worker_poll_interval_s)


if __name__ == "__main__":
    main()
