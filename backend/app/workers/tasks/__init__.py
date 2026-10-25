"""Task registry: job type -> function(db, job) -> result dict."""

from collections.abc import Callable

from sqlalchemy.orm import Session

from app.models import Job
from app.models.enums import JobType

TaskFn = Callable[[Session, Job], dict | None]
TASKS: dict[JobType, TaskFn] = {}


def task(job_type: JobType) -> Callable[[TaskFn], TaskFn]:
    def register(fn: TaskFn) -> TaskFn:
        TASKS[job_type] = fn
        return fn

    return register


def load_all() -> None:
    from app.workers.tasks import inference, ingest  # noqa: F401  (registers tasks)
