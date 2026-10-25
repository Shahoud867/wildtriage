"""Request/response models. The frontend's TypeScript types are generated from the OpenAPI schema."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    JobStatus,
    JobType,
    MediaKind,
    MediaStatus,
    Modality,
    ReviewAction,
    ReviewStatus,
    Role,
    TriageOutcome,
)


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(ORM):
    id: int
    email: str
    name: str
    role: Role


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    settings: dict = Field(default_factory=lambda: {"alpha": 0.05, "review_budget": 0.1})


class ProjectOut(ORM):
    id: int
    name: str
    description: str
    settings: dict
    created_at: datetime | None = None


class DeploymentIn(BaseModel):
    project_id: int
    name: str
    modality: Modality = Modality.camera_trap
    camera_model: str = ""
    latitude: float | None = None
    longitude: float | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None


class DeploymentOut(ORM):
    id: int
    project_id: int
    name: str
    modality: Modality
    camera_model: str
    latitude: float | None
    longitude: float | None
    start_at: datetime | None
    end_at: datetime | None


class DatasetIn(BaseModel):
    project_id: int
    name: str
    modality: Modality = Modality.camera_trap
    source: str = "upload"
    license: str = "unknown"


class DatasetOut(ORM):
    id: int
    project_id: int
    name: str
    modality: Modality
    source: str
    license: str


class UploadResult(BaseModel):
    filename: str
    ok: bool
    media_id: int | None = None
    code: str | None = None  # UNSUPPORTED_TYPE | TOO_LARGE | EMPTY_FILE | DUPLICATE | UNSUPPORTED_MODALITY
    message: str | None = None


class UploadOut(BaseModel):
    results: list[UploadResult]
    job_id: int | None = None


class MediaOut(ORM):
    id: int
    dataset_id: int
    deployment_id: int | None
    kind: MediaKind
    original_filename: str
    sha256: str
    size_bytes: int
    width: int | None
    height: int | None
    captured_at: datetime | None
    is_night: bool | None
    quality: dict
    status: MediaStatus
    error: str | None


class JobIn(BaseModel):
    type: JobType
    queue: str = "cpu"
    project_id: int | None = None
    params: dict = Field(default_factory=dict)
    priority: int = 0


class JobOut(ORM):
    id: int
    project_id: int | None
    type: JobType
    queue: str
    status: JobStatus
    attempts: int
    progress: float
    total_items: int
    done_items: int
    params: dict
    result: dict
    error: str | None
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None


class DetectionOut(ORM):
    id: int
    category: str
    x: float
    y: float
    w: float
    h: float
    score: float
    label: str | None = None
    calibrated_score: float | None = None
    topk: list = Field(default_factory=list)


class ReviewItemOut(BaseModel):
    task_id: int
    media_id: int
    detection_id: int | None
    priority: float
    decision: TriageOutcome
    reasons: list[str]
    label: str | None
    calibrated_score: float | None
    topk: list


class ReviewDecisionIn(BaseModel):
    action: ReviewAction
    new_label: str | None = None
    new_bbox: dict | None = None
    comment: str = ""
    time_spent_ms: int = 0


class ReviewDecisionOut(ORM):
    id: int
    review_task_id: int
    action: ReviewAction
    new_label: str | None
    created_at: datetime | None


class ReviewTaskStatusOut(BaseModel):
    task_id: int
    status: ReviewStatus


class ModelVersionOut(ORM):
    id: int
    name: str
    version: str
    task: str
    license: str
    commercial_ok: bool
    is_active: bool
    metrics: dict


class EvaluationRunOut(ORM):
    id: int
    name: str
    task: str
    model_version_id: int | None
    dataset_version_id: int | None
    metrics: dict
    git_commit: str | None
    created_at: datetime | None


class SummaryOut(BaseModel):
    media_by_status: dict[str, int]
    triage: dict[str, int]
    open_reviews: int
    species_counts: dict[str, int]  # all predictions, including unreviewed
    accepted_species_counts: dict[str, int]  # auto-accepted + human-confirmed/relabelled
